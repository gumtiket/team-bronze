import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from ai.llm import AnthropicClient, LLMError, SchemaValidationError
from ai.llm.anthropic import AnthropicCallError

ENV = {
    "ANTHROPIC_MODEL_ID_STRONG": "claude-sonnet-5-5",
    "ANTHROPIC_MODEL_ID_FAST": "claude-haiku-5-5",
    "ANTHROPIC_REFUSAL_FALLBACK": "off",
}


class Answer(BaseModel):
    ok: bool


def response(text='{"ok":true}', model="claude-haiku-5-5", **overrides):
    return {
        "content": [
            {"type": "thinking", "thinking": "dummy-secret-do-not-use"},
            {"type": "text", "text": text},
        ],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        **overrides,
    }


def sdk(payload=None):
    messages = Mock()
    messages.create.return_value = payload if payload is not None else response()
    return SimpleNamespace(
        messages=messages, beta=SimpleNamespace(messages=messages), credentials=object()
    )


def complete(client, **kwargs):
    return client.complete("rules", "request", tier="fast", stage="check", **kwargs)


def transport_error(name, status=None, retry_after=None):
    # Public SDK exception contracts without making the optional SDK a test dependency.
    error = type(name, (Exception,), {})("dummy-secret-do-not-use request body and headers")
    error.status_code = status
    error.request_id = "req_dummy"
    error.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})
    return error


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5-5", {"output_config": {"effort": "low"}}),
        (
            "claude-sonnet-5-5",
            {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
        ),
        (
            "claude-haiku-5-5",
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
        ),
        ("claude-sonnet-4-6", {"temperature": 0}),
        ("claude-haiku-4-5", {"temperature": 0}),
        ("claude-haiku-4-5-20251001", {"temperature": 0}),
    ],
)
def test_exact_model_parameters_and_text_extraction(model, expected):
    injected = sdk(response(model=model))
    llm = AnthropicClient(client=injected, environ={**ENV, "ANTHROPIC_MODEL_ID_FAST": model})
    traces = []
    with llm.observing(traces.append):
        result = complete(llm, schema=Answer)
    request = injected.messages.create.call_args.kwargs
    assert request == {
        "model": model,
        "system": request["system"],
        "messages": [{"role": "user", "content": "request"}],
        "max_tokens": 16000,
        **expected,
    }
    assert result.parsed.ok
    assert result.model_dump(mode="json")["parsed"] == {"ok": True}
    assert "dummy-secret" not in traces[0].model_dump_json()
    assert traces[0].request_parameters == {"max_tokens": 16000, **expected}


def test_response_model_cache_usage_and_cost_are_preserved():
    payload = response(
        model="claude-haiku-4-5-20251001",
        usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 3},
            "untrusted_sdk_field": "dummy-secret-do-not-use",
        },
    )
    result = complete(AnthropicClient(client=sdk(payload), environ=ENV))
    assert result.model_id == "claude-haiku-4-5-20251001"
    assert result.usage["cache_read_input_tokens"] == 10
    assert "untrusted_sdk_field" not in result.usage
    assert result.cost_usd == pytest.approx((100 + 20 * 5 + 10 * 0.1 + 2 * 1.25 + 3 * 2) / 1e6)


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_beta_default_fallback_records_all_attempts_without_inventing_refusal_cost(model):
    payload = response(model="claude-sonnet-4-6")
    payload["content"].insert(
        0,
        {
            "type": "fallback",
            "from": {"model": model},
            "to": {"model": "claude-sonnet-4-6"},
        },
    )
    payload["usage"]["iterations"] = [
        {"type": "message", "model": model, "input_tokens": 100, "output_tokens": 0},
        {
            "type": "fallback_message",
            "model": "claude-sonnet-4-6",
            "input_tokens": 100,
            "output_tokens": 20,
        },
    ]
    injected = sdk(payload)
    llm = AnthropicClient(
        client=injected,
        environ={
            **ENV,
            "ANTHROPIC_MODEL_ID_FAST": model,
            "ANTHROPIC_REFUSAL_FALLBACK": "default",
        },
    )
    result = complete(llm)
    assert injected.beta.messages.create.call_args.kwargs["fallbacks"] == "default"
    assert result.model_id == "claude-sonnet-4-6"
    assert result.usage["fallback_ran"] and result.usage["served_by_fallback"]
    report = llm.tracker.report()
    assert [call.model_id for call in report.calls] == [model, "claude-sonnet-4-6"]
    assert report.total.input_tokens == 200 and report.total.output_tokens == 20
    assert result.cost_usd is None and report.total.cost_usd is None
    assert report.total.known_cost_usd == pytest.approx(0.0006)


def test_iterations_without_a_model_are_attributed_instead_of_rejected():
    # The API began returning usage.iterations entries with model=None (seen on the live server).
    payload = response(model="claude-sonnet-5-5")
    payload["usage"]["iterations"] = [
        {"type": "message", "model": None, "input_tokens": 100, "output_tokens": 20}
    ]
    llm = AnthropicClient(
        client=sdk(payload),
        environ={**ENV, "ANTHROPIC_MODEL_ID_FAST": "claude-sonnet-5-5",
                 "ANTHROPIC_REFUSAL_FALLBACK": "default"},
    )
    result = complete(llm)
    assert result.usage["iterations"][0]["model"] == "claude-sonnet-5-5"
    assert [call.model_id for call in llm.tracker.report().calls] == ["claude-sonnet-5-5"]


def test_a_fallback_iteration_without_a_model_uses_the_serving_model():
    payload = response(model="claude-sonnet-4-6")
    payload["usage"]["iterations"] = [
        {"type": "message", "input_tokens": 100, "output_tokens": 0},
        {"type": "fallback_message", "model": None, "input_tokens": 100, "output_tokens": 20},
    ]
    llm = AnthropicClient(
        client=sdk(payload),
        environ={**ENV, "ANTHROPIC_MODEL_ID_FAST": "claude-sonnet-5-5",
                 "ANTHROPIC_REFUSAL_FALLBACK": "default"},
    )
    complete(llm)
    assert [call.model_id for call in llm.tracker.report().calls] == [
        "claude-sonnet-5-5", "claude-sonnet-4-6"]


def test_haiku_never_sends_server_fallback():
    llm = AnthropicClient(client=sdk(), environ={**ENV, "ANTHROPIC_REFUSAL_FALLBACK": "default"})
    assert "fallbacks" not in llm.request_parameters("fast")


@pytest.mark.parametrize("reason", ["max_tokens", "refusal"])
def test_truncation_and_refusal_do_not_retry_or_leak_explanation(reason):
    payload = response(
        stop_reason=reason,
        stop_details={
            "category": "cyber",
            "explanation": "dummy-secret-do-not-use",
        },
    )
    injected = sdk(payload)
    llm = AnthropicClient(client=injected, environ=ENV)
    traces = []
    with llm.observing(traces.append), pytest.raises(LLMError):
        complete(llm, schema=Answer)
    assert injected.messages.create.call_count == 1
    assert "dummy-secret" not in traces[0].model_dump_json()
    if reason == "refusal":
        assert traces[0].response_text == ""
        assert traces[0].usage["refusal_category"] == "cyber"


@pytest.mark.parametrize(
    ("name", "status", "retry"),
    [
        ("RateLimitError", 429, True),
        ("InternalServerError", 500, True),
        ("InternalServerError", 529, True),
        ("APIConnectionError", None, True),
        ("APITimeoutError", None, False),
        ("BadRequestError", 400, False),
        ("AuthenticationError", 401, False),
        ("PermissionDeniedError", 403, False),
        ("NotFoundError", 404, False),
    ],
)
def test_retry_policy_and_safe_errors(name, status, retry):
    injected = sdk()
    injected.messages.create.side_effect = [transport_error(name, status), response()]
    delays, traces = [], []
    llm = AnthropicClient(client=injected, environ=ENV, sleep=delays.append)
    with llm.observing(traces.append):
        if retry:
            assert complete(llm).transport_attempts == 2
        else:
            with pytest.raises(AnthropicCallError) as error:
                complete(llm)
            assert error.value.details == {"status_code": status, "request_id": "req_dummy"}
            assert error.value.__suppress_context__
            assert "dummy-secret" not in str(error.value)
    assert injected.messages.create.call_count == (2 if retry else 1)
    assert delays == ([1] if retry else [])
    assert "dummy-secret" not in json.dumps([trace.model_dump(mode="json") for trace in traces])


def test_retry_after_is_honored_and_attempts_are_bounded():
    injected = sdk()
    injected.messages.create.side_effect = transport_error("RateLimitError", 429, "4")
    delays = []
    llm = AnthropicClient(client=injected, environ=ENV, sleep=delays.append)
    with pytest.raises(AnthropicCallError) as error:
        complete(llm)
    assert error.value.transport_attempts == 3
    assert delays == [4, 4]
    assert llm.tracker.report().total.unobserved_requests == 1


def test_long_retry_after_does_not_retry_early():
    injected = sdk()
    injected.messages.create.side_effect = transport_error("RateLimitError", 429, "120")
    delays = []
    with pytest.raises(AnthropicCallError):
        complete(AnthropicClient(client=injected, environ=ENV, sleep=delays.append))
    assert injected.messages.create.call_count == 1 and delays == []


@pytest.mark.parametrize(
    "patch",
    [
        {"ANTHROPIC_MODEL_ID_FAST": "unknown"},
        {"ANTHROPIC_MODEL_ID_FAST": ""},
        {"ANTHROPIC_EFFORT_FAST": "max"},
        {"ANTHROPIC_REFUSAL_FALLBACK": "unknown"},
    ],
)
def test_invalid_configuration_fails_before_sdk_call(patch):
    injected = sdk()
    with pytest.raises(ValueError):
        AnthropicClient(client=injected, environ={**ENV, **patch})
    injected.messages.create.assert_not_called()


def test_opus_small_budget_is_rejected():
    with pytest.raises(ValueError, match="16000"):
        AnthropicClient(
            environ={**ENV, "ANTHROPIC_MODEL_ID_STRONG": "claude-opus-5-5"}, max_tokens=4096
        )


def test_missing_model_environment():
    with pytest.raises(ValueError, match="ANTHROPIC_MODEL_ID_STRONG"):
        AnthropicClient(environ={})


def test_schema_retries_preserve_each_request_usage():
    injected = sdk()
    injected.messages.create.side_effect = [response("broken")] * 3
    llm = AnthropicClient(client=injected, environ=ENV)
    with pytest.raises(SchemaValidationError):
        complete(llm, schema=Answer)
    assert len(llm.tracker.calls) == 3
    assert llm.tracker.report().total.input_tokens == 300


def test_lazy_sdk_uses_default_auth_and_disables_nested_retries(monkeypatch):
    injected = sdk()
    constructor = Mock(return_value=injected)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(Anthropic=constructor)
    )
    complete(AnthropicClient(environ=ENV))
    constructor.assert_called_once_with(max_retries=0, timeout=60.0)


def test_missing_optional_sdk_is_safe_and_injected_client_still_works(monkeypatch):
    def missing(name):
        raise ImportError("dummy-secret-do-not-use")

    monkeypatch.setattr(importlib, "import_module", missing)
    llm = AnthropicClient(environ=ENV)
    with pytest.raises(AnthropicCallError, match="MissingSDK"):
        complete(llm)
    assert llm.tracker.report().total.unobserved_requests == 0
    assert complete(AnthropicClient(client=sdk(), environ=ENV)).text


def test_missing_credentials_fails_before_transmission(monkeypatch):
    injected = sdk()
    injected.credentials = None
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(Anthropic=lambda **kwargs: injected),
    )
    llm = AnthropicClient(environ=ENV)
    with pytest.raises(AnthropicCallError, match="MissingCredentials") as error:
        complete(llm)
    assert error.value.transport_attempts == 0
    injected.messages.create.assert_not_called()
    assert llm.tracker.report().total.unobserved_requests == 0


def test_unknown_refusal_category_does_not_become_free():
    llm = AnthropicClient(
        client=sdk(
            response(
                stop_reason="refusal",
                stop_details={"category": "unknown-future-category"},
                usage={"input_tokens": 100, "output_tokens": 0},
            )
        ),
        environ=ENV,
    )
    with pytest.raises(LLMError):
        complete(llm)
    assert llm.tracker.calls[0].cost_usd is None


@pytest.mark.parametrize(
    "payload",
    [
        response(usage={}),
        response(usage={"input_tokens": -1, "output_tokens": 1}),
        response(model="dummy-secret-do-not-use"),
        response(content="invalid"),
    ],
)
def test_invalid_response_is_unknown_and_never_leaked(payload):
    llm = AnthropicClient(client=sdk(payload), environ=ENV)
    with pytest.raises(AnthropicCallError, match="InvalidResponse"):
        complete(llm)
    assert llm.tracker.report().total.unobserved_requests == 1


def test_long_haiku_price_boundary():
    for tokens, expected in [(100000, 0.01001), (100001, 0.0500505)]:
        result = complete(
            AnthropicClient(
                client=sdk(
                    response(
                        usage={
                            "input_tokens": tokens,
                            "output_tokens": 20,
                        }
                    )
                ),
                environ=ENV,
            )
        )
        assert result.cost_usd == pytest.approx(expected)


def test_cli_analysis_and_artifact_clients_are_injected(tmp_path, monkeypatch):
    from ai.cli import main

    injected = AnthropicClient(client=sdk(), environ=ENV)
    run = Mock(
        return_value=SimpleNamespace(
            status="diagnosed",
            diagnosis=SimpleNamespace(enrichment_status="completed"),
            build_context=None,
        )
    )
    monkeypatch.setattr("ai.cli.AnthropicClient", lambda **kwargs: injected)
    monkeypatch.setattr("ai.cli.run_analysis", run)
    assert (
        main(
            [
                "analyze",
                str(tmp_path),
                "--llm",
                "anthropic",
                "--artifact-llm",
                "anthropic",
                "--no-gate",
            ]
        )
        == 0
    )
    assert all(
        run.call_args.kwargs[key] is injected
        for key in ("llm", "artifact_llm", "decision_llm", "repair_llm")
    )


def test_anthropic_cli_runs_real_pipeline_with_fake_sdk_and_protects_input(tmp_path, monkeypatch):
    from ai.cli import main
    from ai.llm.fake import recommendation_response

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n"
        '@app.get("/healthz")\ndef health():\n    return {"ok": True}\n'
    )
    (repo / "requirements.txt").write_text("fastapi==0.115.12\nuvicorn==0.34.2\n")
    original = {path.name: path.read_bytes() for path in repo.iterdir()}
    injected = sdk()

    def create(**kwargs):
        text = (
            recommendation_response(kwargs["system"], kwargs["messages"][0]["content"])
            if '"selected_set"' in kwargs["system"]
            else '{"explanations":[],"candidates":[]}'
        )
        return response(text, model=kwargs["model"])

    injected.messages.create.side_effect = create
    monkeypatch.setattr(
        "ai.cli.AnthropicClient", lambda: AnthropicClient(client=injected, environ=ENV)
    )
    output = tmp_path / "out"
    assert (
        main(
            [
                "analyze",
                str(repo),
                "--llm",
                "anthropic",
                "--no-gate",
                "--save-llm-trace",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert {path.name: path.read_bytes() for path in repo.iterdir()} == original
    trace = json.loads((output / "llm-trace.json").read_text())
    assert trace["summary"]["client_kind"] == "AnthropicClient"
    assert {exchange["stage"] for exchange in trace["exchanges"]} == {"diagnose", "recommend"}
    assert all(exchange["status"] == "parsed" for exchange in trace["exchanges"])
    gate = json.loads((output / "gate-report.json").read_text())
    assert gate["status"] == "skipped" and not gate["pr_eligible"]
    assert json.loads((output / "cost.json").read_text())["total"]["calls"] == 2


def test_cli_record_selects_provider_without_touching_fixtures(tmp_path, monkeypatch):
    from ai.cli import main

    backend = AnthropicClient(client=sdk(), environ=ENV)
    recorder = Mock()
    run = Mock(
        return_value=SimpleNamespace(
            status="diagnosed",
            diagnosis=SimpleNamespace(enrichment_status="completed"),
            build_context=None,
        )
    )
    monkeypatch.setattr("ai.cli.AnthropicClient", lambda: backend)
    monkeypatch.setattr("ai.cli.RecordingClient", recorder)
    monkeypatch.setattr("ai.cli.run_analysis", run)
    assert (
        main(
            [
                "analyze",
                str(tmp_path),
                "--llm",
                "record",
                "--llm-provider",
                "anthropic",
                "--llm-fixtures",
                str(tmp_path / "fixtures"),
            ]
        )
        == 0
    )
    assert recorder.call_args.args[0] is backend


def test_cli_single_check_has_one_attempt_and_reports_cost(tmp_path, monkeypatch):
    from ai.cli import main

    injected = sdk()
    factory = Mock(
        side_effect=lambda **kwargs: AnthropicClient(
            client=injected,
            environ=ENV,
            **kwargs,
        )
    )
    monkeypatch.setattr("ai.cli.AnthropicClient", factory)
    assert main(["llm-check", "--out", str(tmp_path)]) == 0
    factory.assert_called_once_with(schema_retries=0, transport_attempts=1)
    assert injected.messages.create.call_count == 1
    report = json.loads((tmp_path / "llm-check.json").read_text())
    assert report["status"] == "success" and report["model_id"] == "claude-haiku-5-5"
    assert report["cost_usd"] == pytest.approx(0.00002)
    assert "dummy-secret" not in json.dumps(report)


def test_cli_single_check_failure_is_nonzero_and_safe(tmp_path, monkeypatch):
    from ai.cli import main

    injected = sdk()
    injected.messages.create.side_effect = transport_error("AuthenticationError", 401)
    monkeypatch.setattr(
        "ai.cli.AnthropicClient",
        lambda **kwargs: AnthropicClient(
            client=injected,
            environ=ENV,
            **kwargs,
        ),
    )
    assert main(["llm-check", "--out", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "llm-check.json").read_text())
    assert report["error_code"] == "AuthenticationError"
    assert injected.messages.create.call_count == 1
    assert "dummy-secret" not in json.dumps(report)
