"""Optional first-party transport; SDK credentials never enter our output models."""

import importlib
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from ai.llm.base import LLMError, LLMResult, Tier, ValidatingClient
from ai.llm.cost import CostTracker

MODELS = frozenset(
    {
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
    }
)
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
CATEGORIES = frozenset({"cyber", "bio", "frontier_llm", "reasoning_extraction", "general_harms"})


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def token_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("invalid token count")
    return value


def response_model(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"claude-[a-z0-9-]{1,100}", value):
        raise ValueError("invalid response model")
    return value


def safe_usage(value: Any, iteration_models: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Whitelist numeric usage, including cache fields; omit free-form SDK metadata."""
    usage = {name: token_count(field(value, name, 0)) for name in TOKEN_FIELDS}
    cache = field(value, "cache_creation")
    if cache is not None:
        usage["cache_creation"] = {
            name: token_count(field(cache, name, 0))
            for name in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
        }
    iterations = field(value, "iterations")
    if iterations:
        if not isinstance(iterations, list) or len(iterations) > 4:
            raise ValueError("invalid usage iterations")
        usage["iterations"] = []
        for item in iterations:
            kind = field(item, "type")
            if kind not in {"message", "fallback_message"}:
                raise ValueError("invalid iteration type")
            usage["iterations"].append(
                {
                    **safe_usage_counts(item),
                    "type": kind,
                    # The API may omit an iteration's model; the first attempt always runs
                    # the requested model and a fallback is the one that served the response.
                    "model": response_model(
                        field(item, "model") or (iteration_models or {}).get(kind)
                    ),
                }
            )
    return usage


def safe_usage_counts(value: Any) -> dict[str, Any]:
    # Iterations have the same cache/token fields, without recursive iterations.
    return safe_usage(
        {name: field(value, name, 0) for name in TOKEN_FIELDS}
        | {"cache_creation": field(value, "cache_creation")}
    )


class AnthropicCallError(LLMError):
    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        transport_attempts: int = 0,
        request_id: str | None = None,
    ) -> None:
        self.code = code
        self.details = {"status_code": status_code, "request_id": request_id}
        super().__init__(f"Anthropic 호출 실패 ({code})", transport_attempts=transport_attempts)


class AnthropicClient(ValidatingClient):
    def __init__(
        self,
        *,
        client: Any = None,
        environ: Mapping[str, str] | None = None,
        tracker: CostTracker | None = None,
        schema_retries: int = 2,
        transport_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_tokens: int = 16000,
    ) -> None:
        env = os.environ if environ is None else environ
        required = ("ANTHROPIC_MODEL_ID_STRONG", "ANTHROPIC_MODEL_ID_FAST")
        missing = [name for name in required if not env.get(name, "").strip()]
        if missing:
            raise ValueError("필수 환경변수 누락: " + ", ".join(missing))
        self.models = {
            tier: env[f"ANTHROPIC_MODEL_ID_{tier.upper()}"].strip() for tier in ("strong", "fast")
        }
        if any(model not in MODELS for model in self.models.values()):
            raise ValueError("지원하지 않는 Anthropic 모델 ID입니다. 공식 ID를 확인하세요.")
        if type(transport_attempts) is not int or not 1 <= transport_attempts <= 5:
            raise ValueError("전송 시도 상한은 1~5회입니다.")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens는 양의 정수여야 합니다.")
        if "claude-opus-5-5" in self.models.values() and max_tokens < 16000:
            raise ValueError("Opus 5.5의 max_tokens는 16000 이상이어야 합니다.")
        self.efforts = {
            tier: env.get(f"ANTHROPIC_EFFORT_{tier.upper()}", default).strip()
            for tier, default in (("strong", "medium"), ("fast", "low"))
        }
        fallback = env.get("ANTHROPIC_REFUSAL_FALLBACK", "default").strip()
        if fallback not in {"default", "off"}:
            raise ValueError("ANTHROPIC_REFUSAL_FALLBACK은 default 또는 off여야 합니다.")
        self.refusal_fallback = fallback == "default"
        self.max_tokens = max_tokens
        for tier in ("strong", "fast"):
            self.request_parameters(tier)
        if tracker is None:
            pricing = Path(__file__).resolve().parents[3] / "pricing.json"
            tracker = CostTracker.from_file(pricing) if pricing.is_file() else CostTracker()
        super().__init__(tracker=tracker, schema_retries=schema_retries)
        self.client = client
        self.transport_attempts = transport_attempts
        self.sleep = sleep
        self.clock = clock

    def request_parameters(self, tier: Tier) -> dict[str, Any]:
        model = self.models[tier]
        request: dict[str, Any] = {"max_tokens": self.max_tokens}
        if model in {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}:
            effort = self.efforts[tier]
            allowed = {"low", "medium", "high"}
            if model == "claude-opus-5-5":
                allowed |= {"xhigh", "max"}
            if effort not in allowed:
                raise ValueError("모델의 thinking 설정과 호환되지 않는 Anthropic effort입니다.")
            request["output_config"] = {"effort": effort}
            if model != "claude-opus-5-5":
                request["thinking"] = {
                    "type": "between_tools" if model == "claude-sonnet-5-5" else "disabled"
                }
            if self.refusal_fallback and model != "claude-haiku-5-5":
                request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            request["temperature"] = 0
        return request

    @staticmethod
    def _error(error: Exception, transmissions: int) -> tuple[AnthropicCallError, bool]:
        # Match the SDK's public exception hierarchy without importing the optional
        # SDK for injected fake clients. Timeout is a subclass of connection error.
        names = {kind.__name__ for kind in type(error).__mro__}
        codes = (
            "APITimeoutError",
            "RateLimitError",
            "AuthenticationError",
            "PermissionDeniedError",
            "BadRequestError",
            "NotFoundError",
            "InternalServerError",
            "APIConnectionError",
            "APIStatusError",
            "APIError",
        )
        code = next((name for name in codes if name in names), "ClientConfigurationError")
        status = getattr(error, "status_code", None)
        status = status if type(status) is int and 100 <= status <= 599 else None
        request_id = getattr(error, "request_id", None)
        if not isinstance(request_id, str) or not re.fullmatch(
            r"req_[A-Za-z0-9_-]{1,100}", request_id
        ):
            request_id = None
        retry = code != "APITimeoutError" and (
            status == 429 or (status is not None and status >= 500) or code == "APIConnectionError"
        )
        return AnthropicCallError(
            code, status_code=status, transport_attempts=transmissions, request_id=request_id
        ), retry

    @staticmethod
    def _delay(error: Exception, attempt: int) -> float:
        headers = field(getattr(error, "response", None), "headers", {})
        retry_after = headers.get("retry-after")
        if isinstance(retry_after, str):
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
            except ValueError:
                try:
                    date = parsedate_to_datetime(retry_after)
                    return max(0, (date - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return float(2**attempt)

    def _invoke(self, system: str, user: str, *, tier: Tier, stage: str) -> LLMResult:
        started = self.clock()
        if self.client is None:
            try:
                sdk = importlib.import_module("anthropic")
            except ImportError:
                raise AnthropicCallError("MissingSDK") from None
            try:
                self.client = sdk.Anthropic(max_retries=0, timeout=60.0)
            except Exception:
                raise AnthropicCallError("ClientConfigurationError") from None
            if not any(
                field(self.client, name) for name in ("api_key", "auth_token", "credentials")
            ):
                # The SDK resolves env/profile credentials at construction, but
                # otherwise raises during request preparation before any send.
                self.client = None
                raise AnthropicCallError("MissingCredentials") from None
        parameters = self.request_parameters(tier)
        messages = self.client.beta.messages if "betas" in parameters else self.client.messages
        for attempt in range(self.transport_attempts):
            try:
                response = messages.create(
                    model=self.models[tier],
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    **parameters,
                )
                break
            except Exception as error:
                safe, retry = self._error(error, attempt + 1)
                delay = self._delay(error, attempt) if retry else 0
                # Long Retry-After is returned to the caller; never retry too early.
                if not retry or attempt + 1 == self.transport_attempts or delay > 60:
                    raise safe from None
                self.sleep(delay)
        try:
            raw_usage = field(response, "usage")
            if (
                field(raw_usage, "input_tokens") is None
                or field(raw_usage, "output_tokens") is None
            ):
                raise ValueError("missing usage")
            usage = safe_usage(
                raw_usage,
                {"message": self.models[tier], "fallback_message": field(response, "model")},
            )
            usage["provider"] = "anthropic"
            usage["requested_model"] = self.models[tier]
            content = field(response, "content")
            if not isinstance(content, list):
                raise ValueError("invalid content")
            fallback = any(field(block, "type") == "fallback" for block in content) or any(
                item["type"] == "fallback_message" for item in usage.get("iterations", [])
            )
            usage["fallback_ran"] = fallback
            stop_reason = field(response, "stop_reason")
            if stop_reason not in {
                "end_turn",
                "max_tokens",
                "stop_sequence",
                "tool_use",
                "pause_turn",
                "refusal",
                "model_context_window_exceeded",
            }:
                raise ValueError("invalid stop reason")
            usage["served_by_fallback"] = fallback and stop_reason != "refusal"
            category = field(field(response, "stop_details"), "category")
            if stop_reason == "refusal":
                usage["refusal_category"] = (
                    category if category is None or category in CATEGORIES else "unknown"
                )
            text = "".join(
                field(block, "text") for block in content if field(block, "type") == "text"
            )
            # Partial refused output must never be consumed or persisted as a result.
            if stop_reason == "refusal":
                text = ""
            return LLMResult(
                text=text,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                model_id=response_model(field(response, "model")),
                latency_s=max(0, self.clock() - started),
                transport_attempts=attempt + 1,
                stop_reason=stop_reason,
                usage=usage,
            )
        except (TypeError, ValueError, AttributeError):
            raise AnthropicCallError("InvalidResponse", transport_attempts=attempt + 1) from None
