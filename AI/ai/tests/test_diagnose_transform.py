import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from ai.detectors import RepoView, detect, detect_framework, detect_signals
from ai.diagnose.service import FACTOR_OWNERS, enrich
from ai.llm import FakeLLMClient, LLMResult
from ai.models import Diagnosis
from ai.pipeline import run_analysis
from ai.security import SourceMasker
from ai.transform import llm_patch, propose
from ai.transform.service import _check_semantics, identify_sample
from ai.transform.templates import template_changes
from ai.transform.workspace import Workspace, make_diff, patch_paths, source_files

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def diagnosis(repo):
    framework = detect_framework(repo)
    return Diagnosis(
        status="completed",
        support_grade=framework.support_grade,
        framework=framework,
        violations=detect(repo),
        signals=detect_signals(repo),
        warnings=list(repo.warnings),
    )


def small_repo(tmp_path, text):
    (tmp_path / "main.py").write_text(text)
    return RepoView(tmp_path)


def test_factor_table_uses_exact_owners_and_only_b_factors():
    repo = RepoView(SAMPLES / "todo")
    result = enrich(diagnosis(repo), repo, None, SourceMasker(repo))
    assert len(result.factor_reviews) == 12
    for row in result.factor_reviews:
        assert row.owner == FACTOR_OWNERS[row.factor - 1][1]
        if row.factor not in {2, 3, 4, 6, 7, 11}:
            assert row.status == "n/a"
    assert next(r for r in result.factor_reviews if r.factor == 4).status == "violation"


def test_llm_cannot_delete_rule_findings_change_factors_or_signals():
    repo = RepoView(SAMPLES / "todo-scheduler")
    original = diagnosis(repo)
    first = original.violations[0]
    response = {
        "explanations": [{"id": first.id, "description": "설명", "impact": "영향", "factor": 1}],
        "candidates": [
            {
                "factor": 5,
                "file": "app/main.py",
                "line": 1,
                "evidence": "예시",
                "description": "범위 밖",
            },
            {
                "factor": 3,
                "file": "../outside.py",
                "line": 1,
                "evidence": "예시",
                "description": "잘못된 파일",
            },
            {
                "factor": 3,
                "file": "app/main.py",
                "line": 1,
                "evidence": repo.read("app/main.py").splitlines()[0],
                "description": "추가 확인",
            },
        ],
    }
    fake = FakeLLMClient([json.dumps(response)])
    result = enrich(original, repo, fake, SourceMasker(repo))
    assert result.violations[0].factor == first.factor
    assert result.violations[0].description == "설명"
    assert {v.id for v in original.violations} <= {v.id for v in result.violations}
    assert result.signals == original.signals
    assert {v.id for v in original.violations} == {v.id for v in result.violations}
    assert result.review_candidates[-1].source == "llm"
    assert result.review_candidates[-1].confidence == "needs_review"
    assert "dummy-secret-do-not-use" not in fake.calls[0].user
    assert "VIOLATIONS.json" not in fake.calls[0].user


def test_enrichment_schema_failure_falls_back_without_leaking_source():
    repo = RepoView(SAMPLES / "todo")
    fake = FakeLLMClient(['{"support_grade":"unsupported"}'] * 3)
    result = enrich(diagnosis(repo), repo, fake, SourceMasker(repo))
    assert result.enrichment_status == "failed" and len(result.violations) == 6
    assert result.support_grade == "supported"


@pytest.mark.parametrize("name", ["todo", "todo-scheduler"])
def test_templates_produce_valid_cumulative_diff_and_preserve_models(name, monkeypatch):
    repo = RepoView(SAMPLES / name)
    before = source_files(repo)
    diff, report = propose(repo, diagnosis(repo))
    assert report.patch_valid and report.compile_passed
    assert len(report.addressed_ids) >= 3
    assert report.sample_name == name
    assert report.needs_approval
    assert {v.name for v in report.env_vars} == {"DATABASE_URL", "SECRET_KEY", "PORT", "LOG_LEVEL"}
    assert report.migrate_command == "python -m app.migrate"
    assert "data_migration_unsupported" in {w.code for w in report.warnings}
    assert "local_file_write" in ",".join(report.deferred_ids)
    assert "unpinned_dependency" in ",".join(report.deferred_ids)
    with Workspace(before) as workspace:
        workspace.apply(diff)
        after = workspace.read(set(before) | {"app/migrate.py"})
        assert "logging.StreamHandler(sys.stdout)" in after["app/main.py"]
        assert 'os.environ.get("PORT", "8080")' in after["app/main.py"]
        assert '@app.get("/healthz")' in after["app/main.py"]
        assert "psycopg2-binary" in after["requirements.txt"]
        old_classes = [
            ast.dump(n)
            for n in ast.walk(ast.parse(before["app/db.py"]))
            if isinstance(n, ast.ClassDef)
        ]
        new_classes = [
            ast.dump(n)
            for n in ast.walk(ast.parse(after["app/db.py"]))
            if isinstance(n, ast.ClassDef)
        ]
        assert old_classes == new_classes
        assert workspace.compile()
        # Execute only the owned sample's temporary db module; engine creation does not connect.
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql://dummy:dummy-secret-do-not-use@db.invalid/sample"
        )
        module_spec = importlib.util.spec_from_file_location(
            "transformed_sample_db", workspace.root / "app/db.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        try:
            assert module.engine.dialect.driver == "psycopg2"
            assert module.engine.dialect.dbapi.__name__ == "psycopg2"
        finally:
            module.engine.dispose()


@pytest.mark.parametrize(
    "declaration,added",
    [
        ("psycopg[binary]>=3", True),
        ("# psycopg2-binary is not installed", True),
        ('psycopg2-binary; python_version < "3.10"', True),
        ("psycopg2-binary==2.9.13", False),
        ("psycopg2>=2.9", False),
    ],
)
def test_driver_dependency_is_added_even_when_psycopg3_or_a_comment_exists(declaration, added):
    repo = RepoView(SAMPLES / "todo")
    before = source_files(repo)
    original = before["requirements.txt"].rstrip() + "\n" + declaration + "\n"
    before["requirements.txt"] = original
    after, _ = template_changes(before, diagnosis(repo), SourceMasker(repo), None)
    assert after["requirements.txt"] == original + ("psycopg2-binary\n" if added else "")


@pytest.mark.parametrize(
    "declaration",
    [
        "psycopg2-binary-unrelated",
        "psycopg[binary]",
        "psycopg2-binary @ https://example.invalid/driver.whl",
        'psycopg2-binary; python_version < "3.10"',
    ],
)
def test_patch_guard_rejects_driver_prefix_impersonation_and_alternative_sources(declaration):
    with pytest.raises(ValueError, match="dependency_change_not_allowed"):
        _check_semantics(
            {"requirements.txt": "fastapi\n"},
            {"requirements.txt": "fastapi\n" + declaration + "\n"},
            {"requirements.txt"},
        )


def test_non_dummy_secret_file_is_masked_and_deferred(tmp_path):
    # Clearly fake fixture, deliberately different from the one allowed sample sentinel.
    fake_value = "dummy-test-token-not-real"
    repo = small_repo(
        tmp_path, 'from fastapi import FastAPI\napp=FastAPI()\nSECRET_KEY="' + fake_value + '"\n'
    )
    masker = SourceMasker(repo)
    assert "main.py" in masker.blocked_files
    assert fake_value not in masker.summaries(repo)["main.py"]
    diff, report = propose(repo, diagnosis(repo))
    assert "main.py" not in report.changed_files
    assert "secret_file_deferred" in {w.code for w in report.warnings}
    assert fake_value not in report.model_dump_json()


def test_masking_multiline_unicode_defaults_and_dict_literals(tmp_path):
    repo = small_repo(
        tmp_path,
        'import os\nSECRET_KEY="""dummy-test-value-not-real\n가짜"""\n'
        'config={"password":"dummy-test-password-not-real"}\n'
        'key=os.getenv("API_KEY","dummy-test-key-not-real")\n',
    )
    masker = SourceMasker(repo)
    output = masker.source(repo, "main.py")
    ast.parse(output)
    assert output.count("\n") == repo.read("main.py").count("\n")
    assert "dummy-test" not in output
    assert '"API_KEY"' in output


def test_config_flags_under_secret_keys_are_not_sensitive_values(tmp_path):
    # An already merged AnyShip deploy-spec marks env entries with `secret: true/false`.
    (tmp_path / "deploy-spec.yaml").write_text(
        "env:\n  - name: APP_NAME\n    secret: false\n  - name: DATABASE_URL\n    secret: true\n"
        "token_ttl: 3600\n"
    )
    (tmp_path / "config.yaml").write_text("password: dummy-test-password-not-real\n")
    masker = SourceMasker(RepoView(tmp_path))
    assert not masker.contains_sensitive("websocket: false\nsecret: true\nmax: 3600\n")
    assert masker.contains_sensitive("dummy-test-password-not-real")


def test_package_index_credential_uri_is_not_copied_into_diff(tmp_path):
    shutil.copytree(SAMPLES / "todo", tmp_path / "repo")
    requirements = tmp_path / "repo/requirements.txt"
    dummy_url = "https://dummy-user:dummy-secret-do-not-use@packages.example.invalid/simple"
    with requirements.open("a") as file:
        file.write("\n--extra-index-url " + dummy_url + "\n")
    repo = RepoView(tmp_path / "repo")
    masker = SourceMasker(repo)
    assert "requirements.txt" in masker.blocked_files
    diff, report = propose(repo, diagnosis(repo))
    assert dummy_url not in diff and "b/requirements.txt" not in diff
    assert "driver_dependency_deferred" in {w.code for w in report.warnings}


def test_only_hash_matched_samples_get_migration_entrypoint(tmp_path):
    copy = tmp_path / "todo"
    shutil.copytree(SAMPLES / "todo", copy)
    with (copy / "app/main.py").open("a") as file:
        file.write("\n# different input\n")
    repo = RepoView(copy)
    assert identify_sample(repo) is None
    diff, report = propose(repo, diagnosis(repo))
    assert report.sample_name is None and report.migrate_command is None
    assert "b/app/migrate.py" not in diff


def test_patch_retries_broken_then_success(tmp_path):
    repo = small_repo(
        tmp_path,
        "import os\nimport uvicorn\nfrom fastapi import FastAPI\napp=FastAPI()\n"
        'uvicorn.run("main:app",port=8000)\n',
    )
    before = source_files(repo)
    target = next(v for v in detect(repo) if v.rule == "fixed_port")
    after = {
        "main.py": before["main.py"].replace(
            "port=8000", 'port=int(os.environ.get("PORT", "8080"))'
        )
    }
    good = json.dumps({"diff": make_diff(before, after), "violation_ids": [target.id]})
    bad = json.dumps({"diff": "not a patch", "violation_ids": [target.id]})
    fake = FakeLLMClient([bad, good])
    result, ids, attempts, warnings = llm_patch(before, [target], fake, SourceMasker(repo))
    assert result == after and ids == [target.id] and attempts == 2 and not warnings
    assert "validation_error" in fake.calls[1].user
    assert "patch_non_diff_content" in fake.calls[1].user


def test_patch_exhaustion_tries_individual_target(tmp_path):
    repo = small_repo(tmp_path, 'import os, uvicorn\nuvicorn.run("main:app",port=8000)\n')
    target = next(v for v in detect(repo) if v.rule == "fixed_port")
    fake = FakeLLMClient(
        [json.dumps({"diff": f"bad{i}", "violation_ids": [target.id]}) for i in range(4)]
    )
    result, ids, attempts, warnings = llm_patch(
        source_files(repo), [target], fake, SourceMasker(repo)
    )
    assert result == source_files(repo) and ids == [] and attempts == 4
    assert warnings[0].code == "llm_transform_deferred"


@pytest.mark.parametrize("name", ["../escape.py", ".git/config", "/absolute.py", ".env"])
def test_patch_paths_reject_escape_and_unapproved_files(name):
    diff = f"--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-x\n+y\n"
    with pytest.raises(ValueError):
        patch_paths(diff, {"main.py"})


def test_diff_without_original_trailing_newline_applies():
    before = {"main.py": "value=1"}
    after = {"main.py": "value=2\n"}
    diff = make_diff(before, after)
    with Workspace(before) as workspace:
        workspace.apply(diff)
        assert workspace.read({"main.py"}) == after


def test_workspace_ignores_inherited_git_target(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    monkeypatch.setenv("GIT_DIR", str(outside))
    with Workspace({"main.py": "pass\n"}) as workspace:
        assert "GIT_DIR" not in workspace.env
        assert (workspace.root / ".git").is_dir()
    assert not outside.exists()


def test_pipeline_outputs_metadata_warnings_and_cost_of_only_current_run(tmp_path):
    fake = FakeLLMClient(
        {
            "before": [
                LLMResult(text="ok", input_tokens=100, output_tokens=100, model_id="dummy-model")
            ],
            "diagnose": [
                LLMResult(text="bad", input_tokens=5, output_tokens=2, model_id="dummy-model"),
                LLMResult(
                    text='{"explanations":[],"candidates":[]}',
                    input_tokens=7,
                    output_tokens=3,
                    model_id="dummy-model",
                ),
            ],
        }
    )
    fake.complete("", "", tier="fast", schema=None, stage="before")
    result = run_analysis(SAMPLES / "todo", out_dir=tmp_path / "out", llm=fake, log=lambda *_: None)
    assert result.cost.total.input_tokens == 12 and result.cost.total.output_tokens == 5
    assert result.cost.total.cost_usd is None
    assert result.transformation.patch_valid and result.transformation.compile_passed
    assert len(result.diagnosis.factor_reviews) == 12
    assert result.recommendation.needs_approval
    assert "data_migration_unsupported" in {w.code for w in result.recommendation.warnings}


def test_original_code_and_database_unchanged(tmp_path):
    repo = tmp_path / "original"
    shutil.copytree(SAMPLES / "todo", repo)
    (repo / "todo.db").write_bytes(b"\0dummy-db-do-not-use")
    before = {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in repo.rglob("*")
        if p.is_file()
    }
    run_analysis(repo, out_dir=tmp_path / "out", log=lambda *_: None)
    after = {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in repo.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ('\nos.system("echo dummy-do-not-run")\n', "unsafe_call_added"),
        ('\nimport requests\nrequests.get("https://example.invalid")\n', "unsafe_call_added"),
        ("\nclass ChangedModel:\n    pass\n", "model_class_change_not_allowed"),
        (
            "\nfrom apscheduler.schedulers.background import BackgroundScheduler\n",
            "signals_changed",
        ),
    ],
)
def test_llm_patch_rejects_unrequested_behavior(tmp_path, extra, expected):
    repo = small_repo(
        tmp_path,
        "import os, uvicorn\nfrom fastapi import FastAPI\napp=FastAPI()\n"
        'uvicorn.run("main:app",port=8000)\n',
    )
    before = source_files(repo)
    target = next(v for v in detect(repo) if v.rule == "fixed_port")
    after = {
        "main.py": before["main.py"].replace("port=8000", 'port=int(os.getenv("PORT","8080"))')
        + extra
    }
    fake = FakeLLMClient(
        [json.dumps({"diff": make_diff(before, after), "violation_ids": [target.id]})] * 4
    )
    result, ids, _, warnings = llm_patch(before, [target], fake, SourceMasker(repo))
    assert result == before and ids == [] and warnings
    assert expected in fake.calls[1].user


@pytest.mark.parametrize(
    "comment,expected_error",
    [
        ("\n# no actual fix\n", "patch_scope_violation"),
        (" # no actual fix\n", "requested_violation_unresolved"),
    ],
)
def test_llm_cannot_claim_a_violation_fixed_when_only_comment_changes(
    tmp_path, comment, expected_error
):
    repo = small_repo(tmp_path, 'import uvicorn\nuvicorn.run("main:app",port=8000)\n')
    before = source_files(repo)
    target = next(v for v in detect(repo) if v.rule == "fixed_port")
    after = {"main.py": before["main.py"].rstrip("\n") + comment}
    fake = FakeLLMClient(
        [json.dumps({"diff": make_diff(before, after), "violation_ids": [target.id]})] * 4
    )
    result, ids, _, _ = llm_patch(before, [target], fake, SourceMasker(repo))
    assert result == before and not ids
    assert expected_error in fake.calls[1].user


def test_compile_failure_returns_failed_result_and_no_accepted_changes(tmp_path):
    original = tmp_path / "repo"
    original.mkdir()
    (original / "main.py").write_text("from fastapi import FastAPI\napp=FastAPI()\n")
    (original / "broken.py").write_text("def invalid(:\n")
    result = run_analysis(original, out_dir=tmp_path / "out", log=lambda *_: None)
    assert result.status == "failed"
    assert result.transformation.status == "failed"
    assert not result.transformation.compile_passed
    assert not result.transformation.changed_files
    assert not result.transformation.addressed_ids
    assert not Path(result.output_files["changes.diff"]).read_text()
    assert "transform_validation_failed" in {w.code for w in result.diagnosis.warnings}


@pytest.mark.bedrock
def test_live_bedrock_todo_diff_and_healthz():
    """Opt-in paid Fast-tier enrichment + template diff + trusted sample runtime."""
    from ai.llm import BedrockClient

    script = Path(__file__).resolve().parents[1] / "scripts/smoke_transformed.py"
    spec = importlib.util.spec_from_file_location("bronze_p2_smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.health_check("todo", llm=BedrockClient())
    assert result.diagnosis.enrichment_status == "completed"
    assert any(call.stage == "diagnose" for call in result.cost.calls)
