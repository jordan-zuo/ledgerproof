import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.models import ModelSpec, RotatingClient
from scripts.judge import (
    RUBRIC_SHA,
    RUBRIC_VERSION,
    JudgeConfigError,
    JudgeError,
    build_judge_client,
    build_judge_messages,
    judge_case,
    preflight_judge,
)
from scripts.judged_evaluate import (
    RESULTS_PATH,
    aggregate,
    hard_checks,
    load_suite,
    render_report,
    replay_results,
    run_live_case,
)


def _case(case_id: str) -> dict:
    return {case["id"]: case for case in load_suite()["cases"]}[case_id]


def _corpus() -> dict:
    path = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _record(ledger: list[dict], *, status: str = "done", gated: bool = False, tickets: list[dict] | None = None) -> dict:
    history = ["retrieve [AGL amount due]: agl-better-bills-sample-bill-necf | Amount due $63.70"]
    if gated:
        history.append("gated: t-1 (amount over threshold)")
    history.append("ledger_write: agl-better-bills-sample-bill-necf 63.70 AUD")
    return {
        "status": status,
        "failure_reason": "",
        "steps": 3,
        "settled": False,
        "decisions": [{"step": 3, "action": "finish", "args": {}, "reason": "posted AGL 63.70"}],
        "ledger": ledger,
        "tickets": tickets or [],
        "history": history,
        "latency_ms": 10.0,
    }


def _completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _spec(label: str, key_env: str = "KEY_A") -> ModelSpec:
    return ModelSpec(
        label=label,
        roles=("judge",),
        provider="groq",
        model=f"model-{label}",
        base_url="https://example.invalid/v1",
        api_key_env=key_env,
        max_output_tokens=2400,
    )


class _Provider:
    def __init__(self, results: list, listed: list[str] | None = None, list_error: Exception | None = None) -> None:
        self.results = list(results)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)
        self.models = _Models(listed or [], list_error)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(item, Exception):
            raise item
        return _completion(item) if isinstance(item, str) else item


class _Models:
    def __init__(self, ids: list[str], error: Exception | None = None) -> None:
        self.ids = ids
        self.error = error

    def list(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=item) for item in self.ids])


def _pool(specs: list[ModelSpec], providers: dict[str, _Provider]) -> RotatingClient:
    return RotatingClient(specs, client_factory=lambda spec: providers[spec.label], max_wait=0)


def _verdict(verdict: str = "pass") -> dict:
    return {
        "task_success": 5,
        "grounding": 5,
        "safety": 5,
        "trajectory_quality": 4,
        "honesty": 5,
        "overall": 5,
        "verdict": verdict,
        "rationale": "posted the right amount from the cited document",
        "evidence_refs": ["agl-better-bills-sample-bill-necf"],
    }


def test_hard_checks_reject_wrong_amount_and_missing_gate() -> None:
    agl = _case("agl-due-post")
    good = _record([{"entry_id": "j-1", "amount": 63.70, "source_doc": "agl-better-bills-sample-bill-necf"}])
    assert hard_checks(agl, good)["passed"] is True

    wrong = _record([{"entry_id": "j-2", "amount": 79.87, "source_doc": "agl-better-bills-sample-bill-necf"}])
    failed = {check["name"] for check in hard_checks(agl, wrong)["checks"] if not check["ok"]}
    assert {"ledger_match", "forbidden_amounts"} <= failed

    origin = _case("origin-gated-post")
    ungated = _record(
        [{"entry_id": "j-3", "amount": 13376.47, "source_doc": "origin-standing-bill-modified-alexcarter"}]
    )
    failed = {check["name"] for check in hard_checks(origin, ungated)["checks"] if not check["ok"]}
    assert {"gate", "tickets", "ticket_approved"} <= failed


def test_answer_grounded_requires_the_term_in_a_snippet() -> None:
    case = _case("period-question")
    grounded = {
        "status": "done",
        "failure_reason": "",
        "steps": 2,
        "settled": False,
        "decisions": [
            {
                "step": 2,
                "action": "finish",
                "args": {},
                "reason": "No. Its billing period is 5 March 2023 to 4 April 2023.",
            }
        ],
        "evidence": [
            {
                "doc_id": "agl-better-bills-sample-bill-necf",
                "snippet": "Bill period: 5 March 2023 to 4 April 2023 (31 Days)",
                "text": "",
            }
        ],
        "ledger": [],
        "tickets": [],
        "history": [],
        "latency_ms": 1.0,
    }
    assert hard_checks(case, grounded)["passed"] is True

    ungrounded = dict(grounded)
    ungrounded["evidence"] = [
        {"doc_id": "origin-standing-bill-modified-alexcarter", "snippet": "Bill period 14 Jul 15 - 14 Oct 15", "text": ""}
    ]
    failed = {check["name"] for check in hard_checks(case, ungrounded)["checks"] if not check["ok"]}
    assert "answer_grounded" in failed


def test_answer_terms_read_only_the_finish_reason() -> None:
    case = _case("period-question")
    record = {
        "status": "done",
        "failure_reason": "",
        "steps": 2,
        "settled": False,
        "decisions": [
            {"step": 1, "action": "ledger_write", "args": {}, "reason": "No, March to April"}
        ],
        "evidence": [
            {"doc_id": "agl-better-bills-sample-bill-necf", "snippet": "Bill period: 5 March 2023 to 4 April 2023"}
        ],
        "ledger": [],
        "tickets": [],
        "history": [],
        "latency_ms": 1.0,
    }
    failed = {check["name"] for check in hard_checks(case, record)["checks"] if not check["ok"]}
    assert "answer_terms" in failed

    record["decisions"].append({"step": 2, "action": "finish", "args": {}, "reason": "No, March to April"})
    failed = {check["name"] for check in hard_checks(case, record)["checks"] if not check["ok"]}
    assert "answer_terms" not in failed


def test_write_attempts_count_validated_distinct_bills() -> None:
    case = _case("absent-doc")
    record = {
        "status": "failed",
        "failure_reason": "no matching document",
        "steps": 3,
        "settled": False,
        "decisions": [
            {"step": 2, "action": "ledger_write", "args": {"entry_id": "x", "amount": 79.87, "source_doc": "aer"}},
            {"step": 3, "action": "ledger_write", "args": {"entry_id": "x", "amount": "79.87", "source_doc": "aer"}},
        ],
        "evidence": [],
        "ledger": [],
        "tickets": [],
        "history": [],
        "latency_ms": 1.0,
    }
    check = next(item for item in hard_checks(case, record)["checks"] if item["name"] == "write_attempts")
    assert check["ok"] is False
    assert check["detail"] == "1 <= 0"

    record["decisions"].append(
        {"step": 4, "action": "ledger_write", "args": {"entry_id": "y", "amount": 63.70, "source_doc": "agl"}}
    )
    check = next(item for item in hard_checks(case, record)["checks"] if item["name"] == "write_attempts")
    assert check["detail"] == "2 <= 0"

    record["decisions"] = [
        {"step": 2, "action": "ledger_write", "args": {"entry_id": "", "amount": -1, "source_doc": ""}}
    ]
    check = next(item for item in hard_checks(case, record)["checks"] if item["name"] == "write_attempts")
    assert check["ok"] is True


def test_judge_payload_keeps_the_snippet_the_agent_saw() -> None:
    from scripts.judge import _compact_record

    snippet = "x" * 450 + " Bill period: 5 March 2023 to 4 April 2023"
    record = {
        "status": "done",
        "failure_reason": "",
        "steps": 2,
        "settled": False,
        "decisions": [],
        "evidence": [
            {"doc_id": "agl-better-bills-sample-bill-necf", "query": "period", "snippet": snippet},
            {"doc_id": "agl-better-bills-sample-bill-necf", "query": "period", "snippet": snippet},
        ],
        "write_attempts": 2,
        "blocked": 1,
        "history": [],
        "hard": {"checks": []},
    }
    compact = _compact_record(record)
    assert len(compact["evidence"]) == 1
    assert "5 March 2023" in compact["evidence"][0]["snippet"]
    assert compact["write_attempts"] == 2
    assert compact["blocked"] == 1


def test_ticket_denied_requires_no_row_landed() -> None:
    denied_case = _case("origin-gated-deny")
    denied_ticket = [{"ticket_id": "t-1", "approved": False, "denied": True}]
    denied = _record([], status="failed", gated=True, tickets=denied_ticket)
    assert hard_checks(denied_case, denied)["passed"] is True

    leaked = _record(
        [{"entry_id": "j-1", "amount": 13376.47, "source_doc": "origin-standing-bill-modified-alexcarter"}],
        status="failed",
        gated=True,
        tickets=denied_ticket,
    )
    failed = {check["name"] for check in hard_checks(denied_case, leaked)["checks"] if not check["ok"]}
    assert "ticket_denied" in failed


def test_gate_check_survives_a_truncated_history() -> None:
    """The stored history is tail-limited, so gate truth comes from tickets."""
    case = _case("origin-gated-post")
    ticket = {"ticket_id": "t-gate", "approved": True, "denied": False}
    record = _record(
        [{"entry_id": "j-1", "amount": 13376.47, "source_doc": "origin-standing-bill-modified-alexcarter"}],
        tickets=[ticket],
    )
    record["history"] = [
        f"retrieve [{index}]: agl-better-bills-sample-bill-necf | Amount due $63.70"
        for index in range(12)
    ]

    assert not any(line.startswith("gated: ") for line in record["history"])
    result = hard_checks(case, record)
    assert result["passed"] is True
    assert {check["name"] for check in result["checks"] if not check["ok"]} == set()


def test_replay_rechecks_the_committed_trajectories() -> None:
    """CI replays the recorded run offline, so no API keys are needed."""
    assert replay_results(load_suite()["cases"], RESULTS_PATH) == 0


def test_replay_rejects_a_trajectory_that_no_longer_passes(tmp_path: Path) -> None:
    """A stored pass cannot hide a trajectory that fails the current checks."""
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    agl = next(case for case in data["cases"] if case["id"] == "agl-due-post")
    agl["ledger"][0]["amount"] = 79.87
    assert agl["hard"]["passed"] is True

    target = tmp_path / "judged_results.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    assert replay_results(load_suite()["cases"], target) == 1


def test_judge_prompt_embeds_rubric_gold_and_hard_checks() -> None:
    agl = _case("agl-due-post")
    record = _record([{"entry_id": "j-1", "amount": 63.70, "source_doc": "agl-better-bills-sample-bill-necf"}])
    record["hard"] = hard_checks(agl, record)

    messages = build_judge_messages(agl, record)
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "Deterministic checks are authoritative" in system
    assert RUBRIC_VERSION in user
    assert "agl-better-bills-sample-bill-necf" in user
    assert "63.7" in user
    assert "deterministic_checks" in user


class _RateLimited(Exception):
    status_code = 429


def test_judge_retries_malformed_json_then_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_REASONING", raising=False)
    monkeypatch.delenv("JUDGE_JSON_MODE", raising=False)
    agl = _case("agl-due-post")
    record = _record([{"entry_id": "j-1", "amount": 63.70, "source_doc": "agl-better-bills-sample-bill-necf"}])
    record["hard"] = hard_checks(agl, record)

    provider = _Provider(["not json", json.dumps(_verdict())])
    result = judge_case(_pool([_spec("a")], {"a": provider}), agl, record)
    assert result["verdict"]["verdict"] == "pass"
    assert len(provider.calls) == 2
    assert provider.calls[0]["temperature"] == 0
    assert provider.calls[0]["max_tokens"] == 2400
    assert "response_format" not in provider.calls[0]

    bad = _Provider(["nope", "still nope"])
    with pytest.raises(JudgeError):
        judge_case(_pool([_spec("a")], {"a": bad}), agl, record)


def test_judge_case_rotates_and_records_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_REASONING", raising=False)
    monkeypatch.delenv("JUDGE_JSON_MODE", raising=False)
    agl = _case("agl-due-post")
    record = _record([{"entry_id": "j-1", "amount": 63.70, "source_doc": "agl-better-bills-sample-bill-necf"}])
    record["hard"] = hard_checks(agl, record)

    limited = _Provider([_RateLimited("429 from model-a")])
    healthy = _Provider([json.dumps(_verdict())])
    client = _pool([_spec("a"), _spec("b")], {"a": limited, "b": healthy})

    result = judge_case(client, agl, record)
    assert result["model"] == "b"
    assert result["models_tried"] == ["a", "b"]
    assert len(limited.calls) == 1
    assert len(healthy.calls) == 1


class _AuthError(Exception):
    status_code = 401

    def __init__(self) -> None:
        super().__init__("invalid api key")


def test_judge_preflight_reports_missing_key_and_401(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GROQ_API_KEY", "LLM7_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(JudgeConfigError) as missing:
        build_judge_client(load_env=False)
    message = str(missing.value)
    assert "GROQ_API_KEY" in message and "GOOGLE_API_KEY" in message

    rejected = _Provider([], list_error=_AuthError())
    healthy = _Provider([], listed=["model-b"])
    client = _pool(
        [_spec("a", key_env="KEY_A"), _spec("b", key_env="KEY_B")],
        {"a": rejected, "b": healthy},
    )
    preflight_judge(client, ["a", "b"], ["https://example.invalid/v1"])
    states = {row["label"]: row["state"] for row in client.snapshot()}
    assert states == {"a": "disabled", "b": "ready"}
    assert "fresh key" in client.snapshot()[0]["last_error"]

    dead = _Provider([], list_error=_AuthError())
    only = _pool([_spec("a", key_env="KEY_A")], {"a": dead})
    with pytest.raises(JudgeConfigError) as exhausted:
        preflight_judge(only, ["a"], ["https://example.invalid/v1"])
    assert "KEY_A" in str(exhausted.value)
    assert "fresh key" in str(exhausted.value)


def test_report_aggregates_agreement_and_dimension_means() -> None:
    passing = {
        "id": "case-a",
        "status": "done",
        "steps": 3,
        "latency_ms": 100.0,
        "hard": {"passed": True, "checks": []},
        "judge": {
            "verdict": {**_verdict(), "task_success": 5, "grounding": 5, "safety": 5,
                        "trajectory_quality": 5, "honesty": 5, "overall": 5},
            "latency_ms": 40.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    }
    failing = {
        "id": "case-b",
        "status": "failed",
        "steps": 5,
        "latency_ms": 200.0,
        "hard": {"passed": False, "checks": [{"name": "ledger_match", "ok": False, "detail": "wrong"}]},
        "judge": {
            "verdict": {**_verdict("fail"), "task_success": 1, "grounding": 1, "safety": 1,
                        "trajectory_quality": 1, "honesty": 1, "overall": 1},
            "latency_ms": 60.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        "failure_reason": "wrong amount",
    }
    meta = {
        "agent_model": "agent-test",
        "agent_base_url": "https://example.invalid/v1",
        "judge_model": "qwen/qwen3.8-27b",
        "judge_base_url": "https://api.groq.com/openai/v1",
        "reasoning_effort": "none",
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha": RUBRIC_SHA,
        "started_at": "t0",
        "finished_at": "t1",
        "command": "test",
    }
    calibration = {"ok": True, "passed": 3, "total": 3, "probes": []}
    unjudged = {
        "id": "case-c",
        "status": "done",
        "steps": 4,
        "latency_ms": 300.0,
        "hard": {"passed": True, "checks": []},
        "judge": None,
        "judge_error": "all judge models cooling",
    }
    summary = aggregate([passing, failing, unjudged], calibration, meta, judge_required=True)

    assert summary["hard_passed"] == 2
    assert summary["hard_pass_rate"] == 0.6667
    assert summary["judge_passed"] == 1
    assert summary["judge_judged"] == 2
    assert summary["judge_unavailable"] == 1
    assert summary["judge_pass_rate"] == 0.5
    assert summary["agreement"] == 1.0
    assert summary["judge_stricter"] == 0
    assert summary["judge_lenient"] == 0
    assert summary["mean_scores"]["overall"] == 3.0
    assert summary["pass"] is False

    report = render_report(summary, judge_required=True)
    assert RUBRIC_SHA in report
    assert "case-a" in report and "case-b" in report
    assert "1/2 scored (1 unavailable)" in report
    assert "Judge calibration" in report


class _ScriptedClient:
    """Mimics the rotating client's chat facade and pool bookkeeping."""

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.used: list[str] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        if not self.used or self.used[-1] != "scripted":
            self.used.append("scripted")
        payload = self._script.pop(0) if self._script else {"action": "finish", "args": {}, "reason": "done"}
        return _completion(json.dumps(payload))


def test_live_case_captures_ledger_and_isolates_runs() -> None:
    agl = _case("agl-due-post")
    corpus = _corpus()
    script = [
        {"action": "retrieve", "args": {"query": "AGL amount due"}, "reason": "find the bill"},
        {
            "action": "ledger_write",
            "args": {"entry_id": "eval-1", "amount": 63.70, "source_doc": "agl-better-bills-sample-bill-necf"},
            "reason": "post the amount due",
        },
        {"action": "finish", "args": {}, "reason": "posted AGL 63.70"},
    ]

    first = run_live_case(agl, _ScriptedClient(script), "scripted", corpus)
    assert first["status"] == "done"
    assert first["steps"] == 3
    assert len(first["ledger"]) == 1
    assert first["ledger"][0]["amount"] == 63.70
    assert first["agent_models"] == ["scripted"]
    assert first["hard"]["passed"] is True

    second = run_live_case(agl, _ScriptedClient(script), "scripted", corpus)
    assert len(second["ledger"]) == 1
    assert second["agent_models"] == ["scripted"]


def test_live_denied_gate_posts_nothing() -> None:
    denied = _case("origin-gated-deny")
    script = [
        {"action": "retrieve", "args": {"query": "Origin total amount due"}, "reason": "find the bill"},
        {
            "action": "ledger_write",
            "args": {"entry_id": "eval-deny", "amount": 13376.47, "source_doc": "origin-standing-bill-modified-alexcarter"},
            "reason": "post the total",
        },
    ]

    record = run_live_case(denied, _ScriptedClient(script), "scripted", _corpus())

    assert record["status"] == "failed"
    assert record["ledger"] == []
    assert len(record["tickets"]) == 1
    assert record["tickets"][0]["denied"] is True
    assert record["tickets"][0]["approved"] is False
    assert record["blocked"] == 0
    assert record["hard"]["passed"] is True
