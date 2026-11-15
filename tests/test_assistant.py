"""Tests for the assistant's tools, grounding check and evaluation harness.

No real LLM is called. A scripted "oracle" client stands in for the model, so these tests verify the plumbing
(tool calls, grounding, scoring) and not the quality of any language model.
"""

import json
from types import SimpleNamespace

from src.assistant.agent import answer_question
from src.assistant.evaluation import EvalItem, build_eval_set, number_in_text, run_eval, score_item
from src.assistant.grounding import check_grounding, cites_source, extract_numbers
from src.assistant.tools import DATA_DIR, PitwallData, _interval_caveats, call_tool, fmt_lap


def load_data():
    """Data-dependent tests are skipped (return early) when the processed files are not present."""
    return PitwallData() if (DATA_DIR / "model_metrics.json").exists() else None


def message(content="", calls=None):
    tool_calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                  for i, (n, a) in enumerate(calls or [])] or None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class OracleClient:
    """Pretends to be an ideal model: calls the expected tool, then restates the tool result faithfully."""

    def __init__(self, items):
        self.items = {i.question: i for i in items}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, **_):
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        item = self.items[question]
        tool_messages = [m for m in messages if m["role"] == "tool"]
        if not tool_messages:
            if item.expect_tool:
                return message(calls=[(item.expect_tool, item.expect_args), *item.extra_calls])
            return message("I can't answer that; I only cover Ferrari tyre degradation and pace for 2025 and 2026.")
        results = [json.loads(m["content"]) for m in tool_messages]
        lines = [f"{k}: {v}" for result in results for k, v in _leaves(result)]
        return message("\n".join(lines) + "\nSources: " + "; ".join(r["source"] for r in results))


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in {"source", "method", "note", "target"}:
                yield from _leaves(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from _leaves(v, prefix)
    else:
        yield prefix, value


def test_oracle_client_passes_every_generated_question():
    data = load_data()
    if data is None:
        return
    items = build_eval_set(data)
    report = run_eval(data, OracleClient(items), "oracle", None, items)
    failed = [d for d in report["details"] if not d["passed"]]
    assert not failed, failed[:2]
    assert report["overall_pass_rate"] == 1.0


def test_hallucinating_model_is_caught_by_grounding():
    data = load_data()
    if data is None:
        return
    item = EvalItem("x", "estimate", "q", "get_degradation_estimate", {"year": 2025, "compound": "HARD"},
                    expect_numbers=[0.0347])

    class Liar:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: message("Hard degradation is 0.081 s per lap.")))

    result = answer_question("q", data, Liar(), "m")
    scored = score_item(data, item, result)
    assert not scored["grounded"] and not scored["tool"] and not scored["passed"]


def test_agent_recovers_from_a_bad_tool_call():
    data = load_data()
    if data is None:
        return
    script = iter([
        message(calls=[("get_race_summary", {"year": 2025, "event": "Atlantis"})]),
        message("I can't find that Grand Prix."),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: next(script))))
    result = answer_question("fastest lap at Atlantis?", data, client, "m")
    assert "error" in result.tool_calls[0]["result"] and result.grounding["grounded"]


def test_grounding_uses_written_precision_and_derived_values():
    results = [{"a": 0.0346, "b": 0.0227, "gap": 0.012, "lap": "1:23.456", "n": 31, "pct": 28.3, "source": "f.csv (2026)"}]
    assert check_grounding("Slope 0.035 s per lap over 31 stints.", results)["grounded"]
    assert check_grounding("The gap is 0.012.", results)["grounded"]            # precomputed by the tool
    assert not check_grounding("The gap is 0.020.", results)["grounded"]       # 0.0346 - 0.0227 = 0.0119, not accepted
    assert check_grounding("Best lap 1:23.456, an improvement of 28%.", results)["grounded"]
    assert check_grounding("Slope 0.081 s per lap.", results)["ungrounded"] == [0.081]
    assert check_grounding("The gap is 0.0119.", results)["ungrounded"] == [0.0119]  # derived by the model, so unsupported
    assert check_grounding("Slope 0.035 (95% interval, ci95_low) over 31 stints.", results)["grounded"]
    assert check_grounding("The 90% interval is wide.", results)["grounded"]
    assert not check_grounding("The error is 90 s.", results)["grounded"]  # a bare 90 is still checked
    assert check_grounding("In 2026 there were 3 stints.", results)["grounded"]  # years and tiny counts ignored
    assert cites_source("... Sources: f.csv (2026)", results)
    assert extract_numbers("1:23.456") == [(83.456, 3)]
    assert number_in_text(0.5264, "error of 0.53 s") and not number_in_text(0.5264, "error of 0.45 s")


def test_tools_resolve_circuits_and_report_errors():
    data = load_data()
    if data is None:
        return
    assert call_tool(data, "get_race_summary", {"year": 2025, "event": "Monza"})["grand_prix"] == "Italian Grand Prix"
    assert "error" in call_tool(data, "get_race_summary", {"year": 2025, "event": "Atlantis"})
    assert "error" in call_tool(data, "get_driver_degradation", {"year": 2023, "compound": "HARD"})
    assert "error" in call_tool(data, "get_degradation_estimate", {"year": 2025, "wrong_argument": 1})
    assert "error" in call_tool(data, "no_such_tool", {})
    estimate = call_tool(data, "get_degradation_estimate", {"year": 2026, "compound": "hard"})
    assert estimate["ci95_low"] < estimate["degradation_s_per_tyre_lap"] < estimate["ci95_high"]
    assert fmt_lap(83.456) == "1:23.456"


def test_unsupported_figure_triggers_one_repair_pass():
    data = load_data()
    if data is None:
        return
    script = iter([
        message(calls=[("get_degradation_estimate", {"year": 2025, "compound": "HARD"})]),
        message("Hard lost 0.081 s per tyre lap."),          # figure not in the tool result
        message("Hard lost 0.035 s per tyre lap."),          # repaired
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: next(script))))
    result = answer_question("hard degradation 2025?", data, client, "m")
    assert result.repaired and result.grounding["grounded"] and "0.035" in result.answer


def test_good_answer_is_not_repaired_and_sources_come_from_code():
    data = load_data()
    if data is None:
        return
    script = iter([
        message(calls=[("get_degradation_estimate", {"year": 2025, "compound": "HARD"})]),
        message("Hard lost 0.035 s per tyre lap.\nSources: made_up_file.csv"),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: next(script))))
    result = answer_question("hard degradation 2025?", data, client, "m")
    assert not result.repaired and "made_up_file" not in result.answer
    assert result.sources == ["ferrari_tyre_degradation_estimates.csv (2025, HARD)"] and result.cited


def test_caveats_are_computed_from_the_numbers():
    assert any("includes zero" in c for c in _interval_caveats(-0.02, 0.10, 15))
    assert _interval_caveats(0.01, 0.05, 3) == ["Only 3 stints back this number."]
    assert _interval_caveats(0.01, 0.05, 20) == []
    assert _interval_caveats(None, None, 20) == []


def test_tool_results_carry_summary_and_flag_overlapping_drivers():
    data = load_data()
    if data is None:
        return
    result = call_tool(data, "get_driver_degradation", {"year": 2026, "compound": "SOFT"})
    assert result["summary"] and isinstance(result["caveats"], list)
    assert result["higher_slope_driver"] in {"LEC", "HAM"}
    assert result["intervals_overlap"] == any("overlap" in c for c in result["caveats"])
    estimate = call_tool(data, "get_degradation_estimate", {"year": 2025, "compound": "HARD"})
    assert estimate["degradation_s_per_tyre_lap"] == round(estimate["degradation_s_per_tyre_lap"], 3)
