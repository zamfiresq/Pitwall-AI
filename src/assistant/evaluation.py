"""Evaluate the assistant against questions whose correct answers are computed from the data.

Run with a configured LLM:   python -m src.assistant.evaluation [--limit N]

Each item checks: did the model call the right tool(s), does the answer contain the right values, is every
number in the answer traceable to a tool result (grounding), are sources attached, does it mention the
uncertainty when an interval includes zero, and does it decline questions the data cannot answer.
``repair_rate`` is how often the first answer contained an untraceable figure and had to be rewritten.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.assistant.agent import AgentResult, LLMConfig, answer_question, make_client
from src.assistant.grounding import extract_numbers
from src.assistant.tools import DATA_DIR, DRIVER_NAMES, FUEL_MEDIAN, PitwallData, _driver, fmt_lap
from src.tracking import log_run

UNCERTAINTY_CUES = (
    "zero", "not clear", "unclear", "uncertain", "cannot rule out", "can't rule out", "not distinguishable",
    "not clearly", "overlap", "inconclusive", "not reliably", "cannot say", "can't say",
)
DECLINE_CUES = (
    "can't", "cannot", "can not", "not available", "don't have", "do not have", "no data", "only cover",
    "outside", "unable", "not able", "isn't available", "not covered", "don't cover", "only have", "not something",
)


@dataclass
class EvalItem:
    id: str
    category: str
    question: str
    expect_tool: str | None = None
    expect_args: dict = field(default_factory=dict)
    expect_numbers: list[float] = field(default_factory=list)
    expect_text_any: list[list[str]] = field(default_factory=list)
    expect_decline: bool = False
    extra_calls: list = field(default_factory=list)  # more (tool, args) pairs the answer needs


def build_eval_set(data: PitwallData, seed: int = 7) -> list[EvalItem]:
    items: list[EvalItem] = []
    for row in data.estimates.itertuples():
        items.append(EvalItem(
            f"estimate-{row.Year}-{row.Compound}", "estimate",
            f"What is the adjusted tyre degradation for the {row.Compound.lower()} compound in {row.Year}?",
            "get_degradation_estimate", {"year": int(row.Year), "compound": row.Compound},
            expect_numbers=[float(row.DegradationSecondsPerTyreLap)]))

    median = FUEL_MEDIAN if FUEL_MEDIAN in data.by_driver else "MedianObservedSlopeSecondsPerTyreLap"
    for (year, compound), group in data.by_driver.groupby(["Year", "Compound"]):
        if len(group) < 2 or compound not in ("SOFT", "MEDIUM", "HARD"):
            continue
        top = group.loc[group[median].idxmax(), "Driver"]
        items.append(EvalItem(
            f"drivers-{year}-{compound}", "driver_comparison",
            f"In {year}, which Ferrari driver shows the higher fuel-corrected median degradation slope on the {compound.lower()} compound?",
            "get_driver_degradation", {"year": int(year), "compound": compound},
            expect_text_any=[[top, DRIVER_NAMES[top], DRIVER_NAMES[top].split()[-1]]]))

    rng = random.Random(seed)
    races = data.laps[["Year", "EventName"]].drop_duplicates().values.tolist()
    for year, event in rng.sample(races, k=min(6, len(races))):
        laps = data.laps[(data.laps["Year"] == year) & (data.laps["EventName"] == event)]
        seconds = float(laps["LapTimeSeconds"].min())
        items.append(EvalItem(
            f"race-{year}-{event}", "race_summary",
            f"What was the fastest clean Ferrari lap in the {event} {int(year)}?",
            "get_race_summary", {"year": int(year), "event": event},
            expect_text_any=[[fmt_lap(seconds), f"{seconds:.3f}"]]))

    if data.metrics:
        cv = {r["name"]: r for r in data.metrics["cross_validation"]}
        chosen = cv[data.metrics["selected_model"]]
        baseline = next(r for n, r in cv.items() if n.startswith("Baseline"))
        items += [
            EvalItem("model-baseline", "model", "What is the baseline error of the race-pace model, in seconds?",
                     "get_model_performance", expect_numbers=[baseline["mae"]]),
            EvalItem("model-improvement", "model", "By what percentage does the pace model beat its baseline?",
                     "get_model_performance", expect_numbers=[chosen["improvement_pct"]]),
        ]

    # Comparison across seasons: needs two tool calls and both numbers.
    soft = data.estimates[data.estimates["Compound"] == "SOFT"].set_index("Year")["DegradationSecondsPerTyreLap"]
    if len(soft) == 2:
        first, second = sorted(soft.index)
        items.append(EvalItem(
            "compare-soft-seasons", "comparison", "How did the adjusted soft-tyre degradation change from "
            f"{first} to {second}?", "get_degradation_estimate", {"year": int(first), "compound": "SOFT"},
            expect_numbers=[float(soft[first]), float(soft[second])],
            extra_calls=[("get_degradation_estimate", {"year": int(second), "compound": "SOFT"})]))

    # Uncertainty awareness: pick a driver summary whose interval includes zero.
    low, high = "FuelCorrectedBootstrap95Lower", "FuelCorrectedBootstrap95Upper"
    if low in data.by_driver:
        unsure = data.by_driver[(data.by_driver[low] <= 0) & (data.by_driver[high] >= 0)
                                & data.by_driver["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
        for row in unsure.head(2).itertuples():
            name = DRIVER_NAMES.get(row.Driver, row.Driver)
            items.append(EvalItem(
                f"caveat-{row.Year}-{row.Driver}-{row.Compound}", "uncertainty",
                f"In {row.Year}, is {name}'s {row.Compound.lower()}-tyre degradation clearly different from zero?",
                "get_driver_degradation", {"year": int(row.Year), "compound": row.Compound, "driver": row.Driver},
                expect_text_any=[list(UNCERTAINTY_CUES)]))

    items += [
        EvalItem("scope-championship", "out_of_scope", "Who will win the 2027 drivers' championship?", expect_decline=True),
        EvalItem("scope-other-driver", "out_of_scope",
                 "What was Max Verstappen's tyre degradation on the hard compound in 2025?", expect_decline=True),
        EvalItem("scope-other-year", "out_of_scope", "What was Leclerc's hard-tyre degradation in 2023?", expect_decline=True),
        EvalItem("scope-opinion", "out_of_scope", "Is the Ferrari strategy team doing a bad job?", expect_decline=True),
    ]
    return items


def _args_match(data: PitwallData, expected: dict, actual: dict) -> bool:
    for key, value in expected.items():
        got = actual.get(key)
        if got is None:
            return False
        if key == "event":
            if data.resolve_event(str(got)) != data.resolve_event(str(value)):
                return False
        elif key == "driver":
            if _driver(str(got)) != _driver(str(value)):
                return False
        elif str(got).upper() != str(value).upper():
            return False
    return True


def number_in_text(value: float, text: str) -> bool:
    """True if the text states ``value`` to the precision it was written with (0.526 or 0.53 for 0.5264)."""
    return any(abs(found - value) <= 0.5 * 10 ** (-decimals) + 1e-9 for found, decimals in extract_numbers(text))


def contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(term.lower())}(?![\w])", text.lower()) is not None


def score_item(data: PitwallData, item: EvalItem, result: AgentResult) -> dict:
    answer = result.answer.lower()
    flags: dict[str, bool] = {}
    if item.expect_tool:
        required = [(item.expect_tool, item.expect_args), *item.extra_calls]
        flags["tool"] = all(
            any(c["name"] == tool and _args_match(data, args, c["arguments"]) for c in result.tool_calls)
            for tool, args in required)
        flags["cited"] = result.cited
    if item.expect_numbers:
        flags["values"] = all(number_in_text(v, result.answer) for v in item.expect_numbers)
    for alternatives in item.expect_text_any:
        flags["text"] = flags.get("text", True) and any(contains_term(result.answer, a) for a in alternatives)
    if item.expect_decline:
        flags["declined"] = any(cue in answer for cue in DECLINE_CUES)
    flags["grounded"] = bool(result.grounding.get("grounded", True))
    return {**flags, "passed": all(flags.values())}


def run_eval(data: PitwallData, client, model: str, temperature: float | None = 0.0,
             items: list[EvalItem] | None = None, limit: int | None = None) -> dict:
    items = (items or build_eval_set(data))[:limit]
    details = []
    for item in items:
        try:
            result = answer_question(item.question, data, client, model, temperature)
        except Exception as exc:  # keep one failing call from discarding the whole run
            result = AgentResult(answer=f"[error: {exc}]")
        details.append({"id": item.id, "category": item.category, "question": item.question,
                        "answer": result.answer, "tools": [c["name"] for c in result.tool_calls], "repaired": result.repaired,
                        **score_item(data, item, result)})
    frame = pd.DataFrame(details)

    def rate(column: str) -> float | None:
        values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        return float(values.astype(bool).mean()) if len(values) else None

    return {
        "model": model, "questions": len(frame), "overall_pass_rate": float(frame["passed"].mean()),
        "tool_accuracy": rate("tool"), "value_accuracy": rate("values"), "citation_rate": rate("cited"),
        "grounded_rate": rate("grounded"), "decline_rate": rate("declined"),
        "repair_rate": rate("repaired"),
        "by_category": frame.groupby("category")["passed"].mean().round(3).to_dict(),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "assistant_eval.json")
    args = parser.parse_args()

    config = LLMConfig.from_env()
    if not config.configured:
        raise SystemExit("No LLM configured. Set PITWALL_LLM_API_KEY (or PITWALL_LLM_BASE_URL for a local model); see .env.example.")
    data = PitwallData()
    report = run_eval(data, make_client(config), config.model, config.temperature, limit=args.limit)
    args.out.write_text(json.dumps(report, indent=2))
    log_run("assistant_eval", {"model": config.model, "questions": report["questions"]},
            {k: report[k] for k in ("overall_pass_rate", "tool_accuracy", "value_accuracy", "grounded_rate", "citation_rate")})
    print(f"Model {config.model}: {report['overall_pass_rate']:.0%} of {report['questions']} questions fully correct")
    for name in ("tool_accuracy", "value_accuracy", "citation_rate", "grounded_rate", "decline_rate", "repair_rate"):
        if report[name] is not None:
            print(f"  {name.replace('_', ' '):<16} {report[name]:.0%}")
    for category, value in report["by_category"].items():
        print(f"  [{category}] {value:.0%}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
