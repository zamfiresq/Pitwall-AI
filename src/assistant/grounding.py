"""Check that numbers in an answer come from tool results, to catch invented figures."""

from __future__ import annotations

import re

NUMBER = re.compile(r"[-−]?\d+(?:\.\d+)?")
LAP_TIME = re.compile(r"(\d+):(\d{2}(?:\.\d+)?)")
YEAR_RANGE = (1990, 2100)
# An interval level is a convention, not a measured value ("95% interval", "ci95_low", "interval_90").
INTERVAL_LEVEL = re.compile(r"ci\s*(?:80|90|95)|interval_(?:80|90|95)|\b(?:80|90|95)\s*%|\b(?:80|90|95)th", re.I)


def extract_numbers(text: str) -> list[tuple[float, int]]:
    """Return ``(value, decimals_as_written)``; lap times like 1:23.456 become seconds."""
    found: list[tuple[float, int]] = []

    def lap(match: re.Match) -> str:
        seconds = int(match.group(1)) * 60 + float(match.group(2))
        decimals = len(match.group(2).split(".")[1]) if "." in match.group(2) else 0
        found.append((seconds, decimals))
        return " "

    text = LAP_TIME.sub(lap, str(text))
    for token in NUMBER.findall(text):
        token = token.replace("−", "-")
        found.append((float(token), len(token.split(".")[1]) if "." in token else 0))
    return found


def _flatten(value, out: list[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, str):
        out.extend(v for v, _ in extract_numbers(value))
    elif isinstance(value, dict):
        for item in value.values():
            _flatten(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)


def source_numbers(results: list[dict]) -> list[float]:
    numbers: list[float] = []
    for result in results:
        _flatten(result, numbers)
    unique = sorted(set(numbers))
    # Allow simple derived quantities, e.g. the difference between two reported slopes.
    pool = unique[:80]
    unique += [abs(a - b) for i, a in enumerate(pool) for b in pool[i + 1:]]
    return unique


def matches(value: float, decimals: int, candidates: list[float]) -> bool:
    """True if ``value`` equals a candidate rounded to the precision it was written with."""
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - c) <= tolerance for c in candidates)


def check_grounding(answer: str, tool_results: list[dict]) -> dict:
    body = re.split(r"\n\s*sources?\s*:", answer, flags=re.I)[0]
    body = INTERVAL_LEVEL.sub(" ", body)
    candidates = source_numbers(tool_results)
    ungrounded = []
    checked = 0
    for value, decimals in extract_numbers(body):
        if YEAR_RANGE[0] <= value <= YEAR_RANGE[1] and decimals == 0:
            continue
        if decimals == 0 and abs(value) <= 12:
            continue  # small counts and ordinals ("2 stints", "top 3")
        checked += 1
        if not matches(value, decimals, candidates):
            ungrounded.append(value)
    return {"checked": checked, "ungrounded": ungrounded, "grounded": not ungrounded}


def cites_source(answer: str, tool_results: list[dict]) -> bool:
    return any(
        str(result.get("source", "")).split(" ")[0] in answer
        for result in tool_results if result.get("source"))
