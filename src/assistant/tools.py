"""Data tools the assistant may call. Every result carries a ``source`` for citation.

The model never sees raw laps or invents numbers: it can only quote what these functions return.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DRIVER_NAMES = {"LEC": "Charles Leclerc", "HAM": "Lewis Hamilton"}
DRIVER_ALIASES = {"leclerc": "LEC", "charles": "LEC", "lec": "LEC", "hamilton": "HAM", "lewis": "HAM", "ham": "HAM"}
CIRCUIT_ALIASES = {
    "monza": "italian", "spa": "belgian", "silverstone": "british", "suzuka": "japanese",
    "imola": "emilia romagna", "interlagos": "sao paulo", "cota": "united states", "austin": "united states",
    "zandvoort": "dutch", "baku": "azerbaijan", "jeddah": "saudi arabian", "sakhir": "bahrain",
    "melbourne": "australian", "albert park": "australian", "shanghai": "chinese", "barcelona": "spanish",
    "spielberg": "austrian", "red bull ring": "austrian", "budapest": "hungarian", "hungaroring": "hungarian",
    "marina bay": "singapore", "lusail": "qatar", "yas marina": "abu dhabi", "montreal": "canadian",
    "mexico": "mexico city",
}
LOW_STINTS = 8  # fewer independent stints than this makes a median or interval fragile
FUEL_MEDIAN = "MedianFuelCorrectedSlopeSecondsPerTyreLap"
RAW_MEDIAN = "MedianObservedSlopeSecondsPerTyreLap"


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().strip()


def fmt_lap(seconds: float) -> str:
    minutes, rest = divmod(float(seconds), 60)
    return f"{int(minutes)}:{rest:06.3f}"


def _round(value, digits: int = 3):
    return None if value is None or pd.isna(value) else round(float(value), digits)


class PitwallData:
    """Lazy, read-only access to the processed CSV/JSON outputs."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)
        self.estimates = self._csv("ferrari_tyre_degradation_estimates.csv")
        self.by_driver = self._csv("ferrari_tyre_degradation_by_driver.csv")
        self.by_race = self._csv("ferrari_tyre_degradation_by_race_driver.csv")
        metrics_path = self.dir / "model_metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        frames = [pd.read_csv(p, usecols=lambda c: c in {
            "Year", "EventName", "Driver", "LapNumber", "LapTimeSeconds", "Compound", "Stint", "TyreLife",
            "Weather_TrackTemp"}) for p in sorted(self.dir.glob("ferrari_20*_race_laps.csv"))]
        self.laps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _csv(self, name: str) -> pd.DataFrame:
        path = self.dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    @property
    def events(self) -> list[str]:
        return sorted(self.laps["EventName"].dropna().unique().tolist()) if not self.laps.empty else []

    @property
    def years(self) -> list[int]:
        return sorted(self.laps["Year"].dropna().astype(int).unique().tolist()) if not self.laps.empty else []

    def resolve_event(self, text: str) -> str | None:
        query = CIRCUIT_ALIASES.get(_plain(text), _plain(text))
        matches = [e for e in self.events if query and query in _plain(e)]
        return matches[0] if len(matches) == 1 else (None if not matches else min(matches, key=len))


def _error(message: str, **extra) -> dict:
    return {"error": message, **extra}


def _driver(code: str | None) -> str | None:
    return None if not code else DRIVER_ALIASES.get(_plain(code), str(code).upper())


def list_events(data: PitwallData, year: int | None = None) -> dict:
    events = sorted(data.laps[data.laps["Year"] == int(year)]["EventName"].unique().tolist()) if year else data.events
    return {"seasons_available": data.years, "events": events, "source": "ferrari_YYYY_race_laps.csv"}


def _interval_caveats(low, high, stints: int, label: str = "") -> list[str]:
    """Plain-language warnings computed from the numbers, so the model never has to infer them."""
    notes = []
    if low is not None and high is not None and low <= 0 <= high:
        notes.append(f"{label}The 95% interval includes zero, so the data cannot rule out no degradation.")
    if stints < LOW_STINTS:
        notes.append(f"{label}Only {stints} stints back this number.")
    return notes


def get_degradation_estimate(data: PitwallData, year: int, compound: str) -> dict:
    rows = data.estimates[(data.estimates["Year"] == int(year)) & (data.estimates["Compound"] == str(compound).upper())]
    if rows.empty:
        return _error("No estimate for that season/compound.", seasons=data.years, compounds=["SOFT", "MEDIUM", "HARD"])
    row = rows.iloc[0]
    degradation, low, high = (_round(row[c]) for c in ("DegradationSecondsPerTyreLap", "CI95Lower", "CI95Upper"))
    stints = int(row["Stints"])
    return {
        "summary": (f"{int(year)} {row['Compound'].lower()} compound: adjusted degradation {degradation:.3f} s per tyre lap "
                    f"(95% interval {low:.3f} to {high:.3f}), from {stints} stints."),
        "caveats": _interval_caveats(low, high, stints),
        "season": int(year), "compound": row["Compound"],
        "degradation_s_per_tyre_lap": degradation, "ci95_low": low, "ci95_high": high,
        "clean_laps": int(row["CleanLaps"]), "stints": stints,
        "race_lap_effect_s_per_lap": _round(row.get("RaceLapEffectSecondsPerLap")),
        "method": "Huber regression with event-driver fixed effects; 95% stint-cluster bootstrap interval",
        "source": f"ferrari_tyre_degradation_estimates.csv ({int(year)}, {row['Compound']})",
    }


def get_driver_degradation(data: PitwallData, year: int, compound: str, driver: str | None = None,
                           fuel_corrected: bool = True) -> dict:
    table = data.by_driver
    rows = table[(table["Year"] == int(year)) & (table["Compound"] == str(compound).upper())]
    if driver:
        rows = rows[rows["Driver"] == _driver(driver)]
    if rows.empty:
        return _error("No driver summary for that selection (only LEC and HAM, 2025-2026, dry compounds).")
    use_fuel = fuel_corrected and FUEL_MEDIAN in rows
    median, low_col, high_col = (
        (FUEL_MEDIAN, "FuelCorrectedBootstrap95Lower", "FuelCorrectedBootstrap95Upper") if use_fuel
        else (RAW_MEDIAN, "ClusterBootstrap95Lower", "ClusterBootstrap95Upper"))
    drivers, caveats, parts = [], [], []
    for _, r in rows.iterrows():
        slope, low, high, stints = _round(r[median]), _round(r[low_col]), _round(r[high_col]), int(r["Stints"])
        notes = _interval_caveats(low, high, stints, label=f"{r['Driver']}: ")
        interval = f" (95% interval {low:.3f} to {high:.3f}, {stints} stints)" if low is not None else f" ({stints} stints, no interval)"
        parts.append(f"{r['Driver']} {slope:.3f} s per tyre lap{interval}")
        caveats += notes
        drivers.append({
            "driver": r["Driver"], "name": DRIVER_NAMES.get(r["Driver"], r["Driver"]), "stints": stints,
            "median_slope_s_per_tyre_lap": slope, "ci95_low": low, "ci95_high": high,
        })
    result = {
        "season": int(year), "compound": str(compound).upper(),
        "basis": "fuel-corrected median stint slope" if use_fuel else "raw median stint slope (includes fuel-burn effect)",
        "drivers": drivers,
    }
    if len(drivers) == 2:
        first, second = drivers
        top = max(drivers, key=lambda d: d["median_slope_s_per_tyre_lap"])
        bounds = [d[k] for d in drivers for k in ("ci95_low", "ci95_high")]
        overlap = None if any(b is None for b in bounds) else (
            max(first["ci95_low"], second["ci95_low"]) <= min(first["ci95_high"], second["ci95_high"]))
        result.update({
            "higher_slope_driver": top["driver"],
            "difference_s_per_tyre_lap": _round(abs(first["median_slope_s_per_tyre_lap"] - second["median_slope_s_per_tyre_lap"])),
            "intervals_overlap": overlap,
        })
        if overlap:
            caveats.append("The two drivers' 95% intervals overlap, so the difference is not clearly distinguishable.")
    basis = "fuel-corrected" if use_fuel else "raw"
    result["summary"] = f"{int(year)} {str(compound).lower()} compound, {basis} median stint slope: " + "; ".join(parts) + "."
    result["caveats"] = caveats
    result["source"] = f"ferrari_tyre_degradation_by_driver.csv ({int(year)}, {str(compound).upper()})"
    return result


def get_race_degradation(data: PitwallData, year: int, event: str, driver: str | None = None,
                         compound: str | None = None) -> dict:
    name = data.resolve_event(event)
    if not name:
        return _error(f"Unknown Grand Prix '{event}'.", events=data.events)
    rows = data.by_race[(data.by_race["Year"] == int(year)) & (data.by_race["EventName"] == name)]
    if driver:
        rows = rows[rows["Driver"] == _driver(driver)]
    if compound:
        rows = rows[rows["Compound"] == str(compound).upper()]
    if rows.empty:
        return _error("No stint summaries for that selection.")
    median = FUEL_MEDIAN if FUEL_MEDIAN in rows else RAW_MEDIAN
    out_rows = [{
        "driver": r["Driver"], "compound": r["Compound"], "stints": int(r["Stints"]),
        "median_slope_s_per_tyre_lap": _round(r[median]), "anecdotal": int(r["Stints"]) < 2,
    } for _, r in rows.head(12).iterrows()]
    return {
        "summary": f"{len(out_rows)} driver and compound summaries for the {name} {int(year)} "
                   f"({'fuel-corrected' if median == FUEL_MEDIAN else 'raw'} median stint slopes).",
        "caveats": ["Rows marked anecdotal rest on a single stint and have no interval."] if any(r["anecdotal"] for r in out_rows) else [],
        "season": int(year), "grand_prix": name, "rows": out_rows,
        "source": f"ferrari_tyre_degradation_by_race_driver.csv ({int(year)}, {name})",
    }


def get_race_summary(data: PitwallData, year: int, event: str, driver: str | None = None) -> dict:
    name = data.resolve_event(event)
    if not name:
        return _error(f"Unknown Grand Prix '{event}'.", events=data.events)
    laps = data.laps[(data.laps["Year"] == int(year)) & (data.laps["EventName"] == name)]
    if driver:
        laps = laps[laps["Driver"] == _driver(driver)]
    if laps.empty:
        return _error("No laps for that selection.", seasons=data.years)
    fastest = laps.loc[laps["LapTimeSeconds"].idxmin()]
    stints = laps[["Driver", "Stint"]].drop_duplicates().shape[0]
    drivers = sorted(laps["Driver"].unique().tolist())
    lap = fmt_lap(fastest["LapTimeSeconds"])
    return {
        "summary": (f"{name} {int(year)}: fastest clean lap {lap} by {fastest['Driver']} on {fastest['Compound'].lower()} tyres; "
                    f"{len(laps)} clean laps and {stints} stints in the data."),
        "caveats": ([f"Only {drivers[0]} has clean laps for this race in the data."] if len(drivers) == 1 and not driver else []),
        "season": int(year), "grand_prix": name, "drivers": drivers,
        "clean_laps": int(len(laps)),
        "fastest_clean_lap": lap,
        "fastest_clean_lap_seconds": _round(fastest["LapTimeSeconds"], 3),
        "fastest_lap_driver": fastest["Driver"], "fastest_lap_compound": fastest["Compound"],
        "mean_clean_lap_seconds": _round(laps["LapTimeSeconds"].mean(), 3),
        "compounds_used": sorted(laps["Compound"].unique().tolist()), "stints": int(stints),
        "mean_track_temp_c": _round(laps["Weather_TrackTemp"].mean(), 1),
        "note": "Clean laps exclude pit laps, lap 1, deleted/inaccurate laps and non-green running.",
        "source": f"ferrari_{int(year)}_race_laps.csv ({name})",
    }


def get_model_performance(data: PitwallData) -> dict:
    m = data.metrics
    if not m:
        return _error("Model metrics not available. Run `python -m src.model`.")
    cv = {row["name"]: row for row in m["cross_validation"]}
    baseline = next(r for n, r in cv.items() if n.startswith("Baseline"))
    chosen = cv[m["selected_model"]]
    holdout = {row["name"]: row for row in m["season_holdout"]["models"]}
    hold_base = next(r for n, r in holdout.items() if n.startswith("Baseline"))
    out = {
        "target": m["target"], "cv_scheme": m["cv_scheme"], "race_weekends": m["race_weekends"],
        "selected_model": m["selected_model"],
        "baseline_mae_s": _round(baseline["mae"], 3), "selected_model_mae_s": _round(chosen["mae"], 3),
        "improvement_vs_baseline_pct": _round(chosen["improvement_pct"], 1),
        "season_holdout": {
            "train": m["season_holdout"]["train"], "test": m["season_holdout"]["test"],
            "baseline_mae_s": _round(hold_base["mae"], 3),
            "selected_model_mae_s": _round(holdout[m["selected_model"]]["mae"], 3),
        },
        "source": "model_metrics.json",
    }
    out["summary"] = (
        f"Pace model ({m['selected_model']}): error {out['selected_model_mae_s']:.3f} s versus {out['baseline_mae_s']:.3f} s for the "
        f"baseline, {out['improvement_vs_baseline_pct']:.1f}% lower, cross-validated by race weekend. Train "
        f"{m['season_holdout']['train']}, test {m['season_holdout']['test']}: {out['season_holdout']['selected_model_mae_s']:.3f} s "
        f"versus {out['season_holdout']['baseline_mae_s']:.3f} s.")
    out["caveats"] = []
    if "whatif" in m:
        out["interval_90_half_width_s"] = _round(m["whatif"]["interval_half_width_s"]["0.9"], 2)
        out["interval_90_actual_coverage"] = _round(m["whatif"]["interval_coverage"]["0.9"], 3)
    return out


TOOLS = {
    "list_events": list_events,
    "get_degradation_estimate": get_degradation_estimate,
    "get_driver_degradation": get_driver_degradation,
    "get_race_degradation": get_race_degradation,
    "get_race_summary": get_race_summary,
    "get_model_performance": get_model_performance,
}

_YEAR = {"type": "integer", "description": "Season, 2025 or 2026"}
_COMPOUND = {"type": "string", "enum": ["SOFT", "MEDIUM", "HARD"], "description": "Dry tyre compound"}
_DRIVER = {"type": "string", "description": "LEC (Charles Leclerc) or HAM (Lewis Hamilton)"}
_EVENT = {"type": "string", "description": "Grand Prix name or circuit, e.g. 'Bahrain Grand Prix' or 'Monza'"}


def _spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SPECS = [
    _spec("list_events", "List the Grands Prix and seasons available.", {"year": _YEAR}, []),
    _spec("get_degradation_estimate",
          "Season-level adjusted tyre degradation (seconds per tyre lap) for one compound, with 95% interval.",
          {"year": _YEAR, "compound": _COMPOUND}, ["year", "compound"]),
    _spec("get_driver_degradation",
          "Median per-stint degradation slope for Leclerc and/or Hamilton on one compound and season, with 95% bootstrap "
          "interval. Omit driver to compare both. Fuel-corrected by default.",
          {"year": _YEAR, "compound": _COMPOUND, "driver": _DRIVER,
           "fuel_corrected": {"type": "boolean", "description": "Default true"}}, ["year", "compound"]),
    _spec("get_race_degradation", "Per-race stint slope summaries for one Grand Prix.",
          {"year": _YEAR, "event": _EVENT, "driver": _DRIVER, "compound": _COMPOUND}, ["year", "event"]),
    _spec("get_race_summary",
          "Race overview from clean laps: fastest clean lap, compounds used, stints, track temperature.",
          {"year": _YEAR, "event": _EVENT, "driver": _DRIVER}, ["year", "event"]),
    _spec("get_model_performance",
          "Accuracy of the race-pace model versus its baseline, including the 2025 to 2026 hold-out.", {}, []),
]


def call_tool(data: PitwallData, name: str, arguments: dict) -> dict:
    """Run a tool by name; never raises, so a bad call becomes feedback for the model."""
    function = TOOLS.get(name)
    if function is None:
        return _error(f"Unknown tool '{name}'.", tools=list(TOOLS))
    try:
        return function(data, **(arguments or {}))
    except TypeError as exc:
        return _error(f"Bad arguments: {exc}")
    except Exception as exc:  # defensive: surface the problem to the model instead of crashing the UI
        return _error(f"Tool failed: {exc}")
