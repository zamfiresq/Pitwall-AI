"""Shared lap-cleaning rules used by the degradation and lap-time model modules.

``src.data`` already removes pit in/out laps, deleted/inaccurate laps, non-green
laps and lap 1. This module adds the analysis-time filters that matter for
tyre-degradation work, so they live in exactly one place:

* dry compounds only (SOFT / MEDIUM / HARD) and no rainfall laps: on a drying or
  wetting track, lap time falls or rises for reasons unrelated to tyre wear;
* physically impossible weather readings (e.g. a 0 degC track temperature) become NaN;
* laps much slower than the driver's median pace in that race (traffic, errors,
  lock-ups, damage) are removed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
MAX_PACE_RATIO = 1.04  # keep laps within 104% of the driver's median race lap
WEATHER_COLUMNS = ("Weather_TrackTemp", "Weather_AirTemp", "Weather_Humidity")
REQUIRED_COLUMNS = [
    "LapTimeSeconds", "TyreLife", "Compound", "LapNumber", "Driver", "EventName", "Stint", "Year",
]


def load_season_laps(data_dir: Path, years: tuple[int, ...]) -> pd.DataFrame:
    """Read the per-season CSVs written by ``python -m src.data``."""
    frames = [pd.read_csv(Path(data_dir) / f"ferrari_{year}_race_laps.csv") for year in years]
    return pd.concat(frames, ignore_index=True)


def clean_for_analysis(
    laps: pd.DataFrame,
    max_pace_ratio: float = MAX_PACE_RATIO,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return dry-weather, representative laps suitable for tyre/pace analysis."""
    data = laps.dropna(subset=REQUIRED_COLUMNS).copy()
    start = len(data)

    data = data[data["LapTimeSeconds"].between(60, 180) & data["TyreLife"].between(1, 80)]

    # Wet compounds and any lap flagged with rainfall.
    data = data[data["Compound"].isin(DRY_COMPOUNDS)]
    if "Weather_Rainfall" in data:
        raining = data["Weather_Rainfall"].astype(str).str.lower().eq("true")
        data = data[~raining]
    after_dry = len(data)

    # Zero is a missing-sensor sentinel, not a measurement.
    for column in WEATHER_COLUMNS:
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce")
            data[column] = values.where(values > 0)

    # Slow laps relative to the same driver's median pace in the same race.
    median = data.groupby(["Year", "EventName", "Driver"])["LapTimeSeconds"].transform("median")
    data["PaceRatio"] = data["LapTimeSeconds"] / median
    data = data[data["PaceRatio"] <= max_pace_ratio]

    if verbose:
        print(
            f"Cleaning: {start} laps -> {after_dry} dry -> {len(data)} after the "
            f"{max_pace_ratio:.0%} pace filter."
        )
    return data.reset_index(drop=True)
