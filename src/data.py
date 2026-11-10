# data cleaning and importing fastf1 api

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

YEARS = (2025, 2026)
DRIVERS = {"LEC", "HAM"}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE_DIR = Path(__file__).resolve().parents[1] / ".fastf1_cache"


def load_ferrari_race_laps(year: int) -> pd.DataFrame:
    # load race laps for Ferrari s Leclerc and Hamilton across a season
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows: list[pd.DataFrame] = []

    for _, event in schedule.iterrows():
        if pd.isna(event.get("EventName")):
            continue
        try:
            session = fastf1.get_session(year, event["RoundNumber"], "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps
            if laps.empty:
                continue

            ferrari = laps[
                laps["Driver"].isin(DRIVERS)
                & (laps["Team"] == "Ferrari")
            ].copy()
            if ferrari.empty:
                continue

            ferrari["EventName"] = event["EventName"]
            ferrari["Year"] = year
            ferrari["LapTimeSeconds"] = ferrari["LapTime"].dt.total_seconds()
            rows.append(ferrari)
            print(f"Loaded {event['EventName']}: {len(ferrari)} Ferrari laps")
        except Exception as exc:
            # A missing/partial weekend should not prevent processing the season.
            print(f"Skipping {event['EventName']}: {exc}")

    if not rows:
        raise RuntimeError(f"FastF1 returned no Ferrari race laps for {year}.")
    return pd.concat(rows, ignore_index=True)


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Keep representative green-flag laps and retain useful audit columns."""
    data = laps.copy()
    required = ["LapTimeSeconds", "TyreLife", "Compound", "LapNumber", "Driver", "EventName"]
    data = data.dropna(subset=required)
    data = data[data["LapTimeSeconds"].between(60, 180)]
    data = data[data["TyreLife"].between(1, 80)]

    # Out/in laps and inaccurately timed laps are not representative of race pace.
    for col in ("PitInTime", "PitOutTime"):
        if col in data:
            data = data[data[col].isna()]
    if "IsAccurate" in data:
        data = data[data["IsAccurate"].astype("boolean").fillna(False)]
    if "Deleted" in data:
        data = data[~data["Deleted"].astype("boolean").fillna(False)]
    if "TrackStatus" in data:
        data = data[data["TrackStatus"].astype(str) == "1"]

    # Exclude the opening lap, which has unusual starts and bunching.
    data = data[data["LapNumber"] > 1]
    data["TyreLife"] = data["TyreLife"].astype(float)
    data["LapNumber"] = data["LapNumber"].astype(float)
    return data.reset_index(drop=True)


def save_season_data(year: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = clean_laps(load_ferrari_race_laps(year))
    path = DATA_DIR / f"ferrari_{year}_race_laps.csv"
    cleaned.to_csv(path, index=False)
    print(f"Saved {len(cleaned)} cleaned laps to {path}")
    print(cleaned.groupby("Driver")["LapTimeSeconds"].agg(["count", "mean"]))
    return path


def save_all_seasons(years: tuple[int, ...] = YEARS) -> None:
    for year in years:
        save_season_data(year)


if __name__ == "__main__":
    save_all_seasons()
