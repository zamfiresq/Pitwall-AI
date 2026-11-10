# estimate average tyre age degradation separately for each season

# this is a robust, season-level association model; it controls for race progress 
# event/ driver pace and available weather observations
# it doesn t claim to isolate tyre wear perfectly from every race effect


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_estimates.csv"
STINT_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_stint_slopes.csv"
RACE_DRIVER_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_by_race_driver.csv"
DRIVER_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_by_driver.csv"
YEARS = (2025, 2026)
MIN_STINT_LAPS = 5
MIN_TYRE_AGE_SPAN = 3
BOOTSTRAP_REPLICATES = 2000


def prepare_data(year: int) -> pd.DataFrame:
    path = DATA_DIR / f"ferrari_{year}_race_laps.csv"
    data = pd.read_csv(path)
    required = ["LapTimeSeconds", "TyreLife", "Compound", "LapNumber", "Driver", "EventName", "Stint"]
    data = data.dropna(subset=required).copy()
    data = data[data["LapTimeSeconds"].between(60, 180)]
    data = data[data["TyreLife"].between(1, 80)]

    # Ignore very short stints and stints with too little tyre-age variation.
    stint_keys = ["EventName", "Driver", "Stint", "Compound"]
    stats = data.groupby(stint_keys, observed=True)["TyreLife"].agg(["count", "min", "max"])
    valid_stints = stats[
        (stats["count"] >= MIN_STINT_LAPS)
        & ((stats["max"] - stats["min"]) >= MIN_TYRE_AGE_SPAN)
    ].index
    data = data.set_index(stint_keys).loc[lambda frame: frame.index.isin(valid_stints)].reset_index()

    data["EventDriver"] = (
        data["Year"].astype(str) + " | " + data["EventName"].astype(str)
        + " | " + data["Driver"].astype(str)
    )
    for output, source in (("TrackTemp", "Weather_TrackTemp"), ("AirTemp", "Weather_AirTemp")):
        if source in data:
            data[output] = pd.to_numeric(data[source], errors="coerce")
        else:
            data[output] = np.nan
        median = data[output].median()
        data[output] = data[output].fillna(0 if pd.isna(median) else median)

    rainfall_source = "Weather_Rainfall"
    if rainfall_source in data:
        data["Rainfall"] = data[rainfall_source].map({True: 1.0, False: 0.0})
        data["Rainfall"] = pd.to_numeric(data["Rainfall"], errors="coerce").fillna(0.0)
    else:
        data["Rainfall"] = 0.0
    return data.reset_index(drop=True)


def fit_season(year: int) -> list[dict[str, object]]:
    data = prepare_data(year)
    if data["Compound"].nunique() < 2:
        raise RuntimeError(f"Not enough tyre compounds in {year} to estimate degradation.")

    # Event-driver fixed effects absorb circuit and driver pace differences.
    # LapNumber is a proxy for race progression/fuel burn; weather is included
    # where available. TyreLife-by-compound terms yield a slope per compound.
    formula = (
        "LapTimeSeconds ~ C(Compound) * TyreLife + C(EventDriver) "
        "+ LapNumber + TrackTemp + AirTemp + Rainfall"
    )
    result = smf.rlm(formula, data=data, M=sm.robust.norms.HuberT()).fit()
    params = result.params
    names = list(params.index)
    compounds = sorted(data["Compound"].astype(str).unique())
    reference = compounds[0]
    estimates: list[dict[str, object]] = []

    for compound in compounds:
        contrast = np.zeros(len(names))
        contrast[names.index("TyreLife")] = 1.0
        if compound != reference:
            interaction = f"C(Compound)[T.{compound}]:TyreLife"
            if interaction not in names:
                # Handle a different but equivalent Patsy interaction ordering.
                interaction = f"TyreLife:C(Compound)[T.{compound}]"
            if interaction not in names:
                continue
            contrast[names.index(interaction)] = 1.0

        test = result.t_test(contrast)
        ci = np.asarray(test.conf_int()).reshape(-1, 2)[0]
        estimate = float(np.asarray(test.effect).reshape(-1)[0])
        compound_data = data[data["Compound"].astype(str) == compound]
        estimates.append({
            "Year": year,
            "Compound": compound,
            "DegradationSecondsPerTyreLap": estimate,
            "CI95Lower": float(ci[0]),
            "CI95Upper": float(ci[1]),
            "CleanLaps": int(len(compound_data)),
            "Stints": int(compound_data[["EventName", "Driver", "Stint"]].drop_duplicates().shape[0]),
            "Model": "Huber regression; controls: event-driver, race lap, track/air temperature, rainfall",
        })
    return estimates


def calculate_stint_slopes(year: int) -> pd.DataFrame:
    """Get robust observed pace-vs-tyre-age slope for every eligible stint."""
    data = prepare_data(year)
    keys = ["Year", "EventName", "Driver", "Stint", "Compound"]
    records: list[dict[str, object]] = []
    for values, stint in data.groupby(keys, observed=True):
        stint = stint.sort_values("TyreLife")
        ages = stint["TyreLife"].to_numpy(dtype=float)
        times = stint["LapTimeSeconds"].to_numpy(dtype=float)
        if len(np.unique(ages)) < 2:
            continue
        slope, intercept, _, _ = theilslopes(times, ages)
        year_value, event, driver, stint_number, compound = values
        records.append({
            "Year": int(year_value),
            "EventName": event,
            "Driver": driver,
            "Stint": stint_number,
            "Compound": compound,
            "Laps": len(stint),
            "TyreAgeMin": float(np.min(ages)),
            "TyreAgeMax": float(np.max(ages)),
            "ObservedSlopeSecondsPerTyreLap": float(slope),
            "ObservedInterceptSeconds": float(intercept),
        })
    return pd.DataFrame(records)


def bootstrap_stint_median(
    grouped: pd.DataFrame,
    group_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    """Summarize stint slopes; resample whole stints to preserve lap dependence."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in grouped.groupby(group_columns, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        slopes = group["ObservedSlopeSecondsPerTyreLap"].dropna().to_numpy(dtype=float)
        count = len(slopes)
        row = dict(zip(group_columns, keys))
        row["Stints"] = count
        row["MedianObservedSlopeSecondsPerTyreLap"] = float(np.median(slopes)) if count else np.nan
        if count >= 2:
            draws = rng.choice(slopes, size=(BOOTSTRAP_REPLICATES, count), replace=True)
            medians = np.median(draws, axis=1)
            row["ClusterBootstrap95Lower"] = float(np.quantile(medians, 0.025))
            row["ClusterBootstrap95Upper"] = float(np.quantile(medians, 0.975))
        else:
            # A single stint cannot provide a between-stint uncertainty estimate.
            row["ClusterBootstrap95Lower"] = np.nan
            row["ClusterBootstrap95Upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    estimates: list[dict[str, object]] = []
    stint_tables: list[pd.DataFrame] = []
    for year in YEARS:
        estimates.extend(fit_season(year))
        stint_tables.append(calculate_stint_slopes(year))

    summary = pd.DataFrame(estimates).sort_values(["Year", "Compound"])
    summary.to_csv(OUTPUT_PATH, index=False)

    stint_slopes = pd.concat(stint_tables, ignore_index=True)
    stint_slopes.to_csv(STINT_OUTPUT_PATH, index=False)
    by_race_driver = bootstrap_stint_median(
        stint_slopes,
        ["Year", "EventName", "Driver", "Compound"],
        seed=42,
    ).sort_values(["Year", "EventName", "Driver", "Compound"])
    by_race_driver.to_csv(RACE_DRIVER_OUTPUT_PATH, index=False)
    by_driver = bootstrap_stint_median(
        stint_slopes,
        ["Year", "Driver", "Compound"],
        seed=43,
    ).sort_values(["Year", "Driver", "Compound"])
    by_driver.to_csv(DRIVER_OUTPUT_PATH, index=False)

    print("Estimated average lap-time change per additional tyre lap (seconds):")
    print(summary[[
        "Year", "Compound", "DegradationSecondsPerTyreLap",
        "CI95Lower", "CI95Upper", "CleanLaps", "Stints",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved estimates to {OUTPUT_PATH}")
    print("\nObserved robust slopes by season, driver, and compound:")
    print(by_driver.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved per-stint slopes to {STINT_OUTPUT_PATH}")
    print(f"Saved race/pilot summaries to {RACE_DRIVER_OUTPUT_PATH}")
    print(f"Saved driver summaries to {DRIVER_OUTPUT_PATH}")
    print(
        "Bootstrap intervals resample whole stints. Per-stint slopes are observed "
        "pace trends and can still include fuel, traffic, and track evolution effects."
    )


if __name__ == "__main__":
    run()
