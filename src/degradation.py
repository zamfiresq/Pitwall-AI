# Tyre-degradation estimates by season, compound, driver and stint.
#
# Two complementary views are produced:
#
# 1. A season-level *adjusted* estimate: a Huber-robust regression of lap time on
#    tyre age (one slope per compound) with event-driver fixed effects, race lap
#    (fuel burn / track evolution) and track/air temperature as controls.
#    Its 95% interval comes from a cluster bootstrap that resamples whole stints.
#    Laps inside a stint are strongly correlated, so classical regression
#    standard errors treat them as independent and are far too narrow.
#
# 2. Per-stint *observed* Theil-Sen slopes, summarised per race / driver / season
#    with stint-level bootstrap intervals. A fuel-corrected version subtracts the
#    race-lap effect estimated in (1), because lap time falls as fuel burns and
#    that hides part of the tyre wear in raw slopes.
#
# These are association estimates. They do not isolate tyre wear from every
# race effect (traffic, strategy, car updates), so treat them with uncertainty.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from src.cleaning import clean_for_analysis

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_estimates.csv"
STINT_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_stint_slopes.csv"
RACE_DRIVER_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_by_race_driver.csv"
DRIVER_OUTPUT_PATH = DATA_DIR / "ferrari_tyre_degradation_by_driver.csv"

YEARS = (2025, 2026)
MIN_STINT_LAPS = 5
MIN_TYRE_AGE_SPAN = 3
BOOTSTRAP_REPLICATES = 2000            # stint-slope summaries (cheap)
REGRESSION_BOOTSTRAP_REPLICATES = 1000  # full regression refits
CLUSTER_BY = "stint"                    # "stint" or "weekend" (more conservative)
HUBER_T = 1.345


def prepare_data(year: int) -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / f"ferrari_{year}_race_laps.csv")
    data = clean_for_analysis(data)

    # Ignore very short stints and stints with too little tyre-age variation.
    stint_keys = ["EventName", "Driver", "Stint", "Compound"]
    stats = data.groupby(stint_keys, observed=True)["TyreLife"].agg(["count", "min", "max"])
    valid_stints = stats[
        (stats["count"] >= MIN_STINT_LAPS)
        & ((stats["max"] - stats["min"]) >= MIN_TYRE_AGE_SPAN)
    ].index
    data = data.set_index(stint_keys).loc[lambda frame: frame.index.isin(valid_stints)].reset_index()

    data["EventDriver"] = (
        data["Year"].astype(int).astype(str) + " | " + data["EventName"].astype(str)
        + " | " + data["Driver"].astype(str)
    )
    for output, source in (("TrackTemp", "Weather_TrackTemp"), ("AirTemp", "Weather_AirTemp")):
        values = pd.to_numeric(data[source], errors="coerce") if source in data else pd.Series(np.nan, index=data.index)
        median = values.median()
        data[output] = values.fillna(0.0 if pd.isna(median) else median)
    return data.reset_index(drop=True)


def huber_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    t: float = HUBER_T,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Huber M-estimator with one fixed effect per group (no intercept in X).

    Iteratively re-weighted least squares with a MAD scale estimate, equivalent
    to ``statsmodels`` ``RLM(..., HuberT())`` with group dummies. Fixed effects are
    removed with a weighted within-group transform, which is exact and fast
    enough to refit hundreds of times inside a bootstrap.
    """
    codes, _ = pd.factorize(groups)
    n_groups = int(codes.max()) + 1
    n = len(y)

    def weighted_group_means(values: np.ndarray, w: np.ndarray) -> np.ndarray:
        values = values.reshape(n, -1)
        totals = np.maximum(np.bincount(codes, weights=w, minlength=n_groups), 1e-12)
        return np.column_stack([
            np.bincount(codes, weights=w * values[:, j], minlength=n_groups) / totals
            for j in range(values.shape[1])
        ])

    weights = np.ones(n)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        x_within = X - weighted_group_means(X, weights)[codes]
        y_within = y - weighted_group_means(y, weights)[codes].ravel()
        root = np.sqrt(weights)
        new_beta = np.linalg.lstsq(x_within * root[:, None], y_within * root, rcond=None)[0]

        residual = y_within - x_within @ new_beta
        scale = max(np.median(np.abs(residual)) / 0.6745, 1e-12)
        u = np.abs(residual) / scale
        weights = np.where(u <= t, 1.0, t / u)

        converged = np.max(np.abs(new_beta - beta)) < tol
        beta = new_beta
        if converged:
            break
    return beta


def build_design(data: pd.DataFrame, compounds: list[str]) -> tuple[np.ndarray, int, int]:
    """Design matrix: compound shifts, one tyre-age slope per compound, controls.

    Returns ``(X, first_slope_column, race_lap_column)``.
    """
    columns = [(data["Compound"] == compound).astype(float).to_numpy() for compound in compounds[1:]]
    first_slope = len(columns)
    columns += [np.where(data["Compound"] == c, data["TyreLife"], 0.0) for c in compounds]
    race_lap = len(columns)
    columns += [data[name].to_numpy(dtype=float) for name in ("LapNumber", "TrackTemp", "AirTemp")]
    return np.column_stack(columns), first_slope, race_lap


def cluster_bootstrap(
    data: pd.DataFrame,
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    compounds: list[str],
    first_slope: int,
    rng: np.random.Generator,
    replicates: int = REGRESSION_BOOTSTRAP_REPLICATES,
    cluster_by: str = CLUSTER_BY,
) -> np.ndarray:
    """Refit the regression on resampled clusters; returns draws of the slopes."""
    if cluster_by == "weekend":
        key = data["EventName"].astype(str)
    else:
        key = data["EventName"].astype(str) + "|" + data["Driver"].astype(str) + "|" + data["Stint"].astype(str)
    cluster_codes, uniques = pd.factorize(key)
    rows_by_cluster = [np.flatnonzero(cluster_codes == i) for i in range(len(uniques))]
    compound_codes = pd.Categorical(data["Compound"], categories=compounds).codes

    draws = []
    for _ in range(replicates):
        picked = rng.integers(0, len(rows_by_cluster), len(rows_by_cluster))
        rows = np.concatenate([rows_by_cluster[i] for i in picked])
        if len(np.unique(compound_codes[rows])) < len(compounds):
            continue  # a compound vanished from this resample; its slope is undefined
        beta = huber_fixed_effects(y[rows], X[rows], groups[rows])
        draws.append(beta[first_slope:first_slope + len(compounds)])
    return np.asarray(draws)


def fit_season(year: int, rng: np.random.Generator) -> tuple[list[dict[str, object]], float]:
    data = prepare_data(year)
    compounds = sorted(data["Compound"].astype(str).unique())
    if len(compounds) < 2:
        raise RuntimeError(f"Not enough dry tyre compounds in {year} to estimate degradation.")

    y = data["LapTimeSeconds"].to_numpy(dtype=float)
    groups = data["EventDriver"].to_numpy()
    X, first_slope, race_lap = build_design(data, compounds)

    beta = huber_fixed_effects(y, X, groups)
    slopes = beta[first_slope:first_slope + len(compounds)]
    race_lap_effect = float(beta[race_lap])

    draws = cluster_bootstrap(data, y, X, groups, compounds, first_slope, rng)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    print(f"{year}: {len(draws)}/{REGRESSION_BOOTSTRAP_REPLICATES} bootstrap refits used "
          f"(cluster = {CLUSTER_BY}); race-lap effect {race_lap_effect:+.4f} s/lap")

    estimates: list[dict[str, object]] = []
    for i, compound in enumerate(compounds):
        subset = data[data["Compound"].astype(str) == compound]
        estimates.append({
            "Year": year,
            "Compound": compound,
            "DegradationSecondsPerTyreLap": float(slopes[i]),
            "CI95Lower": float(lower[i]),
            "CI95Upper": float(upper[i]),
            "CleanLaps": int(len(subset)),
            "Stints": int(subset[["EventName", "Driver", "Stint"]].drop_duplicates().shape[0]),
            "Weekends": int(subset["EventName"].nunique()),
            "RaceLapEffectSecondsPerLap": race_lap_effect,
            "Model": (
                "Huber regression; controls: event-driver fixed effects, race lap, track/air "
                f"temperature; 95% CI: {CLUSTER_BY}-cluster bootstrap"
            ),
        })
    return estimates, race_lap_effect


def calculate_stint_slopes(year: int, race_lap_effect: float) -> pd.DataFrame:
    """Robust observed pace-vs-tyre-age slope for every eligible stint."""
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
            # Within a stint, tyre age and race lap rise together, so the observed slope
            # contains the (negative) fuel-burn effect; subtracting it leaves the tyre part.
            "FuelCorrectedSlopeSecondsPerTyreLap": float(slope - race_lap_effect),
        })
    return pd.DataFrame(records)


def bootstrap_stint_median(
    grouped: pd.DataFrame,
    group_columns: list[str],
    seed: int,
    value_column: str,
    median_name: str,
    lower_name: str,
    upper_name: str,
) -> pd.DataFrame:
    """Summarise stint slopes; resample whole stints to preserve lap dependence."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for keys, group in grouped.groupby(group_columns, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        slopes = group[value_column].dropna().to_numpy(dtype=float)
        count = len(slopes)
        row = dict(zip(group_columns, keys))
        row["Stints"] = count
        row[median_name] = float(np.median(slopes)) if count else np.nan
        if count >= 2:
            draws = rng.choice(slopes, size=(BOOTSTRAP_REPLICATES, count), replace=True)
            medians = np.median(draws, axis=1)
            row[lower_name] = float(np.quantile(medians, 0.025))
            row[upper_name] = float(np.quantile(medians, 0.975))
        else:
            # A single stint cannot provide a between-stint uncertainty estimate.
            row[lower_name] = np.nan
            row[upper_name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_stints(stints: pd.DataFrame, group_columns: list[str], seed: int) -> pd.DataFrame:
    """Raw and fuel-corrected stint summaries side by side."""
    raw = bootstrap_stint_median(
        stints, group_columns, seed, "ObservedSlopeSecondsPerTyreLap",
        "MedianObservedSlopeSecondsPerTyreLap", "ClusterBootstrap95Lower", "ClusterBootstrap95Upper",
    )
    corrected = bootstrap_stint_median(
        stints, group_columns, seed, "FuelCorrectedSlopeSecondsPerTyreLap",
        "MedianFuelCorrectedSlopeSecondsPerTyreLap", "FuelCorrectedBootstrap95Lower",
        "FuelCorrectedBootstrap95Upper",
    ).drop(columns=["Stints"])
    return raw.merge(corrected, on=group_columns, how="left").sort_values(group_columns)


def run() -> None:
    rng = np.random.default_rng(42)
    estimates: list[dict[str, object]] = []
    stint_tables: list[pd.DataFrame] = []
    for year in YEARS:
        season_estimates, race_lap_effect = fit_season(year, rng)
        estimates.extend(season_estimates)
        stint_tables.append(calculate_stint_slopes(year, race_lap_effect))

    summary = pd.DataFrame(estimates).sort_values(["Year", "Compound"])
    summary.to_csv(OUTPUT_PATH, index=False)

    stint_slopes = pd.concat(stint_tables, ignore_index=True)
    stint_slopes.to_csv(STINT_OUTPUT_PATH, index=False)
    summarise_stints(stint_slopes, ["Year", "EventName", "Driver", "Compound"], seed=42).to_csv(
        RACE_DRIVER_OUTPUT_PATH, index=False)
    by_driver = summarise_stints(stint_slopes, ["Year", "Driver", "Compound"], seed=43)
    by_driver.to_csv(DRIVER_OUTPUT_PATH, index=False)

    print("\nAdjusted degradation, seconds per additional tyre lap (95% cluster-bootstrap CI):")
    print(summary[[
        "Year", "Compound", "DegradationSecondsPerTyreLap", "CI95Lower", "CI95Upper",
        "CleanLaps", "Stints", "Weekends",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved estimates to {OUTPUT_PATH}")
    print("\nObserved vs fuel-corrected median stint slopes by season, driver and compound:")
    print(by_driver[[
        "Year", "Driver", "Compound", "Stints",
        "MedianObservedSlopeSecondsPerTyreLap", "MedianFuelCorrectedSlopeSecondsPerTyreLap",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved per-stint slopes to {STINT_OUTPUT_PATH}")
    print(f"Saved race/driver summaries to {RACE_DRIVER_OUTPUT_PATH}")
    print(f"Saved driver summaries to {DRIVER_OUTPUT_PATH}")
    print(
        "Intervals resample whole stints. Per-stint slopes are observed pace trends and can "
        "still include traffic, strategy and track-evolution effects."
    )


if __name__ == "__main__":
    run()
