"""Fast, offline tests for the cleaning and estimation code (no FastF1 or network)."""

import numpy as np
import pandas as pd

from src.cleaning import clean_for_analysis
from src.degradation import (
    bootstrap_stint_median, build_design, cluster_bootstrap, huber_fixed_effects,
)

TRUE_TYRE, TRUE_FUEL = 0.05, -0.04


def synthetic_laps(seed: int = 0) -> pd.DataFrame:
    """Race laps with a known tyre slope, known fuel effect, group offsets and outliers."""
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(10):
        offset = 80 + 2 * race
        for driver in ("LEC", "HAM"):
            lap = 2
            for stint in range(1, 4):
                tyre0 = int(rng.integers(1, 9))  # used tyres break the tyre-age / race-lap collinearity
                compound = ["HARD", "MEDIUM"][stint % 2]
                for i in range(int(rng.integers(8, 14))):
                    noise = rng.normal(0, 0.1) + (5.0 if rng.random() < 0.05 else 0.0)
                    rows.append({
                        "EventName": f"GP{race}", "Driver": driver, "Stint": stint, "Compound": compound,
                        "TyreLife": tyre0 + i, "LapNumber": lap, "TrackTemp": 30.0, "AirTemp": 20.0,
                        "LapTimeSeconds": offset + TRUE_TYRE * (tyre0 + i) + TRUE_FUEL * lap + noise,
                    })
                    lap += 1
    data = pd.DataFrame(rows)
    data["EventDriver"] = data["EventName"] + "|" + data["Driver"]
    return data


def test_huber_recovers_known_slope_despite_outliers():
    data = synthetic_laps()
    X = data[["TyreLife", "LapNumber"]].to_numpy(dtype=float)
    beta = huber_fixed_effects(data["LapTimeSeconds"].to_numpy(), X, data["EventDriver"].to_numpy())
    assert abs(beta[0] - TRUE_TYRE) < 0.01
    assert abs(beta[1] - TRUE_FUEL) < 0.01


def test_cluster_bootstrap_interval_is_centred_and_covers_truth():
    """Coverage is a property of many datasets, so check several seeds, not one."""
    compounds = ["HARD", "MEDIUM"]
    covered, total = 0, 0
    for seed in range(8):
        data = synthetic_laps(seed)
        X, first_slope, _ = build_design(data, compounds)
        y, groups = data["LapTimeSeconds"].to_numpy(), data["EventDriver"].to_numpy()
        point = huber_fixed_effects(y, X, groups)[first_slope:first_slope + 2]
        draws = cluster_bootstrap(data, y, X, groups, compounds, first_slope,
                                  np.random.default_rng(seed), replicates=100)
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        assert ((lower < point) & (point < upper)).all()  # interval brackets its own estimate
        covered += int(((lower < TRUE_TYRE) & (TRUE_TYRE < upper)).sum())
        total += 2
    assert covered / total >= 0.7  # nominal 95%; loose bound keeps the test stable


def test_cleaning_removes_wet_rain_slow_laps_and_sensor_zeros():
    base = {"Year": 2026, "EventName": "Test GP", "Driver": "LEC", "Stint": 1, "TyreLife": 5.0,
            "LapNumber": 10.0, "Weather_Rainfall": False, "Weather_TrackTemp": 40.0, "Weather_AirTemp": 25.0}
    laps = pd.DataFrame([
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 90.0},
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 90.4},
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 89.8},
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 99.0},                        # slow lap
        {**base, "Compound": "INTERMEDIATE", "LapTimeSeconds": 95.0},                  # wet compound
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 90.1, "Weather_Rainfall": True},
        {**base, "Compound": "MEDIUM", "LapTimeSeconds": 90.2, "Weather_TrackTemp": 0.0},
    ])
    cleaned = clean_for_analysis(laps)
    assert set(cleaned["Compound"]) == {"MEDIUM"}
    assert cleaned["LapTimeSeconds"].max() < 95
    assert len(cleaned) == 4
    assert cleaned["Weather_TrackTemp"].isna().sum() == 1  # sentinel became missing, lap kept


def test_stint_summary_has_no_interval_for_single_stint():
    stints = pd.DataFrame({"Group": ["a", "b", "b", "b"], "Slope": [0.05, 0.02, 0.04, 0.03]})
    out = bootstrap_stint_median(stints, ["Group"], 0, "Slope", "Median", "Low", "High").set_index("Group")
    assert np.isnan(out.loc["a", "Low"]) and out.loc["a", "Median"] == 0.05
    assert out.loc["b", "Low"] <= out.loc["b", "Median"] <= out.loc["b", "High"]
