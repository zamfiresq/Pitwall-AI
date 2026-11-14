"""Tests for the what-if predictor, strategy simulator and experiment log."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import make_pipeline
from src.tracking import log_run, read_runs
from src.whatif import enumerate_strategies, export_linear_model, predict_linear, simulate_strategies
from sklearn.linear_model import Ridge

MAX_STINT = {"SOFT": 25, "MEDIUM": 32, "HARD": 36}
DEG = {"SOFT": (0.07, 0.05, 0.09), "MEDIUM": (0.045, 0.03, 0.06), "HARD": (0.035, 0.02, 0.05)}
OFFSETS = {"SOFT": -0.7, "MEDIUM": -0.35, "HARD": 0.0}


def test_exported_linear_model_matches_sklearn():
    rng = np.random.default_rng(0)
    n = 400
    data = pd.DataFrame({
        "Compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n), "Progress": rng.uniform(0, 1, n),
        "TyreLife": rng.integers(1, 35, n).astype(float)})
    target = -1.5 * data["Progress"] + 0.04 * data["TyreLife"] + rng.normal(0, 0.2, n)
    pipeline = make_pipeline(["Progress", "TyreLife"], ["Compound"], Ridge(alpha=1.0)).fit(data, target)
    spec = export_linear_model(pipeline, ["Progress", "TyreLife"], ["Compound"])
    for compound in ("SOFT", "MEDIUM", "HARD"):
        frame = pd.DataFrame({"Compound": compound, "Progress": np.linspace(0.1, 0.9, 9), "TyreLife": np.arange(1, 10.0)})
        assert np.allclose(predict_linear(spec, compound, frame["Progress"], frame["TyreLife"]), pipeline.predict(frame))


def test_strategies_respect_rules():
    plans = enumerate_strategies(57, ["SOFT", "MEDIUM", "HARD"], MAX_STINT)
    assert plans
    for compounds, lengths in plans:
        assert len(set(compounds)) >= 2 and sum(lengths) == 57
        assert all(n >= 8 and n <= MAX_STINT[c] for c, n in zip(compounds, lengths))


def test_pit_loss_and_wear_drive_the_number_of_stops():
    cheap_stops = simulate_strategies(57, DEG, OFFSETS, pit_loss=2.0, max_stint=MAX_STINT, draws=300)
    costly_stops = simulate_strategies(57, DEG, OFFSETS, pit_loss=60.0, max_stint=MAX_STINT, draws=300)
    assert costly_stops.iloc[0]["Stops"] == 1
    assert cheap_stops.iloc[0]["Stops"] >= costly_stops.iloc[0]["Stops"]
    assert cheap_stops["ProbabilityBest"].sum() <= 1.0 + 1e-9
    assert costly_stops["ExpectedRaceTimeLoss_s"].is_monotonic_increasing


def test_equivalent_orderings_are_one_strategy():
    table = simulate_strategies(57, DEG, OFFSETS, 22.0, MAX_STINT, draws=300, top=50)
    keys = [tuple(sorted(zip(row.Compounds, row.Stints))) for row in table.itertuples()]
    assert len(keys) == len(set(keys))


def test_infeasible_race_returns_empty_table():
    assert simulate_strategies(200, DEG, OFFSETS, 22.0, MAX_STINT, max_stops=1).empty


def test_experiment_log_round_trip():
    path = Path(tempfile.mkdtemp()) / "runs.jsonl"
    log_run("demo", {"k": 1}, {"mae": np.float64(0.5), "n": np.int64(3)}, path)
    rows = read_runs(path)
    assert rows[0]["run"] == "demo" and rows[0]["metrics"] == {"mae": 0.5, "n": 3}
    json.dumps(rows)
