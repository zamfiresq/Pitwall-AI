# Race-pace model: how much slower/faster than their own race median is a Ferrari
# on a given lap, given tyre age, compound, race progress and weather?
#
# Why relative pace instead of absolute lap time: absolute lap time is dominated
# by circuit length, so a model that sees the circuit name mostly memorises
# circuits and fails on new ones. The original absolute-time Random Forest is
# kept below as a reference so the comparison stays visible.
#
# Evaluation is grouped by race weekend (a weekend is never in both train and
# test) and every model is compared with a naive baseline.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean_for_analysis, load_season_laps
from src.tracking import log_run
from src.whatif import export_linear_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "lap_time_model.joblib"
METRICS_PATH = DATA_DIR / "model_metrics.json"
PREDICTIONS_PATH = DATA_DIR / "model_oof_predictions.csv"

YEARS = (2025, 2026)
TARGET = "PaceDelta"
CV_SPLITS = 8
SIMPLICITY_TOLERANCE = 0.01  # prefer the simpler model if within 1% of the best CV MAE
WHATIF_MODEL = "Linear: + tyre age"  # interpretable model behind the dashboard's what-if tool
INTERVAL_LEVELS = (0.8, 0.9)

# name -> (numeric features, categorical features, estimator factory, complexity rank)
MODEL_SPECS = {
    "Linear: race progress + compound": (
        ["Progress"], ["Compound"], lambda: Ridge(alpha=1.0), 1),
    "Linear: + tyre age": (
        ["Progress", "TyreLife"], ["Compound"], lambda: Ridge(alpha=1.0), 2),
    "Gradient boosting: + driver & weather": (
        ["Progress", "TyreLife", "LapNumber", "TrackTempDelta", "Weather_AirTemp", "Weather_Humidity"],
        ["Compound", "Driver"],
        lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, random_state=42),
        3),
}
BASELINE = "Baseline: race median pace"


def build_dataset() -> pd.DataFrame:
    """Dry, representative laps with a relative-pace target."""
    data = clean_for_analysis(load_season_laps(DATA_DIR, YEARS))
    data["Race"] = data["Year"].astype(int).astype(str) + " | " + data["EventName"].astype(str)
    reference = data.groupby(["Race", "Driver"])["LapTimeSeconds"].transform("median")
    data[TARGET] = data["LapTimeSeconds"] - reference
    data["Progress"] = data["LapNumber"] / data.groupby("Race")["LapNumber"].transform("max")
    data["TrackTempDelta"] = data["Weather_TrackTemp"] - data.groupby("Race")["Weather_TrackTemp"].transform("mean")
    return data.reset_index(drop=True)


def make_pipeline(numeric: list[str], categorical: list[str], estimator) -> Pipeline:
    scale = isinstance(estimator, Ridge)
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
    ])
    return Pipeline([("features", prep), ("regressor", estimator)])


def score(actual: np.ndarray, predicted: np.ndarray, baseline_mae: float | None = None) -> dict[str, float]:
    mae = float(mean_absolute_error(actual, predicted))
    result = {
        "mae": mae,
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }
    result["improvement_pct"] = 0.0 if baseline_mae is None else float(100 * (1 - mae / baseline_mae))
    return result


def cross_validate(data: pd.DataFrame) -> tuple[list[dict], dict[str, np.ndarray]]:
    splits = min(CV_SPLITS, data["Race"].nunique())
    folds = list(GroupKFold(n_splits=splits).split(data, groups=data["Race"]))
    actual = data[TARGET].to_numpy()
    rows = [{"name": BASELINE, **score(actual, np.zeros(len(data)))}]
    baseline_mae = rows[0]["mae"]
    out_of_fold: dict[str, np.ndarray] = {}
    for name, (numeric, categorical, factory, rank) in MODEL_SPECS.items():
        predictions = np.zeros(len(data))
        for train_idx, test_idx in folds:
            fitted = make_pipeline(numeric, categorical, factory()).fit(
                data.iloc[train_idx][numeric + categorical], actual[train_idx])
            predictions[test_idx] = fitted.predict(data.iloc[test_idx][numeric + categorical])
        out_of_fold[name] = predictions
        rows.append({"name": name, "complexity": rank, **score(actual, predictions, baseline_mae)})
    return rows, out_of_fold


def season_holdout(data: pd.DataFrame) -> list[dict]:
    """Train on 2025, test on 2026: does the relationship carry across seasons?"""
    train, test = data[data["Year"] == YEARS[0]], data[data["Year"] == YEARS[1]]
    actual = test[TARGET].to_numpy()
    rows = [{"name": BASELINE, **score(actual, np.zeros(len(test)))}]
    baseline_mae = rows[0]["mae"]
    for name, (numeric, categorical, factory, _) in MODEL_SPECS.items():
        fitted = make_pipeline(numeric, categorical, factory()).fit(
            train[numeric + categorical], train[TARGET])
        rows.append({"name": name, **score(actual, fitted.predict(test[numeric + categorical]), baseline_mae)})
    return rows


def legacy_reference() -> dict[str, float]:
    """The original absolute lap-time Random Forest versus a naive per-circuit mean."""
    features = ["Driver", "EventName", "Compound", "TyreLife", "LapNumber"]
    target = "LapTimeSeconds"
    train = pd.read_csv(DATA_DIR / f"ferrari_{YEARS[0]}_race_laps.csv").dropna(subset=features + [target])
    test = pd.read_csv(DATA_DIR / f"ferrari_{YEARS[1]}_race_laps.csv").dropna(subset=features + [target])
    pipeline = Pipeline([
        ("features", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Driver", "EventName", "Compound"]),
            ("numeric", StandardScaler(), ["TyreLife", "LapNumber"]),
        ])),
        ("regressor", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    ]).fit(train[features], train[target])
    predicted = pipeline.predict(test[features])
    circuit_mean = train.groupby("EventName")[target].mean()
    naive = test["EventName"].map(circuit_mean).fillna(train[target].mean())
    seen = test["EventName"].isin(circuit_mean.index).to_numpy()
    return {
        "random_forest_mae": float(mean_absolute_error(test[target], predicted)),
        "naive_circuit_mean_mae": float(mean_absolute_error(test[target], naive)),
        "random_forest_mae_seen_circuits": float(mean_absolute_error(test[target][seen], predicted[seen])),
        "random_forest_mae_unseen_circuits": (
            float(mean_absolute_error(test[target][~seen], predicted[~seen])) if (~seen).any() else float("nan")),
        "unseen_circuit_laps": int((~seen).sum()),
        "test_laps": int(len(test)),
    }


def ablation(data: pd.DataFrame) -> list[dict]:
    """Cross-validated MAE of the full boosting model after removing one feature group at a time."""
    numeric, categorical, factory, _ = MODEL_SPECS["Gradient boosting: + driver & weather"]
    groups = {
        "Tyre age": ["TyreLife"],
        "Race progress / fuel": ["Progress", "LapNumber"],
        "Compound": ["Compound"],
        "Driver": ["Driver"],
        "Weather": ["TrackTempDelta", "Weather_AirTemp", "Weather_Humidity"],
    }
    folds = list(GroupKFold(n_splits=min(CV_SPLITS, data["Race"].nunique())).split(data, groups=data["Race"]))
    actual = data[TARGET].to_numpy()

    def cv_mae(num: list[str], cat: list[str]) -> float:
        predictions = np.zeros(len(data))
        for train_idx, test_idx in folds:
            fitted = make_pipeline(num, cat, factory()).fit(data.iloc[train_idx][num + cat], actual[train_idx])
            predictions[test_idx] = fitted.predict(data.iloc[test_idx][num + cat])
        return float(mean_absolute_error(actual, predictions))

    full = cv_mae(numeric, categorical)
    rows = [{"removed": "Nothing (full model)", "mae": full, "delta_mae": 0.0}]
    for label, removed in groups.items():
        mae = cv_mae([f for f in numeric if f not in removed], [f for f in categorical if f not in removed])
        rows.append({"removed": label, "mae": mae, "delta_mae": mae - full})
    return rows


def prediction_intervals(data: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Split-conformal style intervals from held-out residuals, plus honest coverage.

    Half-widths are quantiles of absolute held-out errors. Coverage is checked by leaving each race
    weekend out: its interval comes from the residuals of all the other weekends.
    """
    residual = np.abs(data[TARGET].to_numpy() - predictions)
    races = data["Race"].to_numpy()
    half_width = {str(level): float(np.quantile(residual, level)) for level in INTERVAL_LEVELS}
    coverage = {}
    for level in INTERVAL_LEVELS:
        hits = np.zeros(len(data), dtype=bool)
        for race in np.unique(races):
            mask = races == race
            hits[mask] = residual[mask] <= np.quantile(residual[~mask], level)
        coverage[str(level)] = float(hits.mean())
    return {"half_width_s": half_width, "coverage": coverage}


def choose_model(rows: list[dict]) -> str:
    candidates = [row for row in rows if row["name"] != BASELINE]
    best = min(row["mae"] for row in candidates)
    acceptable = [row for row in candidates if row["mae"] <= best * (1 + SIMPLICITY_TOLERANCE)]
    return min(acceptable, key=lambda row: row["complexity"])["name"]


def train() -> None:
    data = build_dataset()
    print(f"{len(data)} dry laps from {data['Race'].nunique()} race weekends; "
          f"target = lap time minus the driver's race median (std {data[TARGET].std():.3f}s)")

    cv_rows, out_of_fold = cross_validate(data)
    holdout_rows = season_holdout(data)
    chosen = choose_model(cv_rows)
    legacy = legacy_reference()

    numeric, categorical, factory, _ = MODEL_SPECS[chosen]
    final_model = make_pipeline(numeric, categorical, factory()).fit(
        data[numeric + categorical], data[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": numeric + categorical, "target": TARGET, "name": chosen}, MODEL_PATH)

    by_compound = []
    chosen_predictions = out_of_fold[chosen]
    for compound, group in data.groupby("Compound"):
        idx = group.index.to_numpy()
        by_compound.append({
            "compound": compound, "laps": int(len(idx)),
            "baseline_mae": float(mean_absolute_error(data[TARGET].iloc[idx], np.zeros(len(idx)))),
            "model_mae": float(mean_absolute_error(data[TARGET].iloc[idx], chosen_predictions[idx])),
        })

    whatif_numeric, whatif_categorical, whatif_factory, _ = MODEL_SPECS[WHATIF_MODEL]
    whatif_pipeline = make_pipeline(whatif_numeric, whatif_categorical, whatif_factory()).fit(
        data[whatif_numeric + whatif_categorical], data[TARGET])
    intervals = prediction_intervals(data, out_of_fold[WHATIF_MODEL])
    feature_ablation = ablation(data)

    metrics = {
        "target": "Lap time minus the driver's median lap time in that race (seconds)",
        "laps": int(len(data)),
        "race_weekends": int(data["Race"].nunique()),
        "cv_scheme": f"{min(CV_SPLITS, data['Race'].nunique())}-fold, grouped by race weekend",
        "cross_validation": cv_rows,
        "season_holdout": {"train": YEARS[0], "test": YEARS[1], "models": holdout_rows},
        "selected_model": chosen,
        "by_compound": by_compound,
        "legacy": legacy,
        "ablation": feature_ablation,
        "whatif": {
            "model_name": WHATIF_MODEL,
            "spec": export_linear_model(whatif_pipeline, whatif_numeric, whatif_categorical),
            "interval_half_width_s": intervals["half_width_s"],
            "interval_coverage": intervals["coverage"],
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    predictions = data[["Year", "EventName", "Driver", "LapNumber", "Compound", "TyreLife", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "ActualPaceDelta"})
    predictions["PredictedPaceDelta"] = chosen_predictions
    predictions["Model"] = chosen
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nGrouped cross-validation ({metrics['cv_scheme']}):")
    for row in cv_rows:
        print(f"  {row['name']:<42} MAE {row['mae']:.3f}s  RMSE {row['rmse']:.3f}s  vs baseline {row['improvement_pct']:+.1f}%")
    print(f"\nTrain {YEARS[0]} -> test {YEARS[1]}:")
    for row in holdout_rows:
        print(f"  {row['name']:<42} MAE {row['mae']:.3f}s  vs baseline {row['improvement_pct']:+.1f}%")
    print(f"\nSelected (simplest within {SIMPLICITY_TOLERANCE:.0%} of best): {chosen}")
    print(
        f"\nReference, original absolute-lap-time Random Forest ({YEARS[0]} -> {YEARS[1]}): "
        f"MAE {legacy['random_forest_mae']:.2f}s vs naive circuit mean {legacy['naive_circuit_mean_mae']:.2f}s; "
        f"{legacy['random_forest_mae_unseen_circuits']:.1f}s on {legacy['unseen_circuit_laps']} laps at unseen circuits."
    )
    print("\nFeature ablation (boosting model; higher MAE = the removed feature mattered):")
    for row in feature_ablation:
        print(f"  remove {row['removed']:<24} MAE {row['mae']:.3f}s  ({row['delta_mae']:+.3f})")
    print("\nWhat-if intervals (held-out weekends): " + ", ".join(
        f"{int(float(level) * 100)}%: ±{width:.2f}s, actual coverage {intervals['coverage'][level]:.0%}"
        for level, width in intervals["half_width_s"].items()))
    log_run(
        "pace_model",
        {"cv_splits": CV_SPLITS, "years": list(YEARS), "selected_model": chosen},
        {"laps": len(data), "baseline_mae": cv_rows[0]["mae"],
         **{f"cv_mae[{row['name']}]": round(row["mae"], 4) for row in cv_rows[1:]}},
    )
    print(f"\nSaved model to {MODEL_PATH}\nSaved metrics to {METRICS_PATH}\nSaved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    train()
