# baseline ML model for race lap-time estimation

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = ROOT / "data" / "processed" / "ferrari_2025_race_laps.csv"
TEST_DATA_PATH = ROOT / "data" / "processed" / "ferrari_2026_race_laps.csv"
MODEL_PATH = ROOT / "models" / "lap_time_model.joblib"
FEATURES = ["Driver", "EventName", "Compound", "TyreLife", "LapNumber"]
TARGET = "LapTimeSeconds"


def train() -> None:
    train_data = pd.read_csv(TRAIN_DATA_PATH).dropna(subset=FEATURES + [TARGET])
    test_data = pd.read_csv(TEST_DATA_PATH).dropna(subset=FEATURES + [TARGET])
    x_train, y_train = train_data[FEATURES], train_data[TARGET]
    x_test, y_test = test_data[FEATURES], test_data[TARGET]

    categorical = ["Driver", "EventName", "Compound"]
    numeric = ["TyreLife", "LapNumber"]
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    model = Pipeline([
        ("features", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(f"Trained on {len(train_data)} cleaned 2025 laps.")
    print(f"Evaluated on {len(test_data)} cleaned 2026 laps.")
    print(f"2026 MAE: {mean_absolute_error(y_test, pred):.3f}s")
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print(f"2026 RMSE: {rmse:.3f}s")
    for driver in sorted(test_data["Driver"].unique()):
        mask = test_data["Driver"].eq(driver)
        driver_mae = mean_absolute_error(y_test[mask], pred[mask.to_numpy()])
        print(f"{driver} 2026 MAE: {driver_mae:.3f}s ({mask.sum()} laps)")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
