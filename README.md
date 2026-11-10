# Pitwall AI

A Ferrari-focused Formula 1 analytics project for Charles Leclerc (`LEC`) and Lewis Hamilton (`HAM`).
It downloads 2025 and 2026 race laps with FastF1, estimates **tyre degradation** per season, compound, driver
and stint with honest uncertainty, and fits a **race-pace model** that is scored against a naive baseline.
Results are explored in a Streamlit dashboard styled after F1 timing screens and Ferrari colours.

> Exploratory analysis, not a live strategy system. Lap time also depends on traffic, fuel, weather, car
> updates and driver behaviour, so every estimate is shown with its uncertainty.

## What it does

- **Data:** FastF1 race laps for both Ferrari drivers, with per-lap weather, cached locally.
- **Cleaning:** removes pit in/out laps, deleted/inaccurate laps, non-green laps and lap 1 (`src/data.py`), then
  dry-weather and outlier filters for analysis (`src/cleaning.py`).
- **Degradation (`src/degradation.py`):**
  - Huber-robust regression of lap time on tyre age, one slope per compound, with event-driver fixed effects,
    race lap and track/air temperature as controls.
  - 95% intervals from a **bootstrap that resamples whole stints**, since laps in one stint are not independent.
  - Per-stint Theil-Sen slopes, with and without a **fuel-burn correction** taken from the regression.
- **Pace model (`src/model.py`):** predicts lap time relative to the driver's race median, evaluated with
  cross-validation that holds out whole race weekends, against a baseline.
- **Dashboard (`app.py`):** filters by season, Grand Prix, driver and compound; five tabs (race pace, tyre
  degradation, race table, pace model, method).

## Results snapshot

Adjusted degradation in seconds lost per additional tyre lap (dry laps, 95% stint-bootstrap interval):

| Season | Soft | Medium | Hard |
|---|---|---|---|
| 2025 | 0.074 [0.048, 0.104] | 0.044 [0.028, 0.059] | 0.035 [0.018, 0.049] |
| 2026 | 0.037 [0.000, 0.071] | 0.032 [0.023, 0.055] | 0.030 [0.015, 0.057] |

Race-pace model (error in seconds, lower is better), 3,879 dry laps from 44 race weekends:

| Evaluation | Baseline MAE | Selected model MAE | Improvement |
|---|---|---|---|
| Grouped cross-validation (by weekend) | 0.734 | 0.526 | 28% |
| Train 2025, test 2026 | 0.690 | 0.528 | 24% |

The first version of the model predicted absolute lap time from the circuit name. On 2026 it scored 3.59 s MAE,
worse than simply using each circuit's 2025 average (3.20 s), and 12.4 s on laps at circuits it had never seen.
Predicting relative pace removes the circuit effect. The comparison is kept in `src/model.py` and in the dashboard.

Two methodological points worth knowing:

1. Classical regression intervals were about three times too narrow (for example 2025 hard: 0.030 to 0.039) because
   they treated laps as independent. The stint bootstrap widens them to 0.018 to 0.049.
2. Raw per-stint slopes understate wear because the car gets lighter every lap. The regression puts that effect at
   about -0.046 s per lap in 2025 and -0.039 s in 2026, which is what the fuel correction removes.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data          # download and clean laps (first run is slow, then cached)
python -m src.degradation   # degradation estimates and stint summaries
python -m src.model         # pace model, metrics and held-out predictions
streamlit run app.py
```

Tests (offline, no FastF1 needed): `pip install pytest && python -m pytest tests`

Outputs land in `data/processed/` (CSVs, `model_metrics.json`) and `models/` (not committed). The dashboard theme is
in `.streamlit/config.toml` and `src/theme.py`.

## Project layout

```
app.py                 Streamlit dashboard
src/data.py            FastF1 download and basic lap cleaning
src/cleaning.py        analysis-time cleaning shared by the modules below
src/degradation.py     degradation regression, cluster bootstrap, stint slopes
src/model.py           relative-pace model, grouped CV, baselines
src/theme.py           F1 / Ferrari palette, CSS and Plotly styling
tests/                 unit tests for cleaning and estimation
```

## Notes and limits

- Two seasons from one team: some compound and driver groups have few stints, and their intervals are wide.
- 2026 follows major regulation changes, so 2025 to 2026 transfer is a genuine test, not a given.
- Fonts: the official F1 and Ferrari typefaces are proprietary, so the dashboard uses Titillium Web. Swap the
  `FONT_DISPLAY` / `FONT_BODY` values in `src/theme.py` if you hold a licence for another font.

## Next steps

1. Add an LLM tool that answers questions from the computed results and cites the selected race and laps.
2. Model traffic with gap-to-car-ahead data (needs telemetry or position data).
3. Extend to the full grid to test whether Ferrari's degradation is unusual.
