# Pitwall-AI
A Ferrari-focused F1 analysis project for Charles Leclerc (`LEC`) and Lewis Hamilton (`HAM`). It downloads 2025 and 2026 race laps, compares a baseline lap-time model across seasons, and estimates tyre-age degradation separately for each season and compound.

## Current scope

- 2025 and 2026 Grand Prix race sessions, Ferrari drivers only
- FastF1 lap data, cached locally
- Filters for pit in/out laps, inaccurate/deleted laps, non-green track status, and lap 1
- A Random Forest baseline trained on 2025 and evaluated on 2026
- A robust, season-level tyre degradation estimate per compound, controlling for event/driver pace, race lap, and weather where available

This is a learning and analysis project, not a real-time strategy system. Lap time is affected by traffic, fuel load, weather, car condition, and driver behavior as well as tyre wear. The model's recommendations must be presented with uncertainty and supporting data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data
python -m src.model
python -m src.degradation
```

FastF1 downloads/cache data locally on first run. The data downloader includes per-lap weather observations, so rerun `python -m src.data` after installing dependencies to refresh the season CSVs. Raw data and model files are not committed. Missing or unavailable events are skipped and reported by the downloader.

The adjusted season-level estimates are saved to `data/processed/ferrari_tyre_degradation_estimates.csv`. Robust observed slopes are also saved per stint, per race/pilot/compound, and per season/pilot/compound. Their percentile intervals resample complete stints, keeping laps from the same stint together. A positive value means lap time tends to rise as tyre age increases. The per-stint slopes are observed pace trends, not pure physical measurements of tyre wear: fuel effects, traffic, car changes, and strategy can still affect lap times. Groups with only one stint have no cluster-bootstrap interval because there is not enough independent data to estimate between-stint uncertainty.

## Next milestones

1. Inspect estimates and confidence intervals by season and compound.
2. Validate estimates by holding out whole race weekends and compare with a simple per-stint baseline.
3. Add a Streamlit view for race, driver, stint, predicted pace, and uncertainty.
4. Add an LLM tool that can query computed results and cite the selected race/laps.
