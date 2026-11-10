# Pitwall-AI
A race strategy assistant focused on Ferrari F1 team using Formula 1 race data starting from 2025 for Charles Leclerc (`LEC`) and Lewis Hamilton (`HAM`). The first milestone downloads and cleans race laps, the next trains a baseline model to estimate lap time from driver, circuit, tyre compound, tyre age, and race lap. A later milestone will let an LLM explain the model's results without inventing the calculations.

## Current scope

- 2025 Grand Prix race sessions, Ferrari drivers only
- FastF1 lap data, cached locally
- Transparent filters for pit in/out laps, inaccurate/deleted laps, non-green track status, and lap 1
- Random Forest baseline with whole Grands Prix held out for evaluation

This is a learning and analysis project, not a real-time strategy system. Lap time is affected by traffic, fuel load, weather, car condition, and driver behavior as well as tyre wear. The model's recommendations must be presented with uncertainty and supporting data.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data
python -m src.model
```

FastF1 downloads/cache data locally on first run. Raw session data are not committed. The 2025 season is used, so missing or unavailable events are skipped and reported by the downloader.

## Next milestones

1. Inspect event coverage and cleaned lap counts for both drivers.
2. Compare the baseline against simple tyre-age-only and stint-level baselines.
3. Add a Streamlit view for race, driver, stint, predicted pace, and uncertainty.
4. Add an LLM tool that can query computed results and cite the selected race/laps.
