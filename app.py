from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.assistant import chat_ui
from src.assistant.agent import LLMConfig, make_client
from src.assistant.tools import PitwallData
from src.tracking import read_runs
from src.whatif import predict_linear, simulate_strategies

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"]

st.set_page_config(
    page_title="Pitwall AI | Ferrari Tyre & Pace Analytics",
    page_icon="🏎️",
    layout="wide",
)
theme.inject_css()


@st.cache_data
def load_laps() -> pd.DataFrame:
    paths = sorted(DATA_DIR.glob("ferrari_20*_race_laps.csv"))
    if not paths:
        return pd.DataFrame()
    laps = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for column in ("Year", "LapNumber", "LapTimeSeconds", "TyreLife", "Stint"):
        if column in laps:
            laps[column] = pd.to_numeric(laps[column], errors="coerce")
    laps = laps.dropna(subset=["Year", "LapNumber", "LapTimeSeconds", "TyreLife"])
    laps["Season"] = laps["Year"].astype(int).astype(str)
    return laps


@st.cache_data
def load_table(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "Year" in table:
        table["Season"] = table["Year"].astype(int).astype(str)
    return table


@st.cache_data
def load_metrics() -> dict:
    path = DATA_DIR / "model_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_metrics_file(filename: str) -> dict:
    path = DATA_DIR / filename
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_resource
def load_pitwall_data() -> PitwallData:
    return PitwallData(DATA_DIR)


@st.cache_resource
def get_client():
    return make_client(LLMConfig.from_env())


def fmt_lap(seconds: float) -> str:
    minutes, rest = divmod(float(seconds), 60)
    return f"{int(minutes)}:{rest:06.3f}"


def only(frame: pd.DataFrame, year_choice, event_choice=None, driver_choice=None, compound_choice=None) -> pd.DataFrame:
    """Apply the sidebar filters to any table that has the matching columns."""
    if year_choice != "All seasons" and "Year" in frame:
        frame = frame[frame["Year"].astype(int) == int(year_choice)]
    if event_choice not in (None, "All Grands Prix") and "EventName" in frame:
        frame = frame[frame["EventName"].astype(str) == event_choice]
    if driver_choice not in (None, "Both") and "Driver" in frame:
        frame = frame[frame["Driver"].astype(str) == driver_choice]
    if compound_choice not in (None, "All compounds") and "Compound" in frame:
        frame = frame[frame["Compound"].astype(str) == compound_choice]
    return frame


theme.hero()

laps = load_laps()
if laps.empty:
    st.warning("No lap files found. Run `python -m src.data` from the project folder first.")
    st.stop()

stint_slopes = load_table("ferrari_tyre_stint_slopes.csv")
race_driver = load_table("ferrari_tyre_degradation_by_race_driver.csv")
driver_summary = load_table("ferrari_tyre_degradation_by_driver.csv")
adjusted = load_table("ferrari_tyre_degradation_estimates.csv")
oof = load_table("model_oof_predictions.csv")
metrics = load_metrics()

fuel_available = (
    not adjusted.empty and "RaceLapEffectSecondsPerLap" in adjusted
    and "FuelCorrectedSlopeSecondsPerTyreLap" in stint_slopes
)
lap_effect = adjusted.groupby("Year")["RaceLapEffectSecondsPerLap"].first().to_dict() if fuel_available else {}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Session filters")
    years = sorted(laps["Year"].dropna().astype(int).unique().tolist())
    year_choice = st.selectbox("Season", ["All seasons", *years], index=len(years))
    scoped = only(laps, year_choice)

    events = sorted(scoped["EventName"].dropna().astype(str).unique().tolist())
    event_choice = st.selectbox("Grand Prix", ["All Grands Prix", *events], index=1 if events else 0)

    drivers = sorted(scoped["Driver"].dropna().astype(str).unique().tolist())
    driver_choice = st.selectbox("Driver", ["Both", *drivers])

    compounds = [c for c in COMPOUND_ORDER if c in set(scoped["Compound"].astype(str))]
    compound_choice = st.selectbox("Tyre compound", ["All compounds", *compounds])

    st.header("Analysis options")
    fuel_on = st.checkbox(
        "Correct for fuel burn",
        value=fuel_available,
        disabled=not fuel_available,
        help="Cars get lighter every lap, which hides tyre wear. This removes the fuel/track-evolution "
             "effect estimated by the regression (about 0.04-0.05 s per lap).",
    )
    if not fuel_available:
        st.caption("Run `python -m src.degradation` to enable the fuel correction.")

    st.header("Tyre compounds")
    theme.compound_legend()

filtered = only(laps, year_choice, event_choice, driver_choice, compound_choice).copy()
filtered["StintKey"] = (
    filtered["Season"] + " · " + filtered["EventName"].astype(str) + " · "
    + filtered["Driver"].astype(str) + " · stint " + filtered["Stint"].fillna(-1).astype(int).astype(str)
)

# ---------------------------------------------------------------- KPI row
kpi = st.columns(4)
kpi[0].metric("Laps shown", f"{len(filtered):,}")
kpi[1].metric("Stints", f"{filtered['StintKey'].nunique():,}")
kpi[2].metric("Grands Prix", f"{filtered[['Year', 'EventName']].drop_duplicates().shape[0]:,}")
kpi[3].metric("Fastest lap", fmt_lap(filtered["LapTimeSeconds"].min()) if not filtered.empty else "—")

tab_pace, tab_deg, tab_table, tab_model, tab_lab, tab_ask = st.tabs(
    ["Race pace", "Tyre degradation", "Race table", "Pace model", "Strategy lab", "Ask Pitwall"]
)

# ---------------------------------------------------------------- race pace
with tab_pace:
    st.subheader("Lap time vs tyre age")
    st.caption("Each line is one stint. Pick a single Grand Prix in the sidebar for the clearest view.")
    if filtered.empty:
        st.info("No laps match the selected filters.")
    else:
        pace = filtered.sort_values(["StintKey", "TyreLife"]).copy()
        y_column, y_label = "LapTimeSeconds", "Lap time (s)"
        if fuel_on:
            pace["LapTimeFuelCorrected"] = pace["LapTimeSeconds"] - pace["Year"].map(lap_effect) * (pace["LapNumber"] - 1)
            y_column, y_label = "LapTimeFuelCorrected", "Fuel-corrected lap time (s)"
        hover_fields = [c for c in [
            "EventName", "Season", "Driver", "Stint", "LapNumber", "Compound", "Weather_TrackTemp",
        ] if c in pace.columns]
        pace_fig = px.line(
            pace, x="TyreLife", y=y_column, color="Compound", line_dash="Driver", line_group="StintKey",
            markers=True, hover_data=hover_fields,
            color_discrete_map=theme.COMPOUND_COLORS, category_orders={"Compound": COMPOUND_ORDER},
            labels={
                "TyreLife": "Tyre age (laps)", y_column: y_label, "Compound": "Compound", "Driver": "Driver",
                "Season": "Season", "LapNumber": "Race lap", "Weather_TrackTemp": "Track temp (°C)",
            },
        )
        pace_fig.update_traces(marker={"size": 6}, line={"width": 1.8})
        theme.style_fig(pace_fig, height=420)
        st.plotly_chart(pace_fig, width="stretch", config={"displayModeBar": False})
        if fuel_on:
            st.caption("Lap times are adjusted to a constant fuel load using the season's race-lap effect.")

# ---------------------------------------------------------------- degradation
with tab_deg:
    st.subheader("Degradation estimates")
    theme.note(
        "Seconds lost per additional lap of tyre age. Positive means the tyre gets slower as it wears. "
        "Estimates use dry-weather laps only; intermediate and wet running is excluded."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Adjusted estimate by season and compound**")
        chart_data = only(adjusted, year_choice)
        if chart_data.empty:
            st.info("No adjusted estimates found. Run `python -m src.degradation`.")
        else:
            chart_data = chart_data.copy()
            chart_data["ErrorUpper"] = (chart_data["CI95Upper"] - chart_data["DegradationSecondsPerTyreLap"]).clip(lower=0)
            chart_data["ErrorLower"] = (chart_data["DegradationSecondsPerTyreLap"] - chart_data["CI95Lower"]).clip(lower=0)
            model_fig = px.bar(
                chart_data, x="Compound", y="DegradationSecondsPerTyreLap", color="Season", barmode="group",
                error_y="ErrorUpper", error_y_minus="ErrorLower",
                color_discrete_map=theme.SEASON_COLORS, category_orders={"Compound": COMPOUND_ORDER},
                hover_data=[c for c in ["CleanLaps", "Stints", "CI95Lower", "CI95Upper"] if c in chart_data.columns],
                labels={"DegradationSecondsPerTyreLap": "Seconds per tyre lap", "Compound": "Compound"},
            )
            model_fig.add_hline(y=0, line_dash="dot", line_color=theme.MUTED)
            theme.style_fig(model_fig, height=330)
            st.plotly_chart(model_fig, width="stretch", config={"displayModeBar": False})
            st.caption(
                "Whiskers: 95% interval from a bootstrap that resamples whole stints, because laps within "
                "a stint are not independent."
            )

    with right:
        st.markdown("**Median stint slope by driver**")
        boot = only(driver_summary, year_choice)
        use_fuel_cols = fuel_on and "MedianFuelCorrectedSlopeSecondsPerTyreLap" in boot
        median_col = "MedianFuelCorrectedSlopeSecondsPerTyreLap" if use_fuel_cols else "MedianObservedSlopeSecondsPerTyreLap"
        lower_col = "FuelCorrectedBootstrap95Lower" if use_fuel_cols else "ClusterBootstrap95Lower"
        upper_col = "FuelCorrectedBootstrap95Upper" if use_fuel_cols else "ClusterBootstrap95Upper"
        if boot.empty:
            st.info("No driver summary found. Run `python -m src.degradation`.")
        else:
            boot = boot[boot["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
            boot["ErrorUpper"] = (boot[upper_col] - boot[median_col]).clip(lower=0).fillna(0)
            boot["ErrorLower"] = (boot[median_col] - boot[lower_col]).clip(lower=0).fillna(0)
            driver_fig = px.bar(
                boot, x="Compound", y=median_col, color="Driver", barmode="group",
                facet_col="Season" if year_choice == "All seasons" else None,
                error_y="ErrorUpper", error_y_minus="ErrorLower",
                color_discrete_map=theme.DRIVER_COLORS, category_orders={"Compound": COMPOUND_ORDER},
                hover_data=[c for c in ["Stints", lower_col, upper_col] if c in boot.columns],
                labels={median_col: "Seconds per tyre lap", "Compound": "Compound", "Stints": "Independent stints"},
            )
            driver_fig.add_hline(y=0, line_dash="dot", line_color=theme.MUTED)
            theme.style_fig(driver_fig, height=330)
            st.plotly_chart(driver_fig, width="stretch", config={"displayModeBar": False})
            st.caption(
                "Bars: median of per-stint slopes. Whiskers: 95% stint bootstrap. "
                + ("Fuel-corrected." if use_fuel_cols else "Raw slopes include the fuel-burn effect, so they understate wear.")
            )

    st.subheader("Spread of individual stint slopes")
    stint_view = only(stint_slopes, year_choice, event_choice, driver_choice, compound_choice)
    slope_col = (
        "FuelCorrectedSlopeSecondsPerTyreLap"
        if fuel_on and "FuelCorrectedSlopeSecondsPerTyreLap" in stint_view
        else "ObservedSlopeSecondsPerTyreLap"
    )
    if stint_view.empty:
        st.info("No eligible stints for the selected filters.")
    else:
        slope_fig = px.strip(
            stint_view, x="Compound", y=slope_col, color="Driver", stripmode="group",
            color_discrete_map=theme.DRIVER_COLORS, category_orders={"Compound": COMPOUND_ORDER},
            hover_data=[c for c in ["Season", "EventName", "Stint", "Laps", "TyreAgeMin", "TyreAgeMax"] if c in stint_view.columns],
            labels={
                slope_col: "Slope (s per tyre lap)", "Compound": "Compound", "EventName": "Grand Prix",
                "Laps": "Laps in stint", "TyreAgeMin": "Min tyre age", "TyreAgeMax": "Max tyre age",
            },
        )
        slope_fig.add_hline(y=0, line_dash="dot", line_color=theme.MUTED)
        theme.style_fig(slope_fig, height=350)
        st.plotly_chart(slope_fig, width="stretch", config={"displayModeBar": False})
        st.caption("Slopes are observed pace trends and can still include traffic, strategy and track evolution.")

# ---------------------------------------------------------------- race table
with tab_table:
    st.subheader("Bootstrap results by Grand Prix and driver")
    race_table = only(race_driver, year_choice, event_choice, driver_choice, compound_choice)
    use_fuel_cols = fuel_on and "MedianFuelCorrectedSlopeSecondsPerTyreLap" in race_table
    median_col = "MedianFuelCorrectedSlopeSecondsPerTyreLap" if use_fuel_cols else "MedianObservedSlopeSecondsPerTyreLap"
    lower_col = "FuelCorrectedBootstrap95Lower" if use_fuel_cols else "ClusterBootstrap95Lower"
    upper_col = "FuelCorrectedBootstrap95Upper" if use_fuel_cols else "ClusterBootstrap95Upper"
    if race_table.empty:
        st.info("No summary for the selected filters. Run `python -m src.degradation` if files are missing.")
    else:
        columns = ["Season", "EventName", "Driver", "Compound", "Stints", median_col, lower_col, upper_col]
        st.dataframe(
            race_table[columns].rename(columns={
                "EventName": "Grand Prix", "Stints": "Stints",
                median_col: "Median (s/tyre lap)", lower_col: "95% low", upper_col: "95% high",
            }).round(3),
            hide_index=True,
        )
        st.caption("Intervals are empty when a combination has only one eligible stint.")

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("Race-pace model")
    if not metrics:
        st.info("No model metrics found. Run `python -m src.model`.")
    else:
        theme.callout(
            "<b>What it predicts:</b> how many seconds slower or faster than their own race median a Ferrari "
            "is on a given lap, from tyre age, compound, race progress, driver and weather. Every score below "
            "comes from race weekends the model never saw during training."
        )
        cv_rows = pd.DataFrame(metrics["cross_validation"])
        baseline_mae = float(cv_rows.loc[cv_rows["name"].str.startswith("Baseline"), "mae"].iloc[0])
        chosen = metrics["selected_model"]
        chosen_mae = float(cv_rows.loc[cv_rows["name"] == chosen, "mae"].iloc[0])

        cards = st.columns(4)
        cards[0].metric("Baseline error (MAE)", f"{baseline_mae:.3f} s")
        cards[1].metric("Model error (MAE)", f"{chosen_mae:.3f} s")
        cards[2].metric("Improvement", f"{100 * (1 - chosen_mae / baseline_mae):.0f}%")
        cards[3].metric("Race weekends", f"{metrics['race_weekends']}")
        st.caption(f"Selected model: {chosen}. Baseline: always predict the driver's race median pace.")

        holdout = pd.DataFrame(metrics["season_holdout"]["models"])
        comparison = pd.concat([
            cv_rows[["name", "mae"]].assign(Evaluation="Grouped cross-validation"),
            holdout[["name", "mae"]].assign(
                Evaluation=f"Train {metrics['season_holdout']['train']} → test {metrics['season_holdout']['test']}"),
        ]).rename(columns={"name": "Model", "mae": "MAE (s)"})
        eval_fig = px.bar(
            comparison, x="Model", y="MAE (s)", color="Evaluation", barmode="group",
            color_discrete_sequence=[theme.RED, theme.YELLOW], text_auto=".3f",
        )
        theme.style_fig(eval_fig, height=320)
        st.plotly_chart(eval_fig, width="stretch", config={"displayModeBar": False})

        st.markdown("**Predicted vs actual pace, held-out predictions**")
        if oof.empty or event_choice == "All Grands Prix":
            st.info("Choose a single Grand Prix in the sidebar to compare predicted and actual pace.")
        else:
            view = only(oof, year_choice, event_choice, driver_choice)
            if view.empty:
                st.info("No predictions for the selected filters.")
            else:
                long = view.melt(
                    id_vars=["Year", "EventName", "Driver", "LapNumber"],
                    value_vars=["ActualPaceDelta", "PredictedPaceDelta"],
                    var_name="Series", value_name="Delta",
                )
                long["Series"] = long["Series"].map({"ActualPaceDelta": "Actual", "PredictedPaceDelta": "Model"})
                long["Season"] = long["Year"].astype(int).astype(str)
                pred_fig = px.line(
                    long.sort_values("LapNumber"), x="LapNumber", y="Delta", color="Driver", line_dash="Series",
                    facet_col="Season" if year_choice == "All seasons" else None, markers=True,
                    color_discrete_map=theme.DRIVER_COLORS,
                    labels={"LapNumber": "Race lap", "Delta": "Seconds vs race median"},
                )
                pred_fig.add_hline(y=0, line_dash="dot", line_color=theme.MUTED)
                theme.style_fig(pred_fig, height=350)
                st.plotly_chart(pred_fig, width="stretch", config={"displayModeBar": False})

        by_compound = pd.DataFrame(metrics.get("by_compound", []))
        if not by_compound.empty:
            st.markdown("**Error by compound**")
            st.dataframe(
                by_compound.rename(columns={
                    "compound": "Compound", "laps": "Laps", "baseline_mae": "Baseline MAE (s)", "model_mae": "Model MAE (s)",
                }).round(3),
                hide_index=True,
            )

        legacy = metrics.get("legacy")
        if legacy:
            theme.callout(
                "<b>Why the first model was replaced:</b> the original Random Forest predicted absolute lap time from "
                f"the circuit name. On 2026 it scored {legacy['random_forest_mae']:.2f} s MAE, worse than simply using each "
                f"circuit's 2025 average ({legacy['naive_circuit_mean_mae']:.2f} s), and "
                f"{legacy['random_forest_mae_unseen_circuits']:.1f} s on the {legacy['unseen_circuit_laps']} laps at circuits it had not seen. "
                "Predicting pace relative to the race median removes the circuit effect and beats its baseline."
            )

# ---------------------------------------------------------------- strategy lab
with tab_lab:
    st.subheader("Strategy simulator")
    theme.callout(
        "<b>Read this first:</b> degradation comes from the data, but the pace gap between compounds cannot be estimated "
        "reliably from these races (the intervals include zero), so the gaps below are <b>assumptions you can change</b>. "
        "Tyre wear is modelled as linear, stint order is ignored, and traffic, safety cars and weather are not modelled. "
        "Use it to see how sensitive a plan is, not to predict a result."
    )
    lab_ready = not adjusted.empty and "TyreAgeP90" in adjusted
    if not lab_ready:
        st.info("Run `python -m src.degradation` to generate the degradation estimates the simulator needs.")
    else:
        seasons = sorted(adjusted["Year"].astype(int).unique().tolist())
        row1 = st.columns(4)
        lab_year = row1[0].selectbox("Degradation from season", seasons, index=len(seasons) - 1, key="lab_year")
        race_laps = row1[1].slider("Race distance (laps)", 30, 80, 57, 1, key="lab_laps")
        pit_loss = row1[2].slider("Pit-stop time loss (s)", 15.0, 30.0, 22.0, 0.5, key="lab_pit")
        max_stops = row1[3].selectbox("Maximum stops", [1, 2], index=1, key="lab_stops")
        row2 = st.columns(2)
        medium_gap = row2[0].slider("Medium pace vs hard (s per lap, assumption)", -1.0, 0.0, -0.35, 0.05, key="lab_medium")
        soft_gap = row2[1].slider("Soft pace vs hard (s per lap, assumption)", -1.5, 0.0, -0.70, 0.05, key="lab_soft")

        season_rows = adjusted[adjusted["Year"].astype(int) == int(lab_year)]
        degradation = {
            r.Compound: (r.DegradationSecondsPerTyreLap, r.CI95Lower, r.CI95Upper) for r in season_rows.itertuples()}
        stint_limit = {r.Compound: int(round(r.TyreAgeP90)) for r in season_rows.itertuples()}
        plans = simulate_strategies(
            race_laps, degradation, {"HARD": 0.0, "MEDIUM": medium_gap, "SOFT": soft_gap},
            pit_loss, stint_limit, max_stops=max_stops)
        if plans.empty:
            st.info("No legal strategy fits this race distance within the observed stint lengths. Try fewer laps or allow 2 stops.")
        else:
            shown = plans.head(8).copy()
            shown["Plan"] = shown["Stops"].map({1: "1 stop", 2: "2 stops"})
            plan_fig = px.bar(
                shown, x="ExpectedRaceTimeLoss_s", y="Strategy", orientation="h", color="Plan",
                color_discrete_map={"1 stop": theme.RED, "2 stops": theme.YELLOW},
                hover_data=["ProbabilityBest"],
                labels={"ExpectedRaceTimeLoss_s": "Expected time lost vs the best plan (s)", "ProbabilityBest": "Chance it is best"},
            )
            plan_fig.update_yaxes(autorange="reversed")
            theme.style_fig(plan_fig, height=380)
            st.plotly_chart(plan_fig, width="stretch", config={"displayModeBar": False})
            st.dataframe(
                shown[["Strategy", "Stops", "ExpectedRaceTimeLoss_s", "ProbabilityBest"]].rename(columns={
                    "ExpectedRaceTimeLoss_s": "Expected loss (s)", "ProbabilityBest": "Chance it is best"}).round(3),
                hide_index=True,
            )
            st.caption(
                "Each plan lists stint lengths in laps (order is not modelled). Uncertainty comes from the 95% degradation "
                f"interval, sampled 1,000 times. Stint lengths are capped at the 90th percentile of observed {lab_year} stints: "
                + ", ".join(f"{c.title()} {n}" for c, n in sorted(stint_limit.items())) + " laps."
            )

    st.subheader("Pace what-if")
    whatif = metrics.get("whatif")
    if not whatif:
        st.info("Run `python -m src.model` to enable the what-if predictor.")
    else:
        wi = st.columns(3)
        wi_compound = wi[0].selectbox("Compound", ["SOFT", "MEDIUM", "HARD"], index=1, key="wi_compound")
        wi_progress = wi[1].slider("Race progress", 0.05, 0.95, 0.5, 0.05, key="wi_progress")
        wi_level = wi[2].selectbox("Prediction interval", ["80%", "90%"], index=1, key="wi_level")
        key = {"80%": "0.8", "90%": "0.9"}[wi_level]
        half_width = whatif["interval_half_width_s"][key]
        ages = np.arange(1, 41)
        centre = predict_linear(whatif["spec"], wi_compound, wi_progress, ages)
        what_fig = go.Figure()
        what_fig.add_trace(go.Scatter(
            x=list(ages) + list(ages[::-1]), y=list(centre + half_width) + list((centre - half_width)[::-1]),
            fill="toself", fillcolor="rgba(232,0,45,0.18)", line={"width": 0}, hoverinfo="skip", name=f"{wi_level} interval"))
        what_fig.add_trace(go.Scatter(
            x=ages, y=centre, mode="lines", name="Predicted pace",
            line={"color": theme.COMPOUND_COLORS[wi_compound], "width": 3}))
        what_fig.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Seconds vs the driver's race median pace")
        theme.style_fig(what_fig, height=400)
        st.plotly_chart(what_fig, width="stretch", config={"displayModeBar": False})
        st.caption(
            f"Held-out check: the {wi_level} band contained {whatif['interval_coverage'][key]:.1%} of laps from race weekends the model "
            "never saw. The band is a constant width, so it is wide for a single lap. Compound shifts in this model are small and "
            "not statistically distinguishable, so read the tyre-age and race-progress effects rather than gaps between compounds."
        )

# ---------------------------------------------------------------- assistant
with tab_ask:
    chat_ui.render(load_pitwall_data, get_client, load_metrics_file("assistant_eval.json"))