from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Pitwall AI | Ferrari Tyre Analysis",
    page_icon="🏎️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; }
    [data-testid="stMetricValue"] { color: #c8102e; }
    .pitwall-note { color: #777; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_laps() -> pd.DataFrame:
    paths = sorted(DATA_DIR.glob("ferrari_20*_race_laps.csv"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths]
    laps = pd.concat(frames, ignore_index=True)
    for column in ("Year", "LapNumber", "LapTimeSeconds", "TyreLife", "Stint"):
        if column in laps:
            laps[column] = pd.to_numeric(laps[column], errors="coerce")
    return laps.dropna(subset=["Year", "LapNumber", "LapTimeSeconds", "TyreLife"])


@st.cache_data
def load_table(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


st.title("🏎️ Pitwall AI")
st.subheader("Ferrari tyre and race-pace explorer")
st.markdown(
    '<p class="pitwall-note">Explore race laps and tyre-age trends for Charles Leclerc and Lewis Hamilton. '
    "Use the uncertainty intervals as a guide; these estimates are for analysis, not live strategy." 
    "</p>",
    unsafe_allow_html=True,
)

laps = load_laps()
if laps.empty:
    st.warning("Nu găsesc CSV-urile de tururi. Rulează `python -m src.data` din folderul proiectului.")
    st.stop()

stint_slopes = load_table("ferrari_tyre_stint_slopes.csv")
race_driver = load_table("ferrari_tyre_degradation_by_race_driver.csv")
driver_summary = load_table("ferrari_tyre_degradation_by_driver.csv")
adjusted = load_table("ferrari_tyre_degradation_estimates.csv")

with st.sidebar:
    st.header("Filtre")
    years = sorted(laps["Year"].dropna().astype(int).unique().tolist())
    year_choice = st.selectbox("Sezon", ["Toate sezoanele", *years], index=1 if years else 0)
    scoped = laps if year_choice == "Toate sezoanele" else laps[laps["Year"].astype(int) == int(year_choice)]

    events = sorted(scoped["EventName"].dropna().astype(str).unique().tolist())
    event_options = ["Toate cursele", *events]
    event_choice = st.selectbox("Cursă", event_options, index=1 if events else 0)

    drivers = sorted(scoped["Driver"].dropna().astype(str).unique().tolist())
    driver_choice = st.selectbox("Pilot", ["Ambii", *drivers])

    compounds = sorted(scoped["Compound"].dropna().astype(str).unique().tolist())
    compound_choice = st.selectbox("Compus", ["Toți compușii", *compounds])

filtered = scoped.copy()
if event_choice != "Toate cursele":
    filtered = filtered[filtered["EventName"].astype(str) == event_choice]
if driver_choice != "Ambii":
    filtered = filtered[filtered["Driver"].astype(str) == driver_choice]
if compound_choice != "Toți compușii":
    filtered = filtered[filtered["Compound"].astype(str) == compound_choice]

filtered["StintKey"] = (
    filtered["Year"].astype(int).astype(str) + " · "
    + filtered["EventName"].astype(str) + " · "
    + filtered["Driver"].astype(str) + " · stint "
    + filtered["Stint"].fillna(-1).astype(int).astype(str)
)

stints_count = filtered["StintKey"].nunique()
weekends_count = filtered[["Year", "EventName"]].drop_duplicates().shape[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tururi afișate", f"{len(filtered):,}")
col2.metric("Stinturi", f"{stints_count:,}")
col3.metric("Weekenduri", f"{weekends_count:,}")
col4.metric("Timp mediu pe tur", f"{filtered['LapTimeSeconds'].mean():.2f} s" if not filtered.empty else "—")

st.divider()
st.subheader("Ritmul pe tur în funcție de vechimea pneului")
st.caption("Fiecare linie reprezintă un stint; alege o singură cursă pentru o comparație mai ușor de citit.")

if filtered.empty:
    st.info("Nu există tururi pentru filtrele alese.")
else:
    hover_fields = [column for column in [
        "EventName", "Year", "Driver", "Stint", "LapNumber", "Compound",
        "Weather_TrackTemp", "Weather_Rainfall",
    ] if column in filtered.columns]
    pace_fig = px.line(
        filtered.sort_values(["StintKey", "TyreLife"]),
        x="TyreLife",
        y="LapTimeSeconds",
        color="Compound",
        line_dash="Driver",
        line_group="StintKey",
        markers=True,
        hover_data=hover_fields,
        labels={
            "TyreLife": "Vechimea pneului (tururi)",
            "LapTimeSeconds": "Timp pe tur (secunde)",
            "Compound": "Compus",
            "Driver": "Pilot",
            "Year": "Sezon",
            "LapNumber": "Turul cursei",
            "Weather_TrackTemp": "Temperatura pistei (°C)",
            "Weather_Rainfall": "Ploaie",
        },
        template="plotly_white",
        height=500,
    )
    pace_fig.update_traces(marker={"size": 6}, line={"width": 1.5})
    pace_fig.update_layout(legend_title_text="Compus / pilot", hovermode="closest")
    st.plotly_chart(pace_fig, use_container_width=True)

st.divider()
st.subheader("Estimarea degradării")
st.caption("Panta pozitivă indică, în model, o creștere a timpului pe tur pentru fiecare tur suplimentar de uzură.")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("**Estimarea ajustată pe sezon și compus**")
    chart_data = adjusted.copy()
    if not chart_data.empty:
        if year_choice != "Toate sezoanele":
            chart_data = chart_data[chart_data["Year"].astype(int) == int(year_choice)]
        chart_data["ErrorUpper"] = (chart_data["CI95Upper"] - chart_data["DegradationSecondsPerTyreLap"]).clip(lower=0)
        chart_data["ErrorLower"] = (chart_data["DegradationSecondsPerTyreLap"] - chart_data["CI95Lower"]).clip(lower=0)
        model_fig = px.scatter(
            chart_data,
            x="Compound",
            y="DegradationSecondsPerTyreLap",
            color="Year",
            error_y="ErrorUpper",
            error_y_minus="ErrorLower",
            hover_data=["CleanLaps", "Stints", "CI95Lower", "CI95Upper"],
            labels={
                "Compound": "Compus",
                "DegradationSecondsPerTyreLap": "Secunde pe tur de uzură",
                "Year": "Sezon",
                "CleanLaps": "Tururi curate",
                "Stints": "Stinturi",
            },
            template="plotly_white",
            height=400,
        )
        model_fig.add_hline(y=0, line_dash="dot", line_color="#777")
        model_fig.update_traces(marker_size=11)
        st.plotly_chart(model_fig, use_container_width=True)
        st.caption("Bare: intervale de încredere ale regresiei ajustate; nu sunt bootstrap pe stinturi.")
    else:
        st.info("Nu găsesc estimările ajustate. Rulează `python -m src.degradation`.")

with chart_right:
    st.markdown("**Pante observate pe pilot, cu bootstrap pe stinturi**")
    boot_data = driver_summary.copy()
    if not boot_data.empty:
        if year_choice != "Toate sezoanele":
            boot_data = boot_data[boot_data["Year"].astype(int) == int(year_choice)]
        boot_data["ErrorUpper"] = (
            boot_data["ClusterBootstrap95Upper"] - boot_data["MedianObservedSlopeSecondsPerTyreLap"]
        ).clip(lower=0).fillna(0)
        boot_data["ErrorLower"] = (
            boot_data["MedianObservedSlopeSecondsPerTyreLap"] - boot_data["ClusterBootstrap95Lower"]
        ).clip(lower=0).fillna(0)
        driver_fig = px.scatter(
            boot_data,
            x="Compound",
            y="MedianObservedSlopeSecondsPerTyreLap",
            color="Driver",
            facet_col="Year" if year_choice == "Toate sezoanele" else None,
            error_y="ErrorUpper",
            error_y_minus="ErrorLower",
            hover_data=["Stints", "ClusterBootstrap95Lower", "ClusterBootstrap95Upper"],
            labels={
                "Compound": "Compus",
                "MedianObservedSlopeSecondsPerTyreLap": "Secunde pe tur de uzură",
                "Driver": "Pilot",
                "Year": "Sezon",
                "Stints": "Stinturi independente",
            },
            template="plotly_white",
            height=400,
        )
        driver_fig.add_hline(y=0, line_dash="dot", line_color="#777")
        driver_fig.update_traces(marker_size=10)
        st.plotly_chart(driver_fig, use_container_width=True)
        st.caption("Puncte: medianele pantelor observate pe stint; bare: percentile bootstrap pe stinturi întregi.")
    else:
        st.info("Nu găsesc sumarul bootstrap. Rulează `python -m src.degradation`.")

st.divider()
st.subheader("Distribuția pantelor pe stint")
if not stint_slopes.empty:
    stint_view = stint_slopes.copy()
    if year_choice != "Toate sezoanele":
        stint_view = stint_view[stint_view["Year"].astype(int) == int(year_choice)]
    if event_choice != "Toate cursele":
        stint_view = stint_view[stint_view["EventName"].astype(str) == event_choice]
    if driver_choice != "Ambii":
        stint_view = stint_view[stint_view["Driver"].astype(str) == driver_choice]
    if compound_choice != "Toți compușii":
        stint_view = stint_view[stint_view["Compound"].astype(str) == compound_choice]

    if stint_view.empty:
        st.info("Nu există stinturi eligibile pentru filtrele alese.")
    else:
        slope_fig = px.strip(
            stint_view,
            x="Compound",
            y="ObservedSlopeSecondsPerTyreLap",
            color="Driver",
            hover_data=["Year", "EventName", "Stint", "Laps", "TyreAgeMin", "TyreAgeMax"],
            labels={
                "Compound": "Compus",
                "ObservedSlopeSecondsPerTyreLap": "Panta observată (secunde / tur de uzură)",
                "Driver": "Pilot",
                "EventName": "Cursă",
                "Stint": "Stint",
                "Laps": "Tururi în stint",
                "TyreAgeMin": "Vârsta minimă a pneului",
                "TyreAgeMax": "Vârsta maximă a pneului",
            },
            template="plotly_white",
            height=420,
        )
        slope_fig.add_hline(y=0, line_dash="dot", line_color="#777")
        st.plotly_chart(slope_fig, use_container_width=True)
        st.caption("Pantele pe stint sunt trenduri observate; pot include efecte de combustibil, trafic și evoluția pistei.")
else:
    st.info("Nu găsesc fișierul ferrari_tyre_stint_slopes.csv. Rulează `python -m src.degradation`.")

st.divider()
st.subheader("Rezultate bootstrap pe cursă și pilot")
if not race_driver.empty:
    race_table = race_driver.copy()
    if year_choice != "Toate sezoanele":
        race_table = race_table[race_table["Year"].astype(int) == int(year_choice)]
    if event_choice != "Toate cursele":
        race_table = race_table[race_table["EventName"].astype(str) == event_choice]
    if driver_choice != "Ambii":
        race_table = race_table[race_table["Driver"].astype(str) == driver_choice]
    if compound_choice != "Toți compușii":
        race_table = race_table[race_table["Compound"].astype(str) == compound_choice]

    display_columns = [
        "Year", "EventName", "Driver", "Compound", "Stints",
        "MedianObservedSlopeSecondsPerTyreLap",
        "ClusterBootstrap95Lower", "ClusterBootstrap95Upper",
    ]
    if race_table.empty:
        st.info("Nu există sumar bootstrap pentru filtrele alese.")
    else:
        st.dataframe(
            race_table[display_columns].rename(columns={
                "Year": "Sezon",
                "EventName": "Cursă",
                "Driver": "Pilot",
                "Compound": "Compus",
                "Stints": "Stinturi",
                "MedianObservedSlopeSecondsPerTyreLap": "Mediană (s/tur de uzură)",
                "ClusterBootstrap95Lower": "Bootstrap 95% inferior",
                "ClusterBootstrap95Upper": "Bootstrap 95% superior",
            }).round(3),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Intervalul este gol când combinația are un singur stint eligibil.")
else:
    st.info("Nu găsesc sumarul pe cursă și pilot. Rulează `python -m src.degradation`.")

st.markdown(
    "<p class='pitwall-note'>Datele și estimările sunt exploratorii. Un interval fără bare înseamnă că există prea puține stinturi independente pentru bootstrap.</p>",
    unsafe_allow_html=True,
)
