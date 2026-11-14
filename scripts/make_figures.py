"""Build the README figures from the computed results: python scripts/make_figures.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "docs"
RED, YELLOW, BLACK, PANEL, GRID, TEXT, MUTED = "#E8002D", "#FFF200", "#15151E", "#1F1F2A", "#34343F", "#F2F2F5", "#9B9BA8"

installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Titillium Web" if "Titillium Web" in installed else "DejaVu Sans",
    "figure.facecolor": BLACK, "axes.facecolor": BLACK, "savefig.facecolor": BLACK,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def degradation() -> None:
    est = pd.read_csv(DATA / "ferrari_tyre_degradation_estimates.csv")
    compounds = ["SOFT", "MEDIUM", "HARD"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    width = 0.36
    for offset, (year, color) in zip((-width / 2, width / 2), ((2025, MUTED), (2026, RED))):
        rows = est[est["Year"] == year].set_index("Compound").loc[compounds]
        value = rows["DegradationSecondsPerTyreLap"].to_numpy()
        err = np.vstack([value - rows["CI95Lower"], rows["CI95Upper"] - value])
        ax.bar(np.arange(3) + offset, value, width, yerr=err, color=color, label=str(year),
               capsize=4, error_kw={"ecolor": TEXT, "lw": 1.3})
    ax.set_xticks(range(3), [c.title() for c in compounds])
    ax.set_ylabel("Seconds lost per tyre lap")
    ax.set_title("Tyre degradation by compound (dry laps, 95% stint-bootstrap interval)", loc="left", fontsize=12, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=160)


def model() -> None:
    m = json.loads((DATA / "model_metrics.json").read_text())
    cv = {r["name"]: r["mae"] for r in m["cross_validation"]}
    hold = {r["name"]: r["mae"] for r in m["season_holdout"]["models"]}
    names = list(cv)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(names))
    ax.bar(x - 0.18, [cv[n] for n in names], 0.36, color=RED, label="Grouped cross-validation")
    ax.bar(x + 0.18, [hold[n] for n in names], 0.36, color=YELLOW,
           label=f"Train {m['season_holdout']['train']}, test {m['season_holdout']['test']}")
    ax.set_xticks(x, [n.replace(": ", ":\n") for n in names], fontsize=8)
    ax.set_ylabel("Mean absolute error (s)")
    ax.set_title("Race-pace model error versus baseline (lower is better)", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    degradation()
    model()
    print(f"Saved figures to {OUT}")
