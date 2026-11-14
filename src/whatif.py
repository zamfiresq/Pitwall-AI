"""What-if pace predictions and a tyre-strategy simulator.

* ``export_linear_model`` / ``predict_linear`` turn the fitted linear pace model into plain
  numbers, so the dashboard predicts with numpy only (no pickle, no scikit-learn needed).
* ``simulate_strategies`` compares 1- and 2-stop plans using the estimated degradation
  and its uncertainty. Fuel burn depends only on race lap, so it is identical for every
  strategy and cancels out of the comparison.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def export_linear_model(pipeline, numeric: list[str], categorical: list[str]) -> dict:
    """Pull encoder, scaler and coefficients out of a fitted one-hot + scaler + Ridge pipeline."""
    prep = pipeline.named_steps["features"]
    encoder = prep.named_transformers_["categorical"]
    scaler = prep.named_transformers_["numeric"]
    regressor = pipeline.named_steps["regressor"]
    return {
        "categorical": categorical,
        "categories": [[str(value) for value in values] for values in encoder.categories_],
        "numeric": numeric,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": np.ravel(regressor.coef_).tolist(),
        "intercept": float(regressor.intercept_),
    }


def predict_linear(spec: dict, compound: str, progress, tyre_age) -> np.ndarray:
    """Predicted seconds relative to the driver's race median pace."""
    if spec["numeric"] != ["Progress", "TyreLife"] or spec["categorical"] != ["Compound"]:
        raise ValueError("predict_linear expects the Compound + Progress + TyreLife model")
    categories = spec["categories"][0]
    coef = np.asarray(spec["coef"])
    onehot = np.array([1.0 if category == compound else 0.0 for category in categories])
    progress, tyre_age = np.broadcast_arrays(np.asarray(progress, float), np.asarray(tyre_age, float))
    scaled_progress = (progress - spec["mean"][0]) / spec["scale"][0]
    scaled_age = (tyre_age - spec["mean"][1]) / spec["scale"][1]
    n_cat = len(categories)
    return (
        spec["intercept"] + onehot @ coef[:n_cat]
        + coef[n_cat] * scaled_progress + coef[n_cat + 1] * scaled_age
    )


def enumerate_strategies(
    race_laps: int,
    compounds: list[str],
    max_stint: dict[str, int],
    min_stint: int = 8,
    max_stops: int = 2,
    step: int = 2,
) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    """All legal plans: at least two different compounds, stint lengths within limits."""
    plans = []
    for first, second in itertools.product(compounds, repeat=2):
        if first == second:
            continue
        for stop in range(min_stint, race_laps - min_stint + 1):
            lengths = (stop, race_laps - stop)
            if lengths[0] <= max_stint[first] and lengths[1] <= max_stint[second]:
                plans.append(((first, second), lengths))
    if max_stops >= 2:
        for sequence in itertools.product(compounds, repeat=3):
            if len(set(sequence)) < 2:
                continue
            for stop1 in range(min_stint, race_laps - 2 * min_stint + 1, step):
                if stop1 > max_stint[sequence[0]]:
                    break
                for stop2 in range(stop1 + min_stint, race_laps - min_stint + 1, step):
                    lengths = (stop1, stop2 - stop1, race_laps - stop2)
                    if lengths[1] <= max_stint[sequence[1]] and lengths[2] <= max_stint[sequence[2]]:
                        plans.append((sequence, lengths))
    return plans


def simulate_strategies(
    race_laps: int,
    degradation: dict[str, tuple[float, float, float]],
    offsets: dict[str, float],
    pit_loss: float,
    max_stint: dict[str, int],
    max_stops: int = 2,
    draws: int = 1000,
    seed: int = 0,
    top: int = 10,
) -> pd.DataFrame:
    """Rank strategies by expected race time, with uncertainty from the degradation interval.

    ``degradation[compound] = (estimate, ci_low, ci_high)`` in seconds per tyre lap;
    ``offsets[compound]`` is the pace advantage (negative = faster) versus the reference tyre.
    Degradation is drawn from a normal built from the 95% interval, floored at zero; the same
    draws are used for every strategy so the comparison is paired. Stint order is not modelled.
    """
    compounds = sorted(degradation)
    # Without fuel or track-position effects the order of stints does not change total time, so plans with
    # the same (compound, length) pairs are one strategy; keeping both would split its probability of being best.
    unique: dict[tuple, tuple] = {}
    for sequence, lengths in enumerate_strategies(race_laps, compounds, max_stint, max_stops=max_stops):
        pairs = tuple(sorted(zip(sequence, lengths), key=lambda pair: (-pair[1], pair[0])))
        unique.setdefault(pairs, (tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)))
    plans = list(unique.values())
    if not plans:
        return pd.DataFrame()

    column = {compound: i for i, compound in enumerate(compounds)}
    wear = np.zeros((len(plans), len(compounds)))   # sum of tyre ages: l(l+1)/2 per stint
    base = np.zeros(len(plans))                     # compound pace offsets + pit stops
    for p, (sequence, lengths) in enumerate(plans):
        for compound, laps in zip(sequence, lengths):
            wear[p, column[compound]] += laps * (laps + 1) / 2
            base[p] += laps * offsets.get(compound, 0.0)
        base[p] += pit_loss * (len(sequence) - 1)

    mean = np.array([degradation[c][0] for c in compounds])
    sd = np.array([(degradation[c][2] - degradation[c][1]) / (2 * 1.96) for c in compounds])
    candidates = np.argsort(base + wear @ mean)[:200]
    rng = np.random.default_rng(seed)
    slopes = np.maximum(rng.normal(mean, sd, size=(draws, len(compounds))), 0.0)
    times = base[candidates][None, :] + slopes @ wear[candidates].T
    regret = times - times.min(axis=1, keepdims=True)
    prob_best = np.bincount(times.argmin(axis=1), minlength=len(candidates)) / draws

    rows = []
    for k, p in enumerate(candidates):
        sequence, lengths = plans[p]
        rows.append({
            "Strategy": " + ".join(f"{c.title()} {n}" for c, n in zip(sequence, lengths)),
            "Stops": len(sequence) - 1,
            "ExpectedRaceTimeLoss_s": float(regret[:, k].mean()),
            "ProbabilityBest": float(prob_best[k]),
            "Stints": list(lengths),
            "Compounds": list(sequence),
        })
    return pd.DataFrame(rows).sort_values("ExpectedRaceTimeLoss_s").head(top).reset_index(drop=True)
