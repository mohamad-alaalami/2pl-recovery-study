"""Recovery metrics. Pure pandas/numpy; no fitting, no thresholds.

Two levels, so replicate variation never disappears into one mean:

1. `replicate_metrics(items)`: one row per unit (condition x replicate), computed over its J items:
     rmse_b, rmse_a, rmse_log_a          sqrt(mean((est - true)^2)); log a uses log(est_a) - log(true_a)
     bias_b, bias_a, bias_log_a          mean(est - true)
     coverage_b, coverage_a              share of items whose posterior interval contains the truth
     width_b, width_a, rel_width_a       median interval width (a also relative to true a)
     within_<d>_b, within_<d>_log_a      share of items with |error| <= d, for each configured d
     corr_b, corr_log_a                  Pearson correlation of estimates with truth
     spearman_b, spearman_log_a          rank correlation
     slope_b, slope_log_a                slope of estimate on truth (1 = no shrinkage)

2. `summarize(rep_metrics, subset)`: per condition, for every metric above: mean, sd across
   replicates, mc_se = sd / sqrt(R), median, q05, q10, q90, q95, plus pooled RMSE
   (sqrt of the mean squared error over all items of all used replicates) and fit-tier counts.

Fit-tier subsets (Policy D): primary = every completed non-gross fit; no_divergent = clean +
mixing_flag; strict = clean only. Results are never filtered silently: tier counts are reported.

`upper_envelope(summary, stat, populations, worlds)`: at each N, the maximum across the selected
conditions of a per-condition statistic (e.g. rmse_b_mean), and which condition attained it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITION_KEYS = ["condition_id", "population", "difficulty_shift", "theta_kind", "theta_shape",
                  "world", "tau_a", "tau_b", "rho", "test_length", "N"]
SUBSETS = {"primary": ("clean", "mixing_flag", "divergent"),
           "no_divergent": ("clean", "mixing_flag"),
           "strict": ("clean",)}
STATS = ("mean", "sd", "mc_se", "median", "q05", "q10", "q90", "q95")
PROTOCOL_KEYS = ("initial_tier", "fallback_triggered", "fallback_tier", "canonical_parameterization")
NON_METRIC = set(CONDITION_KEYS) | set(PROTOCOL_KEYS) | {"unit_id", "block", "replicate", "tier",
                                                          "mse_b", "mse_a", "mse_log_a"}
PROTOCOL_DEFAULTS = {"initial_tier": None, "fallback_triggered": False, "fallback_tier": "",
                     "canonical_parameterization": "centered_b"}   # results from before the fallback


def _tol_label(d: float) -> str:
    return f"{d:g}".replace(".", "p")


def _slope(x, y) -> float:
    return float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else float("nan")


def _corr(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")


def _spearman(x, y) -> float:
    return _corr(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def replicate_metrics(items: pd.DataFrame, tolerances=(0.2,)) -> pd.DataFrame:
    rows = []
    for unit_id, g in items.groupby("unit_id", sort=True):
        first = g.iloc[0]
        row = {k: first[k] for k in CONDITION_KEYS}
        row.update(unit_id=unit_id, block=first["block"], replicate=int(first["replicate"]),
                   tier=first["tier"])
        for k, default in PROTOCOL_DEFAULTS.items():            # fit-protocol outcome of the unit
            row[k] = first[k] if k in g.columns else (first["tier"] if k == "initial_tier" else default)
        row["fallback_triggered"] = bool(row["fallback_triggered"])
        if first["tier"] == "gross" or not np.all(np.isfinite(g["est_a"])) \
                or not np.all(np.isfinite(g["est_b"])):
            rows.append(row)                                  # kept and counted, no metrics
            rows[-1]["tier"] = "gross"
            continue
        tb, eb = g["true_b"].to_numpy(), g["est_b"].to_numpy()
        ta, ea = g["true_a"].to_numpy(), g["est_a"].to_numpy()
        tla, ela = np.log(ta), np.log(ea)
        err = {"b": eb - tb, "a": ea - ta, "log_a": ela - tla}
        for p, e in err.items():
            row[f"mse_{p}"] = float(np.mean(e ** 2))
            row[f"rmse_{p}"] = float(np.sqrt(row[f"mse_{p}"]))
            row[f"bias_{p}"] = float(np.mean(e))
        for p in ("a", "b"):
            lo, hi, t = g[f"lo_{p}"].to_numpy(), g[f"hi_{p}"].to_numpy(), g[f"true_{p}"].to_numpy()
            row[f"coverage_{p}"] = float(np.mean((lo <= t) & (t <= hi)))
            row[f"width_{p}"] = float(np.median(hi - lo))
        row["rel_width_a"] = float(np.median((g["hi_a"] - g["lo_a"]).to_numpy() / ta))
        for d in tolerances:
            row[f"within_{_tol_label(d)}_b"] = float(np.mean(np.abs(err["b"]) <= d))
            row[f"within_{_tol_label(d)}_log_a"] = float(np.mean(np.abs(err["log_a"]) <= d))
        row["corr_b"], row["corr_log_a"] = _corr(tb, eb), _corr(tla, ela)
        row["spearman_b"], row["spearman_log_a"] = _spearman(tb, eb), _spearman(tla, ela)
        row["slope_b"], row["slope_log_a"] = _slope(tb, eb), _slope(tla, ela)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(rep: pd.DataFrame) -> list:
    return [c for c in rep.columns if c not in NON_METRIC]


def summarize(rep: pd.DataFrame, subset: str = "primary") -> pd.DataFrame:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {sorted(SUBSETS)}")
    metrics = metric_columns(rep)
    out = []
    for key, g in rep.groupby(CONDITION_KEYS, sort=True, dropna=False):
        row = dict(zip(CONDITION_KEYS, key))
        counts = g["tier"].value_counts()
        row["units"] = len(g)
        for t in ("clean", "mixing_flag", "divergent", "gross"):
            row[f"n_{t}"] = int(counts.get(t, 0))
        row.update(fallback_counts(g))
        used = g[g["tier"].isin(SUBSETS[subset])]
        row["subset"], row["n_used"] = subset, len(used)
        for m in metrics:
            v = used[m].dropna().to_numpy(dtype=float)
            n = len(v)
            row[f"{m}_mean"] = float(v.mean()) if n else np.nan
            row[f"{m}_sd"] = float(v.std(ddof=1)) if n > 1 else np.nan
            row[f"{m}_mc_se"] = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            row[f"{m}_median"] = float(np.median(v)) if n else np.nan
            for q in (5, 10, 90, 95):
                row[f"{m}_q{q:02d}"] = float(np.percentile(v, q)) if n else np.nan
        for p in ("b", "a", "log_a"):
            mse = used[f"mse_{p}"].dropna() if f"mse_{p}" in used else pd.Series(dtype=float)
            row[f"rmse_{p}_pooled"] = float(np.sqrt(mse.mean())) if len(mse) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def fallback_counts(g: pd.DataFrame) -> dict:
    """Fit-protocol bookkeeping for a set of units (canonical tiers are counted separately):
    triggered = centered divergent/gross -> non-centered refit; rescued = non-centered canonical;
    unresolved = triggered and the canonical fit is still divergent or gross."""
    trig = g["fallback_triggered"].astype(bool)
    rescued = trig & (g["canonical_parameterization"] == "noncentered_b")
    unresolved = trig & g["tier"].isin(["divergent", "gross"])
    n = len(g)
    return {"n_fallback_triggered": int(trig.sum()), "n_fallback_rescued": int(rescued.sum()),
            "n_fallback_unresolved": int(unresolved.sum()),
            "fallback_trigger_rate": float(trig.mean()) if n else np.nan,
            "n_canonical_centered": int((g["canonical_parameterization"] == "centered_b").sum()),
            "n_canonical_noncentered": int((g["canonical_parameterization"] == "noncentered_b").sum())}


def fallback_report(rep: pd.DataFrame) -> pd.DataFrame:
    """Per condition: fallback trigger / rescue / unresolved counts and canonical parameterization
    counts, plus initial (centered) and canonical tier counts."""
    rows = []
    for key, g in rep.groupby(CONDITION_KEYS, sort=True, dropna=False):
        row = dict(zip(CONDITION_KEYS, key))
        row["units"] = len(g)
        row.update(fallback_counts(g))
        for t in ("clean", "mixing_flag", "divergent", "gross"):
            row[f"initial_{t}"] = int((g["initial_tier"] == t).sum())
            row[f"canonical_{t}"] = int((g["tier"] == t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def upper_envelope(summary: pd.DataFrame, stat: str, populations=None, worlds=None) -> pd.DataFrame:
    """Per N: the maximum of `stat` over the selected conditions, with the condition attaining it."""
    s = summary
    if populations is not None:
        s = s[s["population"].isin(populations)]
    if worlds is not None:
        s = s[s["world"].isin(worlds)]
    s = s.dropna(subset=[stat])
    rows = []
    for N, g in s.groupby("N", sort=True):
        k = g[stat].idxmax()
        rows.append({"N": int(N), stat: float(g.loc[k, stat]), "n_conditions": len(g),
                     "attained_by": g.loc[k, "condition_id"]})
    return pd.DataFrame(rows)


def replications_for_mc_se(summary: pd.DataFrame, metric: str, target_se: float) -> pd.DataFrame:
    """Replications per condition needed for MC SE <= target, from SE = SD / sqrt(R)."""
    need = np.ceil((summary[f"{metric}_sd"] / target_se) ** 2)
    return summary[["condition_id"]].assign(**{f"{metric}_sd": summary[f"{metric}_sd"],
                                               "replications_needed": need})
