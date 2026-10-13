"""Final analysis of a completed study: tables, QA checks and figures. Reads results only; never
fits, never writes into the results tree it reads.

    python -m recovery2pl analyze --config configs/final_v1.yaml --out <merged root> --analysis-out <dir>

Scientific metrics use the CANONICAL fit of every unit only (the standard estimate columns written by
the runner); attempt-level columns are used only for fallback/convergence bookkeeping and QA.

Roles of conditions:
    main            populations S0, S1 x worlds small, typical   (enter the main planning envelope)
    stress_S2       population S2                                 (never in the envelope)
    stress_extreme  world extreme                                 (never in the envelope)

Main conservative envelope, per subset and metric:
    E_m(N) = max over main conditions c of mean_c(m, N)
-- the largest mean recovery error observed across the tested main scenarios at N. It is not an
expected value, not a confidence bound and not a guarantee; it is neither smoothed nor forced to be
monotonic, and the condition attaining it is recorded for every point.

Uncertainty conventions: replicate variation = SD and 5th-95th percentiles of replicate-level
metrics; uncertainty of a condition mean = Monte Carlo SE (SD / sqrt(R)). Never min/max of replicates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recovery2pl.metrics import SUBSETS, replicate_metrics, summarize

MAIN_POPULATIONS = ("S0", "S1")
MAIN_WORLDS = ("small", "typical")
SUBSET_ORDER = ("primary", "no_divergent", "strict")
METRICS = ("rmse_b", "rmse_log_a", "rmse_a")
TRUTH_TOL = 1e-12    # simulated truths agree to ~1e-15 across CPU models (see README, Reproducibility)
ITEM_COLS = ("est_a", "lo_a", "hi_a", "sd_a", "rhat_a", "est_b", "lo_b", "hi_b", "sd_b", "rhat_b",
             "est_log_a")


def role(population: str, world: str) -> str:
    if world == "extreme":
        return "stress_extreme"
    if population in MAIN_POPULATIONS and world in MAIN_WORLDS:
        return "main"
    return "stress_" + population


def label(population, world, J, rho) -> str:
    return f"{population} {world} J{int(J)} rho{float(rho):g}"


def load_canonical_items(study_dir) -> pd.DataFrame:
    files = sorted((Path(study_dir) / "units").rglob("rep_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no unit results under {study_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_unit_jsons(study_dir) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(study_dir) / "units").rglob("rep_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        p = d.get("fit_protocol") or {}
        rows.append({"unit_id": d["unit_id"], "tier": d["tier"], "initial_tier": p.get("initial_tier"),
                     "fallback_triggered": p.get("fallback_triggered"), "fallback_tier": p.get("fallback_tier"),
                     "canonical_parameterization": p.get("canonical_parameterization"),
                     "seconds_total": d.get("seconds_total"), "backend": d.get("backend")})
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["role"] = [role(p, w) for p, w in zip(df["population"], df["world"])]
    df["label"] = [label(p, w, j, r) for p, w, j, r in zip(df["population"], df["world"], df["test_length"], df["rho"])]
    return df


# ------------------------------------------------------------------ tables

def condition_tables(rep: pd.DataFrame) -> dict:
    """{subset: per condition x N summary (all metrics; mean/SD/MC SE/median/quantiles; tier and
    fallback counts), annotated with role and label}."""
    return {s: annotate(summarize(rep, s)) for s in SUBSET_ORDER}


def envelope(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per N: max over MAIN conditions of `<metric>_mean`, with the attaining condition and its MC SE."""
    m = summary[summary["role"] == "main"].dropna(subset=[f"{metric}_mean"])
    rows = []
    for N, g in m.groupby("N", sort=True):
        k = g[f"{metric}_mean"].idxmax()
        rows.append({"N": int(N), "metric": metric, "value": float(g.loc[k, f"{metric}_mean"]),
                     "mc_se": float(g.loc[k, f"{metric}_mc_se"]), "driving_condition": g.loc[k, "label"],
                     "driving_condition_id": g.loc[k, "condition_id"], "n_used": int(g.loc[k, "n_used"]),
                     "n_main_conditions": int(len(g))})
    return pd.DataFrame(rows)


def envelope_sensitivity(tables: dict) -> pd.DataFrame:
    out = []
    for metric in METRICS:
        base = envelope(tables["primary"], metric).set_index("N")
        for s in SUBSET_ORDER:
            e = envelope(tables[s], metric).set_index("N")
            for N in base.index:
                v, b = e.loc[N, "value"], base.loc[N, "value"]
                out.append({"metric": metric, "N": int(N), "subset": s, "value": v, "mc_se": e.loc[N, "mc_se"],
                            "driving_condition": e.loc[N, "driving_condition"], "n_used": e.loc[N, "n_used"],
                            "abs_diff_vs_primary": v - b, "rel_diff_vs_primary": (v - b) / b,
                            "same_driver_as_primary": e.loc[N, "driving_condition"] == base.loc[N, "driving_condition"]})
    return pd.DataFrame(out)


def lookup_table(sens: pd.DataFrame) -> pd.DataFrame:
    """Wide main lookup: per N, primary envelope for b and log a with drivers, plus the sensitivity values."""
    rows = []
    for N, g in sens.groupby("N", sort=True):
        row = {"N": int(N)}
        for metric, short in (("rmse_b", "b"), ("rmse_log_a", "log_a")):
            for s in SUBSET_ORDER:
                r = g[(g.metric == metric) & (g.subset == s)].iloc[0]
                row[f"{short}_{s}"] = r["value"]
                if s == "primary":
                    row[f"{short}_primary_mc_se"] = r["mc_se"]
                    row[f"{short}_driving_condition"] = r["driving_condition"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_s2_vs_s1(rep: pd.DataFrame) -> pd.DataFrame:
    """S2 (typical, rho 0.3) minus S1 (typical, rho 0.3) paired by J, N and replicate: the two share
    item banks and differ only in the ability distribution. Primary subset (all non-gross)."""
    keys = ["test_length", "N", "replicate"]
    cols = ["rmse_b", "rmse_log_a", "rmse_a", "coverage_b", "coverage_a", "bias_b", "bias_log_a"]
    s2 = rep[(rep.population == "S2") & (rep.tier != "gross")]
    s1 = rep[(rep.population == "S1") & (rep.world == "typical") & (rep.rho == 0.3) & (rep.tier != "gross")]
    m = s2.merge(s1, on=keys, suffixes=("_s2", "_s1"))
    rows = []
    for (J, N), g in m.groupby(["test_length", "N"], sort=True):
        row = {"test_length": int(J), "N": int(N), "pairs": len(g)}
        for c in cols:
            d = g[f"{c}_s2"] - g[f"{c}_s1"]
            row[f"{c}_s1_mean"], row[f"{c}_s2_mean"] = g[f"{c}_s1"].mean(), g[f"{c}_s2"].mean()
            row[f"{c}_diff_mean"], row[f"{c}_diff_mc_se"] = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        for t in ("divergent", "clean"):
            row[f"{t}_rate_s1"] = float((g["tier_s1"] == t).mean())
            row[f"{t}_rate_s2"] = float((g["tier_s2"] == t).mean())
        row["fallback_rate_s1"] = float(g["fallback_triggered_s1"].mean())
        row["fallback_rate_s2"] = float(g["fallback_triggered_s2"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_table(rep: pd.DataFrame) -> pd.DataFrame:
    """Per condition x N: canonical tier rates and fallback trigger/rescue/unresolved rates."""
    rows = []
    for key, g in annotate(rep).groupby(["role", "label", "condition_id", "population", "world", "test_length",
                                        "rho", "N"], sort=True):
        n = len(g)
        trig = g["fallback_triggered"].astype(bool)
        rows.append({**dict(zip(["role", "label", "condition_id", "population", "world", "test_length", "rho", "N"], key)),
                     "units": n, **{f"rate_{t}": float((g["tier"] == t).mean()) for t in ("clean", "mixing_flag", "divergent", "gross")},
                     "rate_initial_divergent_or_gross": float(g["initial_tier"].isin(["divergent", "gross"]).mean()),
                     "fallback_trigger_rate": float(trig.mean()),
                     "fallback_rescue_rate_of_triggered": float((trig & (g["canonical_parameterization"] == "noncentered_b")).sum() / trig.sum()) if trig.sum() else np.nan,
                     "n_triggered": int(trig.sum()),
                     "n_rescued": int((trig & (g["canonical_parameterization"] == "noncentered_b")).sum()),
                     "n_unresolved": int((trig & g["tier"].isin(["divergent", "gross"])).sum()),
                     "n_canonical_noncentered": int((g["canonical_parameterization"] == "noncentered_b").sum())})
    return pd.DataFrame(rows)


def error_decomposition(items: pd.DataFrame) -> pd.DataFrame:
    """Per condition x N (primary subset): share of the mean replicate MSE that is a common shift of
    all items in a replicate, (mean error)^2 / MSE, for b and log a. A common shift is what the
    N(0, 1) ability identification leaves when a finite sample's ability mean/SD differ from 0/1; it
    does not affect comparisons between items of the same calibration."""
    g = items[items["tier"] != "gross"].assign(e_b=items["est_b"] - items["true_b"],
                                               e_la=np.log(items["est_a"]) - np.log(items["true_a"]))
    keys = ["population", "world", "test_length", "rho", "N"]
    per = g.groupby(keys + ["unit_id"]).agg(mse_b=("e_b", lambda e: float(np.mean(e ** 2))), shift_b=("e_b", "mean"),
                                            mse_la=("e_la", lambda e: float(np.mean(e ** 2))), shift_la=("e_la", "mean"))
    per["shift_b"], per["shift_la"] = per["shift_b"] ** 2, per["shift_la"] ** 2
    m = per.groupby(keys).mean()
    out = pd.DataFrame({"mse_b": m.mse_b, "shift_share_b": m.shift_b / m.mse_b,
                        "mse_log_a": m.mse_la, "shift_share_log_a": m.shift_la / m.mse_la}).reset_index()
    return annotate(out)


def precheck_agreement(items: pd.DataFrame, rep: pd.DataFrame, precheck_study_dir) -> pd.DataFrame:
    """Units also fitted in the earlier centered-only pre-check (same simulated data, other runner):
    pre-check tier vs this run's centered (initial) tier, and the centered RMSEs of both executions.
    Cross-machine MCMC differs at Monte Carlo error, so tiers near a boundary may differ."""
    pre = pd.read_csv(Path(precheck_study_dir) / "replicate_metrics.csv")[["unit_id", "tier", "rmse_b", "rmse_log_a"]]
    cen = items[["unit_id", "true_b", "centered_b__est_b", "true_a", "centered_b__est_a"]].copy()
    cen["e_b"] = (cen["centered_b__est_b"] - cen["true_b"]) ** 2
    cen["e_la"] = (np.log(cen["centered_b__est_a"]) - np.log(cen["true_a"])) ** 2
    cen = np.sqrt(cen.groupby("unit_id")[["e_b", "e_la"]].mean()).rename(
        columns={"e_b": "run_centered_rmse_b", "e_la": "run_centered_rmse_log_a"}).reset_index()
    m = pre.rename(columns={"tier": "precheck_tier", "rmse_b": "precheck_rmse_b", "rmse_log_a": "precheck_rmse_log_a"}) \
        .merge(rep[["unit_id", "initial_tier", "fallback_triggered", "tier", "canonical_parameterization"]], on="unit_id") \
        .merge(cen, on="unit_id")
    m["precheck_would_trigger"] = m["precheck_tier"].isin(["divergent", "gross"])
    m["same_initial_tier"] = m["precheck_tier"] == m["initial_tier"]
    m["same_trigger"] = m["precheck_would_trigger"] == m["fallback_triggered"].astype(bool)
    return m


def coverage_table(primary: pd.DataFrame) -> pd.DataFrame:
    """Per condition x N: 95% coverage of b and a, MC SE, and the deviation from 0.95 in MC SE units
    (a descriptive consistency check, not a gate)."""
    t = primary[["role", "label", "condition_id", "N", "n_used", "coverage_b_mean", "coverage_b_mc_se",
                 "coverage_a_mean", "coverage_a_mc_se", "width_b_mean", "width_a_mean"]].copy()
    for p in ("b", "a"):
        t[f"coverage_{p}_z"] = (t[f"coverage_{p}_mean"] - 0.95) / t[f"coverage_{p}_mc_se"]
    return t


# ------------------------------------------------------------------ QA

def expected_cells(config: dict) -> pd.DataFrame:
    from recovery2pl.units import conditions_by_block
    reps = {b["name"]: b["replications"] for b in config["blocks"]}
    return pd.DataFrame([{"condition_id": c.condition_id, "N": c.N, "expected_replicates": reps[b]}
                         for b, conds in conditions_by_block(config).items() for c in conds])


def qa_checks(config, items, rep, merged_rep, tables, merged_tables, sens, jsons, merged_envelopes) -> list:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    exp = expected_cells(config)
    got = rep.groupby(["condition_id", "N"]).size().rename("replicates").reset_index()
    m = exp.merge(got, on=["condition_id", "N"], how="outer")
    add("every expected condition x N cell exists with its configured replicate count",
        len(m) == len(exp) and (m.replicates == m.expected_replicates).all(),
        f"{len(exp)} cells expected, {int(m.replicates.notna().sum())} present, "
        f"{int((m.replicates != m.expected_replicates).sum())} with a different count")
    add("unit count", rep.unit_id.nunique() == int(exp.expected_replicates.sum()), f"{rep.unit_id.nunique()} units")
    # canonical columns equal the canonical attempt's columns (no mixing of attempts)
    mism = 0
    for p in ("centered_b", "noncentered_b"):
        sel = items["canonical_parameterization"] == p
        for c in ITEM_COLS:
            a, b = items.loc[sel, c].to_numpy(), items.loc[sel, f"{p}__{c}"].to_numpy()
            mism += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    add("canonical estimates are exactly the canonical attempt's estimates", mism == 0, f"{mism} mismatching values")
    # replicate-level metrics recomputed from unit files == the run's merged replicate_metrics.csv
    num = [c for c in rep.columns if c in merged_rep.columns and pd.api.types.is_numeric_dtype(rep[c])]
    a = rep.set_index("unit_id").sort_index()[num]
    b = merged_rep.set_index("unit_id").sort_index().loc[a.index, num]
    d = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
    add("replicate metrics recomputed from unit files match the run's replicate_metrics.csv", d <= 1e-9, f"max |diff| {d:.2e}")
    for s in SUBSET_ORDER:
        t, mt = tables[s], merged_tables[s]
        cols = [c for c in t.columns if c in mt.columns and pd.api.types.is_numeric_dtype(t[c]) and c not in ("N",)]
        x = t.set_index(["condition_id", "N"]).sort_index()[cols]
        y = mt.set_index(["condition_id", "N"]).sort_index().loc[x.index, cols]
        d = float(np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))))
        add(f"{s} summary recomputed from replicate metrics matches the run's summary_{s}.csv", d <= 1e-9, f"max |diff| {d:.2e}")
        # Policy D subset definitions
        cnt = rep.groupby(["condition_id", "N"])["tier"].value_counts().unstack(fill_value=0)
        want = cnt[[c for c in SUBSETS[s] if c in cnt.columns]].sum(axis=1)
        have = t.set_index(["condition_id", "N"])["n_used"]
        add(f"{s} subset size equals its Policy-D tier counts", (want.loc[have.index] == have).all())
    # envelope = explicit max over main condition means; stress never enters
    for s in SUBSET_ORDER:
        for metric in METRICS:
            t = tables[s]
            explicit = t[t.role == "main"].pivot_table(index="N", columns="condition_id", values=f"{metric}_mean").max(axis=1)
            e = sens[(sens.subset == s) & (sens.metric == metric)].set_index("N")["value"]
            ok = np.allclose(explicit.loc[e.index].to_numpy(), e.to_numpy(), rtol=0, atol=1e-15)
            drivers = sens[(sens.subset == s) & (sens.metric == metric)]["driving_condition"]
            main_only = all(d.split()[0] in MAIN_POPULATIONS and d.split()[1] in MAIN_WORLDS for d in drivers)
            add(f"{s} {metric} envelope equals the explicit max over main conditions, drivers all main", ok and main_only)
    for metric, f in merged_envelopes.items():
        e = sens[(sens.subset == "primary") & (sens.metric == metric)].set_index("N")["value"]
        add(f"primary {metric} envelope matches the run's envelope_main file",
            np.allclose(f.set_index("N").loc[e.index, f"{metric}_mean"].to_numpy(), e.to_numpy(), rtol=0, atol=1e-12))
    # fallback reporting agrees with unit JSONs
    j = jsons.set_index("unit_id").sort_index()
    r = rep.set_index("unit_id").sort_index().loc[j.index]
    ok = ((j.tier == r.tier).all() and (j.initial_tier == r.initial_tier).all()
          and (j.fallback_triggered.astype(bool) == r.fallback_triggered.astype(bool)).all()
          and (j.canonical_parameterization == r.canonical_parameterization).all())
    add("tiers and fallback fields in the parquet files agree with every unit JSON", ok)
    add("every fit used the compiled C backend", (jsons.backend == "c").all(), jsons.backend.value_counts().to_dict())
    # S2 vs S1 pairing: identical true item banks for the same J, N, replicate
    cols = ["test_length", "N", "replicate", "item_id", "true_a", "true_b"]
    t2 = items[items.population == "S2"][cols]
    t1 = items[(items.population == "S1") & (items.world == "typical") & (items.rho == 0.3)][cols]
    pb = t2.merge(t1, on=cols[:4], suffixes=("_2", "_1"))
    d = max((pb.true_a_2 - pb.true_a_1).abs().max(), (pb.true_b_2 - pb.true_b_1).abs().max())
    add("S2 units are paired with S1-typical units that have the same true item banks (cross-CPU tol 1e-12)",
        len(pb) == len(t2) and d <= TRUTH_TOL, f"{len(pb)} of {len(t2)} item rows paired; max |diff| {d:.1e}")
    t0 = items[(items.population == "S0") & items.world.isin(MAIN_WORLDS) & (items.rho == 0.3)][cols + ["world"]]
    t1 = items[(items.population == "S1")][cols + ["world"]]
    pb = t1.merge(t0, on=cols[:4] + ["world"], suffixes=("_1", "_0"))
    d = max((pb.true_a_1 - pb.true_a_0).abs().max(), (pb.true_b_1 - pb.true_b_0 + 1.5).abs().max())
    add("S1 banks are the paired S0 banks translated by -1.5 (cross-CPU tol 1e-12)",
        len(pb) == len(t1) and d <= TRUTH_TOL, f"{len(pb)} of {len(t1)} item rows paired; max |diff| {d:.1e}")
    roles = annotate(rep).groupby("role").unit_id.nunique().to_dict()
    add("units by role (main / stress_S2 / stress_extreme) partition the study",
        sum(roles.values()) == rep.unit_id.nunique(), roles)
    add("primary subset = all units (no canonical gross fit)", int((rep.tier == "gross").sum()) == 0,
        f"{int((rep.tier == 'gross').sum())} canonical gross")
    return checks


# ------------------------------------------------------------------ figures

COL = {"S0 small": "#2a78d6", "S0 typical": "#eb6834", "S1 small": "#1baf7a", "S1 typical": "#e87ba4",
       "S2 typical": "#4a3aa7", "S0 extreme": "#e34948"}
NS = (50, 75, 100, 150, 200, 300, 500)


def _axes_n(ax):
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("responses per item, N")
    ax.grid(alpha=.25, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _short(lbl: str) -> str:
    p, w, j, r = lbl.split()
    return f"{p} {w[:3]} {j} {r.replace('rho', 'ρ')}"


def figures(sens: pd.DataFrame, tables: dict, s2: pd.DataFrame, conv: pd.DataFrame, cov: pd.DataFrame, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    made = []
    names = {"rmse_b": "difficulty b: RMSE", "rmse_log_a": "discrimination: RMSE(log a)"}

    # A. main figure: primary envelope only, no threshold line
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for ax, metric in zip(axes, ("rmse_b", "rmse_log_a")):
        e = sens[(sens.metric == metric) & (sens.subset == "primary")].sort_values("N")
        ax.errorbar(e.N, e.value, yerr=1.96 * e.mc_se, fmt="-o", color="#222", ms=5, lw=2, capsize=3)
        for _, r in e.iterrows():
            ax.annotate(_short(r.driving_condition), (r.N, r.value), xytext=(4, 8), textcoords="offset points",
                        ha="left", va="bottom", rotation=30, fontsize=6.5, color="#555")
        _axes_n(ax)
        ax.set_ylim(0, 1.25 * e.value.max())
        ax.set_ylabel(names[metric])
        ax.set_title(f"{names[metric]}\nconservative observed upper envelope (main scenarios)", fontsize=10)
    fig.text(0.01, 0.01, "At each N: the largest condition-mean RMSE across the tested main scenarios (populations S0, S1 x "
             "small, typical item worlds; all J, rho); label = condition attaining it;\nbars = +-1.96 Monte Carlo SE of "
             "that mean. Primary subset (all canonical fits). Not an expected value, not a confidence bound, not a guarantee.",
             fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "fig1_main_envelope.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # B. envelope sensitivity to convergence filtering
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    style = {"primary": ("-", "o", "primary (all canonical fits)"), "no_divergent": ("--", "s", "excluding divergent fits"),
             "strict": (":", "^", "clean fits only")}
    for ax, metric in zip(axes, ("rmse_b", "rmse_log_a")):
        for s in SUBSET_ORDER:
            e = sens[(sens.metric == metric) & (sens.subset == s)].sort_values("N")
            ls, mk, lab = style[s]
            ax.plot(e.N, e.value, ls=ls, marker=mk, ms=5, lw=1.8, label=lab)
        _axes_n(ax); ax.set_ylim(0, None); ax.set_ylabel(names[metric])
        ax.set_title(f"main envelope by convergence subset — {names[metric]}", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Main conservative envelope (max over S0, S1 x small, typical conditions of the condition-mean RMSE) "
             "computed separately on three Policy-D subsets: all canonical fits (primary), excluding divergent fits, and "
             "clean fits only.\nFiltering removes replicates non-randomly (e.g. 50 of 100 remain in the N = 50 driver cell "
             "under 'clean only'); the primary subset is the headline.", fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "fig2_envelope_sensitivity.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # C. condition-specific mean curves (main conditions), +-1.96 MC SE
    prim = tables["primary"]
    for metric in ("rmse_b", "rmse_log_a", "rmse_a"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, pop in zip(axes, MAIN_POPULATIONS):
            g = prim[(prim.role == "main") & (prim.population == pop)]
            for lbl, c in g.groupby("label"):
                c = c.sort_values("N"); _, w, j, r = lbl.split()
                ax.errorbar(c.N, c[f"{metric}_mean"], yerr=1.96 * c[f"{metric}_mc_se"], color=COL[f"{pop} {w}"],
                            ls="-" if j == "J25" else "--", marker="o" if r == "rho0" else "s", ms=4, lw=1.5, capsize=2,
                            label=_short(lbl))
            _axes_n(ax); ax.set_title(f"{pop} — condition means (±1.96 MC SE)", fontsize=10)
            ax.legend(fontsize=7, frameon=False, ncol=2)
        axes[0].set_ylabel(metric.replace("rmse_", "RMSE ").replace("log_a", "log a"))
        fig.text(0.01, 0.01, "Mean over replicates of the per-replicate RMSE for every main condition (sma = small, typ = "
                 "typical item world; J = test length; ρ = corr(α, β)); bars = +-1.96 Monte Carlo SE of the mean. "
                 "Primary subset; R = 100 per point.", fontsize=7.2, color="#555")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p = out / f"fig3_conditions_{metric}.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # D. replicate variation (5th-95th percentile of replicate RMSE) for main conditions, rho 0.3
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for col, (pop, w) in enumerate([(p, w) for p in MAIN_POPULATIONS for w in MAIN_WORLDS]):
        for row, metric in enumerate(("rmse_b", "rmse_log_a")):
            ax = axes[row][col]
            g = prim[(prim.population == pop) & (prim.world == w) & (prim.rho == 0.3)]
            for j, c in g.groupby("test_length"):
                c = c.sort_values("N"); color = COL[f"{pop} {w}"]
                ax.fill_between(c.N, c[f"{metric}_q05"], c[f"{metric}_q95"], color=color, alpha=.12 if j == 25 else .22, lw=0)
                ax.plot(c.N, c[f"{metric}_mean"], color=color, ls="-" if j == 25 else "--", lw=1.6, label=f"J{j} mean")
            _axes_n(ax)
            if row == 0: ax.set_title(f"{pop} {w}, ρ 0.3", fontsize=10)
            if col == 0: ax.set_ylabel(metric.replace("rmse_", "RMSE ").replace("log_a", "log a"))
            ax.legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.005, "Lines: mean over replicates; bands: 5th-95th percentile of replicate-level RMSE "
             "(between-replicate variation, not uncertainty of the mean).", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "fig4_replicate_variation.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # E. S2 vs matched S1: paired difference (same banks, replicates 0-49)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, metric in zip(axes, ("rmse_b", "rmse_log_a")):
        for j, g in s2.groupby("test_length"):
            g = g.sort_values("N")
            ax.errorbar(g.N * (0.98 if j == 25 else 1.02), g[f"{metric}_diff_mean"], yerr=1.96 * g[f"{metric}_diff_mc_se"],
                        color=COL["S2 typical"], ls="-" if j == 25 else "--", marker="o" if j == 25 else "s", ms=4,
                        capsize=2, label=f"J{j}")
        ax.axhline(0, color="#222", lw=.8)
        _axes_n(ax); ax.set_ylabel("S2 minus S1 (typical, ρ 0.3)")
        ax.set_title(f"S2 stress vs matched S1, paired difference — {names[metric]}", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Mean over 50 replicate pairs (same item banks, same replicate index); bars = +-1.96 MC SE of the "
             "paired mean difference. Primary subset.", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / "fig5_s2_vs_s1.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # F. extreme world, per condition, by convergence subset (n used annotated; subsets can be tiny)
    ext_labels = sorted(tables["primary"].loc[tables["primary"].role == "stress_extreme", "label"].unique())
    fig, axes = plt.subplots(2, len(ext_labels), figsize=(4 * len(ext_labels), 7.5), sharey="row")
    env = {m: sens[(sens.metric == m) & (sens.subset == "primary")].sort_values("N") for m in ("rmse_b", "rmse_log_a")}
    for col, lbl in enumerate(ext_labels):
        for row, metric in enumerate(("rmse_b", "rmse_log_a")):
            ax = axes[row][col]
            for s in SUBSET_ORDER:
                c = tables[s][tables[s].label == lbl].sort_values("N").dropna(subset=[f"{metric}_mean"])
                ls, mk, lab = style[s]
                ax.plot(c.N, c[f"{metric}_mean"], ls=ls, marker=mk, ms=4, lw=1.4, label=lab)
                if s != "primary":
                    for _, r in c.iterrows():
                        ax.annotate(f"n={int(r.n_used)}", (r.N, r[f"{metric}_mean"]), xytext=(3, 3 if s == "strict" else -9),
                                    textcoords="offset points", fontsize=6, color="#666")
            ax.plot(env[metric].N, env[metric].value, color="#222", lw=1, alpha=.6, label="main envelope (primary)")
            _axes_n(ax)
            if row == 0: ax.set_title(_short(lbl) + " (R = 20)", fontsize=9)
            if col == 0: ax.set_ylabel(names[metric])
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.005, "Extreme item world: stress condition only, never part of the main envelope. Primary uses all 20 "
             "replicates; n = replicates left after excluding divergent / non-clean fits (missing point: none left).",
             fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "fig6_extreme_stress.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # G. convergence and fallback by condition (not pooled)
    groups = [("S0", "small"), ("S0", "typical"), ("S1", "small"), ("S1", "typical"), ("S2", "typical"), ("S0", "extreme")]
    fig, axes = plt.subplots(2, len(groups), figsize=(3.2 * len(groups), 6.4), sharey=True)
    for col, (pop, w) in enumerate(groups):
        g = conv[(conv.population == pop) & (conv.world == w)]
        for (j, r), c in g.groupby(["test_length", "rho"]):
            c = c.sort_values("N")
            kw = dict(color=COL[f"{pop} {w}"], ls="-" if j == 25 else "--", marker="o" if r == 0 else "s", ms=3.5, lw=1.3,
                      label=f"J{j} ρ{r:g}")
            axes[0][col].plot(c.N, 100 * c.fallback_trigger_rate, **kw)
            axes[1][col].plot(c.N, 100 * c.rate_divergent, **kw)
        axes[0][col].set_title(f"{pop} {w}", fontsize=10)
        axes[0][col].legend(fontsize=7, frameon=False)
        for ax in axes[:, col]: _axes_n(ax)
    axes[0][0].set_ylabel("fallback triggered, % of units\n(centered fit divergent or gross)")
    axes[1][0].set_ylabel("canonical fit divergent, % of units\n(after fallback)")
    fig.text(0.01, 0.005, "Top: share of units whose centered-b fit was divergent or gross, triggering the non-centered-b "
             "refit. Bottom: share whose canonical (selected) fit is still divergent; these stay in the primary subset, "
             "flagged. One line per J x ρ condition; R = 100 (S0, S1 main), 50 (S2), 20 (extreme).", fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / "fig7_convergence_fallback.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)

    # G. coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, par in zip(axes, ("b", "a")):
        for lbl, g in cov.groupby("label"):
            g = g.sort_values("N"); pop, w = lbl.split()[:2]
            ax.plot(g.N, g[f"coverage_{par}_mean"], color=COL[f"{pop} {w}"], lw=1, alpha=.8, marker=".", ms=3)
        ax.axhline(0.95, color="#222", lw=1, ls="--")
        _axes_n(ax); ax.set_title(f"empirical 95% interval coverage of {par}, every condition (nominal 0.95 dashed)", fontsize=9)
    axes[0].set_ylabel("coverage")
    handles = [plt.Line2D([], [], color=c, label=k) for k, c in COL.items()]
    axes[1].legend(handles=handles, fontsize=7, frameon=False)
    fig.text(0.01, 0.01, "Share of items whose 95% posterior interval contains the true value, averaged over replicates; one "
             "line per condition (all J, ρ), colour = population x item world. Dashed line: nominal 0.95 (a reference, "
             "not a threshold). Primary subset.", fontsize=7.2, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / "fig8_coverage.png"; fig.savefig(p, dpi=170); plt.close(fig); made.append(p)
    return made


# ------------------------------------------------------------------ driver

def run_analysis(config: dict, merged_root, analysis_out, precheck_dir=None) -> dict:
    study = Path(merged_root) / config["study_id"]
    out = Path(analysis_out)
    if out.resolve().is_relative_to(study.resolve()):
        raise ValueError("analysis output must not be inside the results tree it reads")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    items = load_canonical_items(study)
    rep = replicate_metrics(items, config["reporting"]["item_error_tolerances"])
    tables = condition_tables(rep)
    sens = envelope_sensitivity(tables)
    look = lookup_table(sens)
    s2 = paired_s2_vs_s1(rep)
    conv = convergence_table(rep)
    cov = coverage_table(tables["primary"])
    jsons = load_unit_jsons(study)
    merged_rep = pd.read_csv(study / "replicate_metrics.csv")
    merged_tables = {s: pd.read_csv(study / f"summary_{s}.csv") for s in SUBSET_ORDER}
    merged_env = {m: pd.read_csv(study / f"envelope_main_{m}.csv") for m in METRICS}
    checks = qa_checks(config, items, rep, merged_rep, tables, merged_tables, sens, jsons, merged_env)
    t = out / "tables"
    rep.to_parquet(t / "replicate_metrics.parquet", index=False)
    for s, df in tables.items():
        df.to_csv(t / f"conditions_{s}.csv", index=False)
    for name, df in (("envelope_sensitivity", sens), ("main_lookup", look), ("s2_vs_s1_paired", s2),
                     ("convergence_fallback", conv), ("coverage", cov)):
        df.to_csv(t / f"{name}.csv", index=False)
    error_decomposition(items).to_csv(t / "error_decomposition.csv", index=False)
    if precheck_dir:
        precheck_agreement(items, rep, precheck_dir).to_csv(t / "precheck_agreement.csv", index=False)
    (out / "qa_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    figs = figures(sens, tables, s2, conv, cov, out / "figures")
    return {"checks": checks, "tables": sorted(str(p.name) for p in t.iterdir()), "figures": [str(p) for p in figs]}
