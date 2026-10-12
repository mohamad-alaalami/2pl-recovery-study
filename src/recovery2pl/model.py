"""The fitted model: hierarchical Bayesian 2PL with a centered difficulty hierarchy (PyMC, NUTS).

    theta_s ~ N(0, 1)                                  identification of the latent scale
    log a_i = mu_log_a + sigma_log_a * z_a_i,  z_a_i ~ N(0, 1)      (non-centered)
    b_i     ~ N(mu_b, sigma_b)                                       (centered)
    mu_log_a ~ N(m_la, s_la)   sigma_log_a ~ HalfNormal(c_la)
    mu_b     ~ N(m_b, s_b)     sigma_b     ~ HalfNormal(c_b)
    Y_si ~ Bernoulli(sigmoid(a_i (theta_s - b_i)))

The model sees only the response matrix Y; true parameters are used afterwards, for recovery.

Difficulty parameterization (same model and posterior, different sampling geometry):
    hierarchical_2pl_centered_b      b ~ N(mu_b, sigma_b)                 (default, fitted first)
    hierarchical_2pl_noncentered_b   b = mu_b + sigma_b * z_b, z_b ~ N(0, 1)
                                     (the automatic fallback of recovery2pl.fallback; also usable
                                      alone for diagnostics)

Diagnostics (Vehtari et al. 2021, rank-normalized): split R-hat, bulk-ESS and tail-ESS over every
item parameter (a, b) and every hyperparameter; divergences; E-BFMI; max-treedepth hits; theta
summarized separately (never used for the fit tier).

Fit tiers (`classify_fit`), first match wins:
    gross        sampler error, non-finite estimates, or max R-hat > gross_rhat
    divergent    any divergent transition
    mixing_flag  R-hat > rhat_max, bulk-ESS < ess_bulk_min or tail-ESS < ess_tail_min
    clean        otherwise
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from recovery2pl import toolchain

toolchain.configure()

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402

GROUP_VARS = ["mu_log_a", "sigma_log_a", "mu_b", "sigma_b"]
TIERS = ("clean", "mixing_flag", "divergent", "gross")
from recovery2pl.fallback import CENTERED, NONCENTERED  # noqa: E402


def build_model(Y: np.ndarray, priors: dict, kind: str = CENTERED) -> pm.Model:
    if kind not in (CENTERED, NONCENTERED):
        raise ValueError(f"unknown estimator kind {kind!r}")
    N, J = Y.shape
    with pm.Model() as model:
        mu_log_a = pm.Normal("mu_log_a", priors["mu_log_a_mean"], priors["mu_log_a_sd"])
        sigma_log_a = pm.HalfNormal("sigma_log_a", priors["sigma_log_a_scale"])
        mu_b = pm.Normal("mu_b", priors["mu_b_mean"], priors["mu_b_sd"])
        sigma_b = pm.HalfNormal("sigma_b", priors["sigma_b_scale"])
        z_a = pm.Normal("z_a", 0.0, 1.0, shape=J)
        a = pm.Deterministic("a", pm.math.exp(mu_log_a + sigma_log_a * z_a))
        if kind == CENTERED:
            b = pm.Normal("b", mu_b, sigma_b, shape=J)
        else:
            z_b = pm.Normal("z_b", 0.0, 1.0, shape=J)
            b = pm.Deterministic("b", mu_b + sigma_b * z_b)
        theta = pm.Normal("theta", 0.0, 1.0, shape=N)
        pm.Bernoulli("y", logit_p=a[None, :] * (theta[:, None] - b[None, :]), observed=Y)
    return model


def _extreme(stat, names, fn):
    return float(fn([float(fn(stat[v].values)) for v in names]))


def diagnostics(idata) -> dict:
    checked = ["a", "b"] + GROUP_VARS
    rhat = az.rhat(idata, var_names=checked + ["theta"])
    bulk = az.ess(idata, var_names=checked + ["theta"], method="bulk")
    tail = az.ess(idata, var_names=checked + ["theta"], method="tail")
    st = idata.sample_stats
    tree = st["reached_max_treedepth"] if "reached_max_treedepth" in st else None
    n_draws = int(st["diverging"].size)
    div = int(st["diverging"].sum())
    return {
        "rhat_max": _extreme(rhat, checked, np.nanmax),
        "ess_bulk_min": _extreme(bulk, checked, np.nanmin),
        "ess_tail_min": _extreme(tail, checked, np.nanmin),
        "rhat_max_group": _extreme(rhat, GROUP_VARS, np.nanmax),
        "ess_bulk_min_group": _extreme(bulk, GROUP_VARS, np.nanmin),
        "theta_rhat_max": float(np.nanmax(rhat["theta"].values)),
        "theta_ess_bulk_min": float(np.nanmin(bulk["theta"].values)),
        "divergences": div,
        "divergence_share": div / n_draws if n_draws else float("nan"),
        "ebfmi_min": float(np.min(az.bfmi(idata))),
        "treedepth_hits": int(tree.sum()) if tree is not None else None,
        "rhat_a": rhat["a"].values, "rhat_b": rhat["b"].values,
    }


def classify_fit(d: dict, rules: dict) -> str:
    """Policy D tier of one fit (see module docstring). `d` holds the fit result / diagnostics."""
    if d.get("error"):
        return "gross"
    est = [d.get(k) for k in ("est_a", "est_b")]
    if any(e is None or not np.all(np.isfinite(e)) for e in est):
        return "gross"
    stats = [d.get(k) for k in ("rhat_max", "ess_bulk_min", "ess_tail_min")]
    if any(s is None or not np.isfinite(s) for s in stats):
        return "gross"
    if d["rhat_max"] > rules["gross_rhat"]:
        return "gross"
    if d.get("divergences", 0) > rules["max_divergences"]:
        return "divergent"
    if (d["rhat_max"] > rules["rhat_max"] or d["ess_bulk_min"] < rules["ess_bulk_min"]
            or d["ess_tail_min"] < rules["ess_tail_min"]):
        return "mixing_flag"
    return "clean"


def fit_2pl(Y: np.ndarray, estimator: dict, rules: dict, seed: int, cores: int = 1) -> dict:
    """Fit one dataset. Never raises on a poor or failed fit: a sampler error is returned in
    `error` and classified gross. Raises only if no C backend is available (environment error)."""
    backend = toolchain.require_c_backend()
    s = estimator["sampler"]
    t0 = time.perf_counter()
    out = {"backend": backend, "error": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with build_model(Y, estimator["priors"], estimator["kind"]):
                idata = pm.sample(draws=s["draws"], tune=s["tune"], chains=s["chains"], cores=cores,
                                  target_accept=s["target_accept"], random_seed=seed,
                                  progressbar=False, compute_convergence_checks=False)
    except Exception as exc:                                   # noqa: BLE001 -- recorded, not hidden
        out.update(error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)
        out["tier"] = classify_fit(out, rules)
        return out
    post = idata.posterior
    lo_q, hi_q = (1 - s["interval"]) / 2, 1 - (1 - s["interval"]) / 2
    for name in ("a", "b"):
        draws = post[name].stack(sample=("chain", "draw")).values           # J x S
        out[f"est_{name}"] = draws.mean(axis=1)
        out[f"lo_{name}"] = np.quantile(draws, lo_q, axis=1)
        out[f"hi_{name}"] = np.quantile(draws, hi_q, axis=1)
        out[f"sd_{name}"] = draws.std(axis=1, ddof=1)
    la = np.log(post["a"].stack(sample=("chain", "draw")).values)
    out["est_log_a"] = la.mean(axis=1)                    # posterior mean of log a (not log of mean a)
    out.update(diagnostics(idata))
    out["hyper"] = {v: float(post[v].mean()) for v in GROUP_VARS}
    out["seconds"] = time.perf_counter() - t0
    out["tier"] = classify_fit(out, rules)
    return out
