"""Run explicit work units; each unit writes its own compact, independent result files.

Layout under an output root:
    <out>/<study_id>/study.json                            config copy (written once, verified after)
    <out>/<study_id>/units/<condition_id>/rep_XXX.parquet  item rows: true / estimate / interval
    <out>/<study_id>/units/<condition_id>/rep_XXX.json     fit diagnostics, tier, seeds, hash

Both files are written atomically (temp file + rename); the .json is written LAST and marks the
unit complete. A complete unit whose scientific hash matches is skipped; a hash mismatch raises and
nothing is overwritten. Results never depend on which other units ran, or in what order.
"""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from recovery2pl import __version__
from recovery2pl.config import unit_scientific_hash
from recovery2pl.simulate import simulate_replicate

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def unit_paths(out_root, study_id: str, unit) -> tuple:
    d = Path(out_root) / study_id / "units" / unit.condition.condition_id
    return d / f"rep_{unit.replicate:03d}.parquet", d / f"rep_{unit.replicate:03d}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ensure_study(config: dict, out_root) -> Path:
    sdir = Path(out_root) / config["study_id"]
    meta = sdir / "study.json"
    if meta.exists():
        stored = json.loads(meta.read_text(encoding="utf-8"))
        if stored["config"] != config:
            raise RuntimeError(f"{meta} holds a different config for study_id "
                               f"{config['study_id']!r}; use a new study_id")
        return sdir
    sdir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(meta, json.dumps({"study_id": config["study_id"], "config": config,
                                         "package_version": __version__}, indent=2))
    return sdir


def unit_status(config: dict, unit, out_root) -> str:
    """'done' | 'pending'; raises on a stored unit with a different scientific hash."""
    pq, js = unit_paths(out_root, config["study_id"], unit)
    if not js.exists():
        return "pending"
    stored = json.loads(js.read_text(encoding="utf-8"))
    if stored["scientific_hash"] != unit_scientific_hash(config, unit.condition):
        raise RuntimeError(f"{js} was produced with a different scientific config; not overwritten")
    return "done" if pq.exists() else "pending"


def _default_fit(Y, config, seed):
    from recovery2pl.model import fit_2pl                      # PyMC is heavy: import lazily
    return fit_2pl(Y, config["estimator"], config["diagnostics"], seed,
                   cores=config["execution"]["chain_cores"])


def run_unit(config: dict, unit, out_root, fit_fn=None) -> dict:
    """Simulate + fit + persist one unit. `fit_fn(Y, config, seed) -> dict` is injectable (tests)."""
    fit_fn = fit_fn or _default_fit
    cond, rep = unit.condition, unit.replicate
    pq, js = unit_paths(out_root, config["study_id"], unit)
    sim = simulate_replicate(config, cond, rep)
    fit = fit_fn(sim["Y"], config, sim["sampler_seed"])
    J = cond.test_length
    nan = float("nan")
    rows = pd.DataFrame({
        "unit_id": unit.unit_id, "block": unit.block, "condition_id": cond.condition_id,
        "population": cond.population, "difficulty_shift": cond.difficulty_shift,
        "theta_kind": cond.theta_kind, "theta_shape": cond.theta_shape,
        "N": cond.N, "test_length": J, "world": cond.world, "tau_a": cond.tau_a,
        "tau_b": cond.tau_b, "rho": cond.rho, "replicate": rep, "tier": fit["tier"],
        "item_id": range(J), "true_a": sim["a"], "true_b": sim["b"],
    })
    for p in ("a", "b"):
        for k in ("est", "lo", "hi", "sd"):
            rows[f"{k}_{p}"] = fit.get(f"{k}_{p}", nan)
        rows[f"rhat_{p}"] = fit.get(f"rhat_{p}", nan)
    rows["est_log_a"] = fit.get("est_log_a", nan)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq.with_name(pq.name + ".tmp")
    rows.to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    keys = ("tier", "error", "seconds", "backend", "rhat_max", "ess_bulk_min", "ess_tail_min",
            "rhat_max_group", "ess_bulk_min_group", "theta_rhat_max", "theta_ess_bulk_min",
            "divergences", "divergence_share", "ebfmi_min", "treedepth_hits", "hyper")
    diag = {"unit_id": unit.unit_id, "block": unit.block, "condition_id": cond.condition_id,
            "replicate": rep, "scientific_hash": unit_scientific_hash(config, cond),
            "sampler_seed": sim["sampler_seed"], "item_tries": sim["item_tries"],
            "observed_p_correct": float(sim["Y"].mean()),
            **{k: fit.get(k) for k in keys},
            "package_version": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "finished_at": _now()}
    _atomic_write_text(js, json.dumps(diag, indent=2, default=float))
    return diag


def collect(sources, out_root, study_id: str, atol: float = 1e-8) -> dict:
    """Merge unit results found anywhere under `sources` (e.g. downloaded artifact folders) into
    <out_root>/<study_id>/units/. A unit already present is kept; a second copy (e.g. from a job
    retry) must carry the same scientific hash and the same true parameters, and its estimates must
    agree within `atol`, otherwise an error is raised. Nothing is overwritten."""
    import shutil
    import numpy as np
    copied = duplicates = 0
    for src in sources:
        for js in sorted(Path(src).rglob("rep_*.json")):
            if js.parent.parent.name != "units" or js.parent.parent.parent.name != study_id:
                continue
            pq = js.with_suffix(".parquet")
            if not pq.exists():
                continue
            dst_dir = Path(out_root) / study_id / "units" / js.parent.name
            dst_js, dst_pq = dst_dir / js.name, dst_dir / pq.name
            if dst_js.exists():
                a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (dst_js, js))
                if a["scientific_hash"] != b["scientific_hash"]:
                    raise RuntimeError(f"{js}: scientific hash differs from the collected copy")
                ra, rb = pd.read_parquet(dst_pq), pd.read_parquet(pq)
                for col in ("true_a", "true_b"):
                    if not np.array_equal(ra[col].to_numpy(), rb[col].to_numpy()):
                        raise RuntimeError(f"{js}: simulated {col} differs from the collected copy")
                for col in ("est_a", "est_b"):
                    if not np.allclose(ra[col], rb[col], rtol=0, atol=atol, equal_nan=True):
                        raise RuntimeError(f"{js}: {col} differs beyond {atol} from the collected copy")
                duplicates += 1
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pq, dst_pq)
            shutil.copy2(js, dst_js)                       # json last: marks the unit complete
            copied += 1
    return {"copied": copied, "verified_duplicates": duplicates}


def run_units(config: dict, units: list, out_root, fit_fn=None, log=print) -> dict:
    """Run the given units sequentially (sampler chains run in parallel inside a unit). Units
    already complete are skipped. A crash outside the fit (e.g. disk) is reported, not hidden."""
    for name in THREAD_ENV:
        os.environ.setdefault(name, str(config["execution"]["threads_per_process"]))
    ensure_study(config, out_root)
    t0 = time.perf_counter()
    done = skipped = crashed = 0
    failures, sampler_errors = [], []
    for k, unit in enumerate(units, 1):
        if unit_status(config, unit, out_root) == "done":
            skipped += 1
            continue
        try:
            d = run_unit(config, unit, out_root, fit_fn)
            done += 1
            if d["error"]:
                sampler_errors.append(unit.unit_id)          # recorded as a gross fit, and reported
            log(f"[{_now()}] {k}/{len(units)} {unit.unit_id}: tier={d['tier']} "
                f"{(d['seconds'] or 0):.1f}s rhat={d['rhat_max']} div={d['divergences']}"
                + (f" ERROR {d['error']}" if d["error"] else ""))
        except Exception as exc:                                   # noqa: BLE001
            crashed += 1
            failures.append(unit.unit_id)
            log(f"[{_now()}] {unit.unit_id} CRASHED: {exc!r}")
    return {"ran": done, "skipped": skipped, "crashed": crashed, "failed_units": failures,
            "sampler_error_units": sampler_errors, "seconds": time.perf_counter() - t0}
