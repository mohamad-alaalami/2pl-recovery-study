"""Run explicit work units; each unit writes its own compact, independent result files.

Layout under an output root:
    <out>/<study_id>/study.json                            config copy (written once, verified after)
    <out>/<study_id>/units/<condition_id>/rep_XXX.parquet  item rows: truth, canonical estimates,
                                                           every attempt's estimates
    <out>/<study_id>/units/<condition_id>/rep_XXX.json     canonical diagnostics, fit protocol,
                                                           per-attempt diagnostics, seeds, hashes
    <out>/<study_id>/failures/<condition_id>/...json       sampler/program failures (unit stays pending)

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
from recovery2pl.config import simulation_hash, unit_scientific_hash
from recovery2pl.fallback import (CENTERED, LABEL, NONCENTERED, RULE, plan_for, select_canonical,
                                  triggers)
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


DIAG_KEYS = ("tier", "error", "seconds", "backend", "rhat_max", "ess_bulk_min", "ess_tail_min",
             "rhat_max_group", "ess_bulk_min_group", "theta_rhat_max", "theta_ess_bulk_min",
             "divergences", "divergence_share", "ebfmi_min", "treedepth_hits", "hyper")
ITEM_KEYS = ("est_a", "lo_a", "hi_a", "sd_a", "rhat_a", "est_b", "lo_b", "hi_b", "sd_b", "rhat_b",
             "est_log_a")


class UnitFailure(RuntimeError):
    """A program/sampler/runtime failure (NOT a Policy-D gross fit): the unit is not written, a
    failure record is kept, and the unit stays pending so it can be retried."""

    def __init__(self, message: str, kind: str = "sampler_error"):
        super().__init__(message)
        self.kind = kind


def _default_fit(Y, config, seed, kind):
    from recovery2pl.model import fit_2pl                      # PyMC is heavy: import lazily
    est = {k: v for k, v in config["estimator"].items() if k != "fallback"}
    est["kind"] = kind
    return fit_2pl(Y, est, config["diagnostics"], seed, cores=config["execution"]["chain_cores"])


def _attempt_record(fit: dict, kind: str, position) -> dict:
    return {"parameterization": LABEL[kind], "estimator_kind": kind,
            **{k: fit.get(k) for k in DIAG_KEYS}, "runner": runner_metadata(),
            "position_in_batch": position, "finished_at": _now()}


def _record_failure(config, unit, out_root, attempts: dict, failed_kind: str, error) -> None:
    d = Path(out_root) / config["study_id"] / "failures" / unit.condition.condition_id
    d.mkdir(parents=True, exist_ok=True)
    stamp = _now().replace(":", "")
    _atomic_write_text(d / f"rep_{unit.replicate:03d}__{stamp}.json", json.dumps(
        {"unit_id": unit.unit_id, "failed_attempt": LABEL[failed_kind], "error": str(error),
         "attempts": attempts, "scientific_hash": unit_scientific_hash(config, unit.condition),
         "recorded_at": _now()}, indent=2, default=float))


def run_unit(config: dict, unit, out_root, fit_fn=None, position: int = None) -> dict:
    """Simulate once, fit under the configured protocol, persist one unit.

    Protocol (recovery2pl.fallback): fit the configured estimator; if the config enables the frozen
    fallback and the first fit is divergent or gross, refit the SAME dataset with non-centered b and
    pick the canonical fit by Policy-D tier only (ties keep the first fit). Both attempts are kept:
    canonical estimates in the standard columns, every attempt's item estimates in
    `<parameterization>__<column>` columns, every attempt's diagnostics in the json.

    A sampler/program exception in ANY attempt is a UnitFailure, not a gross fit: nothing is written
    for the unit (it stays pending for a retry), a failure record is kept under <study>/failures/,
    and the exception is raised. `fit_fn(Y, config, seed, kind) -> dict` is injectable (tests)."""
    fit_fn = fit_fn or _default_fit
    cond, rep = unit.condition, unit.replicate
    pq, js = unit_paths(out_root, config["study_id"], unit)
    sim = simulate_replicate(config, cond, rep)              # simulated ONCE; every attempt reuses it
    first_kind, fallback_enabled = plan_for(config)
    fits, attempts = {}, {}

    def attempt(kind):
        fit = fit_fn(sim["Y"], config, sim["sampler_seed"], kind)   # same data, same seed
        attempts[LABEL[kind]] = _attempt_record(fit, kind, position)
        if fit.get("error"):
            _record_failure(config, unit, out_root, attempts, kind, fit["error"])
            raise UnitFailure(f"{unit.unit_id}: {LABEL[kind]} attempt failed: {fit['error']}")
        fits[kind] = fit

    attempt(first_kind)
    initial_tier = fits[first_kind]["tier"]
    triggered = first_kind == CENTERED and fallback_enabled and triggers(initial_tier)
    if triggered:
        attempt(NONCENTERED)
    if first_kind == CENTERED and fallback_enabled:
        canonical_label, reason = select_canonical(initial_tier, fits[NONCENTERED]["tier"] if triggered else None)
    else:
        canonical_label, reason = LABEL[first_kind], "single-fit protocol (no fallback configured)"
    canonical = fits[CENTERED if canonical_label == LABEL[CENTERED] else NONCENTERED]
    protocol = {"rule": RULE if fallback_enabled else None,
                "initial_parameterization": LABEL[first_kind], "initial_tier": initial_tier,
                "fallback_triggered": triggered,
                "fallback_parameterization": LABEL[NONCENTERED] if triggered else None,
                "fallback_tier": fits[NONCENTERED]["tier"] if triggered else None,
                "canonical_parameterization": canonical_label, "canonical_tier": canonical["tier"],
                "selection_reason": reason}

    J = cond.test_length
    nan = float("nan")
    rows = pd.DataFrame({
        "unit_id": unit.unit_id, "block": unit.block, "condition_id": cond.condition_id,
        "population": cond.population, "difficulty_shift": cond.difficulty_shift,
        "theta_kind": cond.theta_kind, "theta_shape": cond.theta_shape,
        "N": cond.N, "test_length": J, "world": cond.world, "tau_a": cond.tau_a,
        "tau_b": cond.tau_b, "rho": cond.rho, "replicate": rep, "tier": canonical["tier"],
        "initial_tier": initial_tier, "fallback_triggered": triggered,
        "fallback_tier": protocol["fallback_tier"] or "", "canonical_parameterization": canonical_label,
        "item_id": range(J), "true_a": sim["a"], "true_b": sim["b"],
    })
    for k in ITEM_KEYS:                                       # canonical estimates (used by metrics)
        rows[k] = canonical.get(k, nan)
    for kind, fit in fits.items():                            # every attempt, preserved
        for k in ITEM_KEYS:
            rows[f"{LABEL[kind]}__{k}"] = fit.get(k, nan)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq.with_name(pq.name + ".tmp")
    rows.to_parquet(tmp, index=False)
    os.replace(tmp, pq)
    diag = {"unit_id": unit.unit_id, "block": unit.block, "condition_id": cond.condition_id,
            "replicate": rep, "scientific_hash": unit_scientific_hash(config, cond),
            "simulation_hash": simulation_hash(config, cond),
            "sampler_seed": sim["sampler_seed"], "item_tries": sim["item_tries"],
            "observed_p_correct": float(sim["Y"].mean()),
            **{k: canonical.get(k) for k in DIAG_KEYS},           # canonical diagnostics
            "seconds_total": float(sum((f.get("seconds") or 0.0) for f in fits.values())),
            "fit_protocol": protocol, "attempts": attempts,
            "package_version": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "runner": runner_metadata(),
            "position_in_batch": position, "finished_at": _now()}
    _atomic_write_text(js, json.dumps(diag, indent=2, default=float))
    return diag


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def runner_metadata() -> dict:
    """Execution provenance (never part of the scientific hash)."""
    env = os.environ
    return {"cpu_model": cpu_model(), "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "github_run_id": env.get("GITHUB_RUN_ID"), "github_run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
            "github_job": env.get("GITHUB_JOB"), "shard": env.get("SHARD"),
            "runner_name": env.get("RUNNER_NAME"), "runner_image": env.get("ImageOS"),
            "runner_image_version": env.get("ImageVersion")}


TRUTH_TOL = 1e-12      # simulated parameters must agree (rounding-level differences only)
COMPARED = ("est_a", "est_b", "est_log_a", "lo_a", "hi_a", "lo_b", "hi_b")


def _execution_key(js: Path) -> str:
    """Byte digest of a unit's two files: copies of one execution are byte-identical."""
    import hashlib
    h = hashlib.sha256(js.read_bytes())
    h.update(js.with_suffix(".parquet").read_bytes())
    return h.hexdigest()


def collect(sources, out_root, study_id: str) -> dict:
    """Merge unit results found under `sources` (e.g. downloaded artifact folders) into
    <out_root>/<study_id>/units/ with ONE canonical result per scientific unit.

    Canonical copy: a unit already collected in <out_root> is never replaced; otherwise the copy
    with the earliest `finished_at` (ties: lexicographically first source path). Copies of the SAME
    execution (e.g. a shard artifact and the merged artifact) are recognized and ignored.

    Other executions of the same unit (retries, reruns on another machine) must have the same
    scientific hash and the same simulated parameters (within 1e-12) -- otherwise an error is raised.
    Posterior summaries MAY differ (MCMC is not bit-reproducible across CPUs); they are never
    averaged or overwritten: each such duplicate is preserved under <study>/duplicates/ and logged,
    with its numeric differences, in <study>/collect_duplicates.jsonl."""
    import shutil
    import numpy as np
    sdir = Path(out_root) / study_id
    found = {}
    for src in sorted(map(str, sources)):
        for js in sorted(Path(src).rglob("rep_*.json")):
            if js.parent.parent.name != "units" or js.parent.parent.parent.name != study_id:
                continue
            if Path(out_root).resolve() in js.resolve().parents:
                continue                                    # never re-read our own output tree
            if js.with_suffix(".parquet").exists():
                found.setdefault((js.parent.name, js.name), []).append(js)
    copied = same_execution = already_logged = 0
    logged = []
    for (cond_dir, name), copies in sorted(found.items()):
        metas = [(json.loads(p.read_text(encoding="utf-8")), p) for p in copies]
        dst_dir = sdir / "units" / cond_dir
        dst_js = dst_dir / name
        if dst_js.exists():
            canon_meta, canon_js = json.loads(dst_js.read_text(encoding="utf-8")), dst_js
            others = metas
        else:
            metas.sort(key=lambda mp: (mp[0].get("finished_at") or "", str(mp[1])))
            (canon_meta, src_js), others = metas[0], metas[1:]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_js.with_suffix(".parquet"), dst_js.with_suffix(".parquet"))
            shutil.copy2(src_js, dst_js)                    # json last: marks the unit complete
            canon_js = dst_js
            copied += 1
        canon_rows = pd.read_parquet(canon_js.with_suffix(".parquet"))
        seen_exec = {_execution_key(canon_js)}
        for meta, js in others:
            if meta["scientific_hash"] != canon_meta["scientific_hash"]:
                raise RuntimeError(f"{js}: scientific hash differs from the canonical copy")
            key = _execution_key(js)
            if key in seen_exec:
                same_execution += 1
                continue
            seen_exec.add(key)
            rows = pd.read_parquet(js.with_suffix(".parquet"))
            truth = max(float(np.max(np.abs(rows[c].to_numpy() - canon_rows[c].to_numpy())))
                        for c in ("true_a", "true_b"))
            if not truth <= TRUTH_TOL:
                raise RuntimeError(f"{js}: simulated parameters differ from the canonical copy "
                                   f"(max |diff| {truth:.3e}); not the same scientific dataset")
            diffs = {c: float(np.nanmax(np.abs(rows[c].to_numpy() - canon_rows[c].to_numpy())))
                     for c in COMPARED if c in rows and c in canon_rows}
            tag = key[:12]                                  # deterministic: content digest
            dup_dir = sdir / "duplicates" / cond_dir
            stem = Path(name).stem
            if (dup_dir / f"{stem}__{tag}.json").exists():
                already_logged += 1                         # preserved by an earlier collect
                continue
            dup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(js.with_suffix(".parquet"), dup_dir / f"{stem}__{tag}.parquet")
            shutil.copy2(js, dup_dir / f"{stem}__{tag}.json")
            logged.append({"unit_id": canon_meta["unit_id"], "duplicate_source": str(js),
                           "fit_protocol": [canon_meta.get("fit_protocol"), meta.get("fit_protocol")],
                           "canonical_runner": canon_meta.get("runner"), "duplicate_runner": meta.get("runner"),
                           "truth_max_abs_diff": truth, "max_abs_diff": diffs,
                           "rhat_max": [canon_meta.get("rhat_max"), meta.get("rhat_max")],
                           "ess_bulk_min": [canon_meta.get("ess_bulk_min"), meta.get("ess_bulk_min")],
                           "tier": [canon_meta.get("tier"), meta.get("tier")]})
    if logged:
        with (sdir / "collect_duplicates.jsonl").open("a", encoding="utf-8") as f:
            for rec in logged:
                f.write(json.dumps(rec, default=float) + "\n")
    failures = 0                                           # failure records are preserved too
    for src in sorted(map(str, sources)):
        for fj in sorted(Path(src).rglob("rep_*__*.json")):
            if fj.parent.parent.name != "failures" or fj.parent.parent.parent.name != study_id:
                continue
            if Path(out_root).resolve() in fj.resolve().parents:
                continue
            dst = sdir / "failures" / fj.parent.name / fj.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fj, dst)
                failures += 1
    return {"copied": copied, "same_execution_copies": same_execution,
            "logged_duplicate_executions": len(logged), "previously_logged_duplicates": already_logged,
            "failure_records": failures}


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
            d = run_unit(config, unit, out_root, fit_fn, position=done + 1)
            done += 1
            p = d["fit_protocol"]
            log(f"[{_now()}] {k}/{len(units)} {unit.unit_id}: tier={d['tier']} "
                f"({p['initial_parameterization']} {p['initial_tier']}"
                + (f" -> fallback {p['fallback_tier']}" if p["fallback_triggered"] else "")
                + f"; canonical {p['canonical_parameterization']}) {d['seconds_total']:.1f}s "
                f"rhat={d['rhat_max']} div={d['divergences']}")
        except UnitFailure as exc:                         # sampler/program error: kept, retryable
            sampler_errors.append(unit.unit_id)
            log(f"[{_now()}] {unit.unit_id} FAILED ({exc.kind}): {exc}")
        except Exception as exc:                                   # noqa: BLE001
            crashed += 1
            failures.append(unit.unit_id)
            log(f"[{_now()}] {unit.unit_id} CRASHED: {exc!r}")
    return {"ran": done, "skipped": skipped, "crashed": crashed, "failed_units": failures,
            "sampler_error_units": sampler_errors, "seconds": time.perf_counter() - t0}
