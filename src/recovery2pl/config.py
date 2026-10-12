"""Strict loading and validation of a study configuration (YAML).

Every scientific choice lives in the config: the item generator, the item-heterogeneity worlds,
the student populations, the experimental blocks (which population x worlds x N x test length x
rho, and how many replicates), the estimator's priors and sampler, and the fit-diagnostic rules.
Nothing is defaulted silently: every field is required and unknown fields are rejected.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

TOP_FIELDS = ("study_id", "base_seed", "item_generator", "worlds", "populations", "blocks",
              "estimator", "diagnostics", "reporting", "execution")
GENERATOR_FIELDS = ("kind", "alpha_location", "beta_location", "aux_lower", "aux_upper",
                    "accept_alpha", "accept_beta", "center", "max_tries")
WORLD_FIELDS = ("tau_alpha", "tau_beta")
POPULATION_FIELDS = ("difficulty_shift", "theta")
THETA_FIELDS = {"normal": ("kind",), "skewnormal_standardized": ("kind", "shape")}
BLOCK_FIELDS = ("name", "population", "worlds", "N", "test_length", "rho", "replications")
ESTIMATOR_KINDS = ("hierarchical_2pl_centered_b",      # the study's estimator
                   "hierarchical_2pl_noncentered_b")   # diagnostic variant only
PRIOR_FIELDS = ("mu_log_a_mean", "mu_log_a_sd", "sigma_log_a_scale", "mu_b_mean", "mu_b_sd",
                "sigma_b_scale")
SAMPLER_FIELDS = ("chains", "tune", "draws", "target_accept", "interval")
DIAGNOSTIC_FIELDS = ("rhat_max", "ess_bulk_min", "ess_tail_min", "max_divergences", "gross_rhat")
REPORTING_FIELDS = ("item_error_tolerances", "envelopes")
ENVELOPE_FIELDS = ("name", "populations", "worlds")
EXECUTION_FIELDS = ("chain_cores", "threads_per_process")


class ConfigError(ValueError):
    pass


def _exact(data, fields, where):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: must be a mapping")
    missing = [f for f in fields if f not in data]
    unknown = sorted(str(k) for k in data if k not in fields)
    if missing or unknown:
        raise ConfigError(f"{where}: missing {missing}, unknown {unknown}")


def _num(v, where, lo=None, lo_open=False, integer=False):
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    if integer:
        ok = ok and isinstance(v, int)
    if ok and lo is not None:
        ok = v > lo if lo_open else v >= lo
    if not ok:
        raise ConfigError(f"{where}: invalid value {v!r}")
    return v


def _unique_list(v, where):
    if not isinstance(v, list) or not v or len(set(v)) != len(v):
        raise ConfigError(f"{where} must be a non-empty list without duplicates")
    return v


def _validate_generator(gen):
    _exact(gen, GENERATOR_FIELDS, "item_generator")
    if gen["kind"] != "koenig2022":
        raise ConfigError("item_generator.kind must be 'koenig2022'")
    for f in ("alpha_location", "beta_location"):
        _num(gen[f], f"item_generator.{f}")
    for f in ("aux_lower", "aux_upper", "accept_alpha", "accept_beta"):
        v = gen[f]
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
            raise ConfigError(f"item_generator.{f} must be a list of two numbers")
    if not all(lo < hi for lo, hi in zip(gen["aux_lower"], gen["aux_upper"])):
        raise ConfigError("item_generator.aux_lower must be below aux_upper")
    for f in ("accept_alpha", "accept_beta"):
        if not gen[f][0] < gen[f][1]:
            raise ConfigError(f"item_generator.{f} must be an increasing interval")
    if gen["accept_alpha"][0] < 0:
        raise ConfigError("item_generator.accept_alpha must exclude alpha <= 0 (a > 0)")
    if gen["center"] != "truncated_mean":
        raise ConfigError("item_generator.center must be 'truncated_mean'")
    _num(gen["max_tries"], "item_generator.max_tries", 1, integer=True)


def _validate_population(name, p):
    _exact(p, POPULATION_FIELDS, f"populations.{name}")
    _num(p["difficulty_shift"], f"populations.{name}.difficulty_shift")
    t = p["theta"]
    if not isinstance(t, dict) or t.get("kind") not in THETA_FIELDS:
        raise ConfigError(f"populations.{name}.theta.kind must be one of {sorted(THETA_FIELDS)}")
    _exact(t, THETA_FIELDS[t["kind"]], f"populations.{name}.theta")
    if t["kind"] == "skewnormal_standardized":
        _num(t["shape"], f"populations.{name}.theta.shape")


FALLBACK_RULES = ("noncentered_b_if_divergent_or_gross",)


def validate_estimator(est: dict) -> dict:
    """`fallback` is optional. Absent = a single fit (the protocol of every config before the fallback
    was frozen, whose scientific hashes are therefore unchanged). `noncentered_b_if_divergent_or_gross`
    = the frozen study protocol: centered-b first, non-centered-b refit when the centered fit is
    divergent or gross, canonical result chosen by Policy-D tier only (ties -> centered)."""
    _exact({k: v for k, v in est.items() if k != "fallback"}, ("kind", "priors", "sampler"), "estimator")
    if est["kind"] not in ESTIMATOR_KINDS:
        raise ConfigError(f"estimator.kind must be one of {ESTIMATOR_KINDS}")
    if "fallback" in est:
        if est["fallback"] not in FALLBACK_RULES:
            raise ConfigError(f"estimator.fallback must be one of {FALLBACK_RULES}")
        if est["kind"] != "hierarchical_2pl_centered_b":
            raise ConfigError("estimator.fallback requires kind hierarchical_2pl_centered_b")
    pr = est["priors"]
    _exact(pr, PRIOR_FIELDS, "estimator.priors")
    for f in PRIOR_FIELDS:
        positive = f.endswith(("_sd", "_scale"))
        _num(pr[f], f"estimator.priors.{f}", 0 if positive else None, lo_open=positive)
    s = est["sampler"]
    _exact(s, SAMPLER_FIELDS, "estimator.sampler")
    for f in ("chains", "tune", "draws"):
        _num(s[f], f"estimator.sampler.{f}", 1, integer=True)
    if not 0 < s["target_accept"] < 1 or not 0 < s["interval"] < 1:
        raise ConfigError("estimator.sampler.target_accept and interval must lie in (0, 1)")
    return est


def validate_diagnostics(d: dict) -> dict:
    _exact(d, DIAGNOSTIC_FIELDS, "diagnostics")
    _num(d["rhat_max"], "diagnostics.rhat_max", 1)
    _num(d["gross_rhat"], "diagnostics.gross_rhat", 1)
    if d["gross_rhat"] < d["rhat_max"]:
        raise ConfigError("diagnostics.gross_rhat must be >= diagnostics.rhat_max")
    _num(d["ess_bulk_min"], "diagnostics.ess_bulk_min", 0)
    _num(d["ess_tail_min"], "diagnostics.ess_tail_min", 0)
    _num(d["max_divergences"], "diagnostics.max_divergences", 0, integer=True)
    return d


def validate_config(data: dict) -> dict:
    _exact(data, TOP_FIELDS, "config")
    if not isinstance(data["study_id"], str) or not data["study_id"] or "/" in data["study_id"] \
            or "\\" in data["study_id"]:
        raise ConfigError("study_id must be a non-empty path-safe string")
    _num(data["base_seed"], "base_seed", 0, integer=True)
    _validate_generator(data["item_generator"])

    worlds = data["worlds"]
    if not isinstance(worlds, dict) or not worlds:
        raise ConfigError("worlds must be a non-empty mapping")
    for name, w in worlds.items():
        _exact(w, WORLD_FIELDS, f"worlds.{name}")
        for k in WORLD_FIELDS:
            _num(w[k], f"worlds.{name}.{k}", 0, lo_open=True)

    pops = data["populations"]
    if not isinstance(pops, dict) or not pops:
        raise ConfigError("populations must be a non-empty mapping")
    for name, p in pops.items():
        if not isinstance(name, str) or not name.isalnum():
            raise ConfigError(f"population name {name!r} must be alphanumeric")
        _validate_population(name, p)

    blocks = data["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ConfigError("blocks must be a non-empty list")
    names = set()
    for i, b in enumerate(blocks):
        where = f"blocks[{i}]"
        _exact(b, BLOCK_FIELDS, where)
        if b["name"] in names:
            raise ConfigError(f"{where}: duplicate block name {b['name']!r}")
        names.add(b["name"])
        if b["population"] not in pops:
            raise ConfigError(f"{where}: unknown population {b['population']!r}")
        for w in _unique_list(b["worlds"], f"{where}.worlds"):
            if w not in worlds:
                raise ConfigError(f"{where}: unknown world {w!r}")
        for f in ("N", "test_length"):
            for v in _unique_list(b[f], f"{where}.{f}"):
                _num(v, f"{where}.{f}", 2, integer=True)
        for r in _unique_list(b["rho"], f"{where}.rho"):
            _num(r, f"{where}.rho")
            if not -1 < r < 1:
                raise ConfigError(f"{where}.rho must lie in (-1, 1), got {r}")
        _num(b["replications"], f"{where}.replications", 1, integer=True)

    validate_estimator(data["estimator"])
    validate_diagnostics(data["diagnostics"])

    rep = data["reporting"]
    _exact(rep, REPORTING_FIELDS, "reporting")
    for t in _unique_list(rep["item_error_tolerances"], "reporting.item_error_tolerances"):
        _num(t, "reporting.item_error_tolerances", 0, lo_open=True)
    if not isinstance(rep["envelopes"], list):
        raise ConfigError("reporting.envelopes must be a list")
    env_names = set()
    for i, e in enumerate(rep["envelopes"]):
        where = f"reporting.envelopes[{i}]"
        _exact(e, ENVELOPE_FIELDS, where)
        if not isinstance(e["name"], str) or not e["name"].replace("_", "").isalnum() \
                or e["name"] in env_names:
            raise ConfigError(f"{where}.name must be a unique identifier")
        env_names.add(e["name"])
        for p in _unique_list(e["populations"], f"{where}.populations"):
            if p not in pops:
                raise ConfigError(f"{where}: unknown population {p!r}")
        for w in _unique_list(e["worlds"], f"{where}.worlds"):
            if w not in worlds:
                raise ConfigError(f"{where}: unknown world {w!r}")

    x = data["execution"]
    _exact(x, EXECUTION_FIELDS, "execution")
    _num(x["chain_cores"], "execution.chain_cores", 1, integer=True)
    _num(x["threads_per_process"], "execution.threads_per_process", 1, integer=True)

    # every condition must belong to exactly one block (checked here so a bad config fails early)
    from recovery2pl.units import conditions_by_block
    seen = {}
    for block, conds in conditions_by_block(data).items():
        for c in conds:
            if c.condition_id in seen:
                raise ConfigError(f"condition {c.condition_id} appears in blocks "
                                  f"{seen[c.condition_id]!r} and {block!r}")
            seen[c.condition_id] = block
    return copy.deepcopy(data)


def load_config(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def simulation_hash(config: dict, cond) -> str:
    """Identity of the GENERATED DATASET of a condition (with the replicate index it fixes a unit's
    data): seed base, item generator, world, population and condition id -- nothing about the fit.
    Equal simulation hashes + equal replicate => identical simulated a, b, theta and Y."""
    return _digest({"base_seed": config["base_seed"], "item_generator": config["item_generator"],
                    "world": config["worlds"][cond.world],
                    "population": config["populations"][cond.population],
                    "condition_id": cond.condition_id})


def unit_scientific_hash(config: dict, cond) -> str:
    """Identity of everything that determines ONE unit's numbers: seed base, item generator, the
    condition's world, population definition, estimator (including its fallback protocol) and
    diagnostic rules. Block layout, replication counts, reporting and execution settings are
    excluded, so adding replicates or blocks never invalidates finished units. Configs without
    estimator.fallback keep the hashes they had before the fallback existed."""
    return _digest({"base_seed": config["base_seed"],
                    "item_generator": config["item_generator"],
                    "world": config["worlds"][cond.world],
                    "population": config["populations"][cond.population],
                    "estimator": config["estimator"],
                    "diagnostics": config["diagnostics"],
                    "condition_id": cond.condition_id})
