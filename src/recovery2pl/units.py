"""Scientific work units: one (condition, replicate) pair = one simulated dataset + one fit.

Unit id: "{condition_id}__r{replicate:03d}". Enumeration is deterministic (config order of blocks,
then test length, world, rho, N, replicate) and a unit's data and seeds depend only on its own
condition and replicate, so any subset of units can run anywhere, in any order, and a single failed
unit can be rerun without changing any other result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recovery2pl.simulate import Condition

SEP = "__r"


@dataclass(frozen=True)
class Unit:
    condition: Condition
    replicate: int
    block: str

    @property
    def unit_id(self) -> str:
        return f"{self.condition.condition_id}{SEP}{self.replicate:03d}"


def _condition(config: dict, block: dict, J: int, world: str, rho: float, N: int) -> Condition:
    w = config["worlds"][world]
    p = config["populations"][block["population"]]
    t = p["theta"]
    return Condition(N=int(N), test_length=int(J), world=world, tau_a=float(w["tau_alpha"]),
                     tau_b=float(w["tau_beta"]), rho=float(rho), population=block["population"],
                     difficulty_shift=float(p["difficulty_shift"]), theta_kind=t["kind"],
                     theta_shape=float(t.get("shape", 0.0)))


def conditions_by_block(config: dict) -> dict:
    """{block name: [Condition, ...]} in deterministic order."""
    out = {}
    for block in config["blocks"]:
        out[block["name"]] = [_condition(config, block, J, world, rho, N)
                              for J in block["test_length"]
                              for world in block["worlds"]
                              for rho in block["rho"]
                              for N in block["N"]]
    return out


def enumerate_units(config: dict, blocks=None, populations=None) -> list:
    """All units of the study (optionally only some blocks / populations), deterministic order."""
    by_block = conditions_by_block(config)
    reps = {b["name"]: b["replications"] for b in config["blocks"]}
    units = []
    for name, conds in by_block.items():
        if blocks is not None and name not in blocks:
            continue
        for c in conds:
            if populations is not None and c.population not in populations:
                continue
            units.extend(Unit(c, r, name) for r in range(reps[name]))
    return units


def shard(units: list, index: int, of: int) -> list:
    """Deterministic round-robin partition: shard `index` of `of` (0-based). Every unit lands in
    exactly one shard."""
    if of < 1 or not 0 <= index < of:
        raise ValueError(f"invalid shard {index} of {of}")
    return [u for k, u in enumerate(units) if k % of == index]


EXTREME_FACTOR_AT_500 = 25.0       # conservative (measured locally: about 10-24x; cloud unmeasured)
FALLBACK_WORST_CASE = 2.0          # a unit may need a second full fit (fallback); assume it does


def expected_fit_seconds(cond: Condition) -> float:
    """One fit's wall time on a 4-vCPU hosted runner (4 chains in parallel), calibrated on the
    measured small/typical-world cloud fits (about 19 s at N*J = 1,250 to 200 s at 25,000). Extreme
    world: multiplied by a factor rising log-linearly in N from 1 at N=50 to EXTREME_FACTOR_AT_500,
    deliberately conservative. Used only for scheduling; never affects results."""
    secs = 10.0 + 0.0076 * cond.N * cond.test_length
    if cond.world == "extreme":
        frac = min(max(np.log(cond.N / 50.0) / np.log(10.0), 0.0), 1.0)
        secs *= EXTREME_FACTOR_AT_500 ** frac
    return float(secs)


def estimated_cost(cond: Condition) -> float:
    """Worst-case seconds for one UNIT: the centered fit plus a full fallback refit (which units
    trigger the fallback is unknown in advance)."""
    return FALLBACK_WORST_CASE * expected_fit_seconds(cond)


def balanced_shards(units: list, of: int, cost=estimated_cost) -> list:
    """Deterministic longest-processing-time-first packing into `of` shards: units sorted by
    (cost desc, unit id), each placed on the currently cheapest shard (ties -> lowest index).
    Returns a list of `of` unit lists; every unit lands in exactly one shard."""
    if of < 1:
        raise ValueError("need at least one shard")
    order = sorted(units, key=lambda u: (-cost(u.condition), u.unit_id))
    loads, shards = [0.0] * of, [[] for _ in range(of)]
    for u in order:
        k = min(range(of), key=lambda i: (loads[i], i))
        shards[k].append(u)
        loads[k] += cost(u.condition)
    return shards


JOB_OVERHEAD_SECONDS = 180.0     # setup, install and one-time C compilation per job (measured ~65 s)
MAX_MATRIX_JOBS = 256            # GitHub Actions matrix limit per workflow run


def plan(units: list, of: int, strategy: str, max_shard_seconds: float = None) -> list:
    """[{label, units: [ids], estimated_cost, worst_case_hours}] for `of` shards (labels
    sNNNofMMM). estimated_cost is the worst-case seconds of the shard's units (every unit assumed to
    need the fallback refit). With `max_shard_seconds`, every shard's worst case plus job overhead
    must fit the budget, otherwise ValueError (use more shards, see shards_for_budget)."""
    if strategy == "round_robin":
        parts = [shard(units, i, of) for i in range(of)]
    elif strategy == "balanced":
        parts = balanced_shards(units, of)
    else:
        raise ValueError(f"unknown strategy {strategy!r} (round_robin | balanced)")
    width = max(3, len(str(of)))
    out = []
    for i, p in enumerate(parts):
        if not p:
            continue
        cost = sum(estimated_cost(u.condition) for u in p)
        worst = cost + JOB_OVERHEAD_SECONDS
        if max_shard_seconds is not None and worst > max_shard_seconds:
            raise ValueError(f"shard {i} worst case {worst / 3600:.2f} h exceeds the budget "
                             f"{max_shard_seconds / 3600:.2f} h; use more shards")
        out.append({"label": f"s{i:0{width}d}of{of:0{width}d}", "units": [u.unit_id for u in p],
                    "estimated_cost": round(cost, 1), "worst_case_hours": round(worst / 3600, 3)})
    return out


def shards_for_budget(units: list, max_shard_seconds: float, minimum: int = 1) -> int:
    """Smallest shard count >= minimum whose balanced plan keeps every shard's worst case (all units
    needing the fallback refit, plus job overhead) within the budget."""
    biggest = max(estimated_cost(u.condition) for u in units) + JOB_OVERHEAD_SECONDS
    if biggest > max_shard_seconds:
        raise ValueError(f"a single unit's worst case ({biggest / 3600:.2f} h) exceeds the budget")
    total = sum(estimated_cost(u.condition) for u in units)
    k = max(minimum, int(np.ceil(total / (max_shard_seconds - JOB_OVERHEAD_SECONDS))))
    while True:
        try:
            plan(units, k, "balanced", max_shard_seconds)
            return k
        except ValueError:
            k += 1


def resolve(config: dict, unit_ids) -> list:
    """Unit objects for explicit unit ids; unknown ids raise (never silently skipped)."""
    lookup = {u.unit_id: u for u in enumerate_units(config)}
    missing = [i for i in unit_ids if i not in lookup]
    if missing:
        raise KeyError(f"unknown unit ids (not in this config): {missing[:5]}"
                       + (f" and {len(missing) - 5} more" if len(missing) > 5 else ""))
    return [lookup[i] for i in unit_ids]


def parse_unit_id(unit_id: str) -> tuple:
    cond_id, _, rep = unit_id.rpartition(SEP)
    if not cond_id or not rep.isdigit():
        raise ValueError(f"malformed unit id {unit_id!r}")
    return cond_id, int(rep)
