"""Frozen fit protocol: centered-b first, automatic non-centered-b fallback (a computational
sampling fallback for the SAME statistical model and posterior; not a different estimator).

    1. Fit the centered-b model; classify it with Policy D.
    2. clean or mixing_flag  -> stop; centered is canonical.
    3. divergent or gross    -> refit the SAME dataset with non-centered b (same priors,
                                likelihood, identification, sampler settings and seed).
    4. The canonical fit is chosen by Policy-D tier ONLY:
           clean > mixing_flag > divergent > gross
       non-centered becomes canonical only if its tier is strictly better; ties keep centered.

Nothing else may enter the choice (no divergence counts, R-hat, ESS, E-BFMI, runtime, recovery
error, true parameters or condition labels): `select_canonical` sees two tier strings and nothing
else, so the rule is usable when true parameters are unknown.

Validation (pre-check, 18 problematic centered units paired with non-centered fits): 13 improved in
tier, 5 unchanged, 0 worse; divergences 803 -> 12; every gross fit rescued; RMSE(b) changed by at
most 0.030 (mean about -0.0006) and RMSE(log a) by at most 0.016.
"""
from __future__ import annotations

RULE = "noncentered_b_if_divergent_or_gross"
TIER_ORDER = ("clean", "mixing_flag", "divergent", "gross")      # best -> worst
TRIGGER_TIERS = ("divergent", "gross")
CENTERED, NONCENTERED = "hierarchical_2pl_centered_b", "hierarchical_2pl_noncentered_b"
LABEL = {CENTERED: "centered_b", NONCENTERED: "noncentered_b"}


def _rank(tier: str) -> int:
    if tier not in TIER_ORDER:
        raise ValueError(f"unknown Policy-D tier {tier!r}")
    return TIER_ORDER.index(tier)


def triggers(initial_tier: str) -> bool:
    """Does a centered fit with this tier trigger the non-centered refit?"""
    _rank(initial_tier)
    return initial_tier in TRIGGER_TIERS


def select_canonical(initial_tier: str, fallback_tier: str | None) -> tuple:
    """(canonical label, reason) from the two Policy-D tiers only."""
    if fallback_tier is None:
        if triggers(initial_tier):
            raise ValueError("a triggering initial tier needs a fallback tier")
        return LABEL[CENTERED], f"centered tier {initial_tier}: fallback not triggered"
    if _rank(fallback_tier) < _rank(initial_tier):
        return LABEL[NONCENTERED], (f"fallback tier {fallback_tier} strictly better than "
                                     f"centered tier {initial_tier}")
    return LABEL[CENTERED], (f"fallback tier {fallback_tier} not strictly better than centered "
                             f"tier {initial_tier}: tie or worse keeps centered")


def plan_for(config: dict) -> tuple:
    """(initial estimator kind, fallback enabled?) for a config."""
    est = config["estimator"]
    return est["kind"], est.get("fallback") == RULE
