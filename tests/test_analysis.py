"""Final-analysis logic: condition roles, the main envelope (max over main conditions only, per
subset, with its driving condition), subset sensitivity, S2-S1 pairing and the error decomposition."""
import unittest

import numpy as np
import pandas as pd

from recovery2pl.analysis import (condition_tables, envelope, envelope_sensitivity, error_decomposition,
                                  lookup_table, paired_s2_vs_s1, role)
from recovery2pl.metrics import replicate_metrics


def items(population, world, N, shifts, tiers=None, rho=0.3, J=4):
    cond = f"N{N}_J{J}_{world}_rho{rho:g}_full" + ("" if population == "S0" else f"_{population}")
    tiers = tiers or ["clean"] * len(shifts)
    rows = []
    for r, (sb, tier) in enumerate(zip(shifts, tiers)):
        for j in range(J):
            ta, tb = 1.0 + 0.1 * j, -1.0 + 0.5 * j
            sb_j = sb + (0.05 if j % 2 else -0.05)          # not a pure common shift
            rows.append({"unit_id": f"{cond}__r{r:03d}", "block": population, "condition_id": cond,
                         "population": population, "difficulty_shift": 0.0, "theta_kind": "normal",
                         "theta_shape": 0.0, "N": N, "test_length": J, "world": world, "tau_a": 0.25,
                         "tau_b": 0.9, "rho": rho, "replicate": r, "tier": tier, "initial_tier": tier,
                         "fallback_triggered": tier == "divergent", "fallback_tier": "",
                         "canonical_parameterization": "centered_b", "item_id": j, "true_a": ta,
                         "true_b": tb, "est_a": ta * 1.1, "est_b": tb + sb_j, "lo_a": ta - 1,
                         "hi_a": ta + 1, "lo_b": tb - 1, "hi_b": tb + 1})
    return pd.DataFrame(rows)


class RoleTests(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(role("S0", "small"), "main")
        self.assertEqual(role("S1", "typical"), "main")
        self.assertEqual(role("S2", "typical"), "stress_S2")
        self.assertEqual(role("S0", "extreme"), "stress_extreme")


class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        frames = [items("S0", "small", 50, [0.1, 0.2]),
                  items("S1", "typical", 50, [0.3, 0.4, 2.0], tiers=["clean", "clean", "divergent"]),
                  items("S2", "typical", 50, [5.0, 5.0]),                 # stress: larger, must not enter
                  items("S0", "extreme", 50, [9.0, 9.0], rho=0.0),        # stress: larger, must not enter
                  items("S0", "small", 100, [0.1, 0.1]),
                  items("S1", "typical", 100, [0.05, 0.05])]
        self.it = pd.concat(frames, ignore_index=True)
        self.rep = replicate_metrics(self.it)
        self.tables = condition_tables(self.rep)

    def test_envelope_is_max_over_main_conditions_only(self):
        e = envelope(self.tables["primary"], "rmse_b").set_index("N")
        p = self.tables["primary"]
        main = p[p.role == "main"]
        for N in (50, 100):
            self.assertAlmostEqual(e.loc[N, "value"], main[main.N == N]["rmse_b_mean"].max())
            self.assertLess(e.loc[N, "value"], 1.0)                       # stress values (~5, ~9) excluded
        self.assertTrue(e.loc[50, "driving_condition"].startswith("S1 typical"))
        self.assertTrue(e.loc[100, "driving_condition"].startswith("S0 small"))   # driver may change with N
        self.assertEqual(int(e.loc[50, "n_main_conditions"]), 2)

    def test_sensitivity_and_lookup(self):
        sens = envelope_sensitivity(self.tables)
        b50 = sens[(sens.metric == "rmse_b") & (sens.N == 50)].set_index("subset")
        self.assertEqual(b50.loc["primary", "abs_diff_vs_primary"], 0.0)
        self.assertLess(b50.loc["no_divergent", "value"], b50.loc["primary", "value"])   # divergent rep dropped
        self.assertEqual(int(b50.loc["primary", "n_used"]), 3)
        self.assertEqual(int(b50.loc["no_divergent", "n_used"]), 2)
        self.assertAlmostEqual(b50.loc["no_divergent", "rel_diff_vs_primary"],
                               (b50.loc["no_divergent", "value"] - b50.loc["primary", "value"]) / b50.loc["primary", "value"])
        look = lookup_table(sens).set_index("N")
        self.assertAlmostEqual(look.loc[50, "b_primary"], b50.loc["primary", "value"])
        self.assertIn("b_driving_condition", look.columns)

    def test_s2_pairing_by_test_length_n_replicate(self):
        s2 = paired_s2_vs_s1(self.rep).set_index("N")
        self.assertEqual(int(s2.loc[50, "pairs"]), 2)                     # S1 rep 2 has no S2 partner
        self.assertGreater(s2.loc[50, "rmse_b_diff_mean"], 4)

    def test_error_decomposition(self):
        d = error_decomposition(self.it)
        r = d[(d.population == "S0") & (d.world == "small") & (d.N == 100)].iloc[0]
        # error = 0.1 +- 0.05: shift^2 / mse = 0.01 / (0.01 + 0.0025)
        self.assertAlmostEqual(r.shift_share_b, 0.01 / 0.0125)
        self.assertAlmostEqual(r.shift_share_log_a, 1.0)                  # est_a = 1.1 * true_a for all items


if __name__ == "__main__":
    unittest.main()
