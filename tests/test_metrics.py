"""Replicate-level metrics, Policy D subsets, replicate-distribution summaries and envelopes."""
import unittest

import numpy as np
import pandas as pd

from recovery2pl.metrics import (SUBSETS, replicate_metrics, replications_for_mc_se, summarize,
                                 upper_envelope)


def items(cond="c1", N=50, reps=(0, 1), shift_b=(0.2, 0.3), scale_a=1.1, tier="clean",
          population="S0", world="typical"):
    rows = []
    for r, sb in zip(reps, shift_b):
        for j in range(4):
            ta, tb = 1.0 + 0.1 * j, -1.0 + 0.5 * j
            rows.append({"unit_id": f"{cond}__r{r:03d}", "block": "B", "condition_id": cond,
                         "population": population, "difficulty_shift": 0.0, "theta_kind": "normal",
                         "theta_shape": 0.0, "N": N, "test_length": 4, "world": world,
                         "tau_a": 0.25, "tau_b": 0.9, "rho": 0.0, "replicate": r, "tier": tier,
                         "item_id": j, "true_a": ta, "true_b": tb,
                         "est_a": ta * scale_a, "est_b": tb + sb,
                         "lo_a": ta - 1, "hi_a": ta + 1, "lo_b": tb + 0.25, "hi_b": tb + 1.0})
    return pd.DataFrame(rows)


class ReplicateMetricTests(unittest.TestCase):
    def test_formulas(self):
        rep = replicate_metrics(items(), tolerances=(0.2, 0.25)).set_index("replicate")
        self.assertAlmostEqual(rep.loc[0, "rmse_b"], 0.2)
        self.assertAlmostEqual(rep.loc[1, "rmse_b"], 0.3)
        self.assertAlmostEqual(rep.loc[0, "bias_b"], 0.2)
        self.assertAlmostEqual(rep.loc[0, "rmse_log_a"], np.log(1.1))
        self.assertAlmostEqual(rep.loc[0, "bias_log_a"], np.log(1.1))
        self.assertEqual(rep.loc[0, "coverage_a"], 1.0)
        self.assertEqual(rep.loc[0, "coverage_b"], 0.0)              # interval [b+.25, b+1] misses b
        self.assertAlmostEqual(rep.loc[0, "width_b"], 0.75)
        self.assertEqual(rep.loc[0, "within_0p2_b"], 1.0)            # |0.2| <= 0.2
        self.assertEqual(rep.loc[1, "within_0p2_b"], 0.0)
        self.assertEqual(rep.loc[1, "within_0p25_b"], 0.0)
        self.assertAlmostEqual(rep.loc[0, "corr_b"], 1.0)
        self.assertAlmostEqual(rep.loc[0, "slope_b"], 1.0)
        self.assertAlmostEqual(rep.loc[0, "slope_log_a"], 1.0)
        self.assertAlmostEqual(rep.loc[0, "spearman_log_a"], 1.0)

    def test_gross_units_are_kept_without_metrics(self):
        it = items()
        it.loc[it.replicate == 1, "est_b"] = np.nan
        rep = replicate_metrics(it).set_index("replicate")
        self.assertEqual(rep.loc[1, "tier"], "gross")
        self.assertTrue(np.isnan(rep.loc[1, "rmse_b"]))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        frames = [items(reps=(0, 1), shift_b=(0.2, 0.3), tier="clean"),
                  items(reps=(2,), shift_b=(0.5,), tier="mixing_flag"),
                  items(reps=(3,), shift_b=(0.9,), tier="divergent"),
                  items(reps=(4,), shift_b=(5.0,), tier="gross")]
        self.rep = replicate_metrics(pd.concat(frames, ignore_index=True))

    def test_subsets_and_tier_counts(self):
        expect = {"primary": [0.2, 0.3, 0.5, 0.9], "no_divergent": [0.2, 0.3, 0.5],
                  "strict": [0.2, 0.3]}
        for subset, vals in expect.items():
            s = summarize(self.rep, subset).iloc[0]
            self.assertEqual(s["units"], 5)
            self.assertEqual((s["n_clean"], s["n_mixing_flag"], s["n_divergent"], s["n_gross"]),
                             (2, 1, 1, 1))
            self.assertEqual(s["n_used"], len(vals))
            v = np.array(vals)
            self.assertAlmostEqual(s["rmse_b_mean"], v.mean())
            self.assertAlmostEqual(s["rmse_b_sd"], v.std(ddof=1))
            self.assertAlmostEqual(s["rmse_b_mc_se"], v.std(ddof=1) / np.sqrt(len(v)))
            self.assertAlmostEqual(s["rmse_b_median"], np.median(v))
            self.assertAlmostEqual(s["rmse_b_q95"], np.percentile(v, 95))
            self.assertAlmostEqual(s["rmse_b_q05"], np.percentile(v, 5))
            self.assertAlmostEqual(s["rmse_b_pooled"], np.sqrt(np.mean(v ** 2)))
        self.assertEqual(set(SUBSETS), {"primary", "no_divergent", "strict"})
        with self.assertRaises(ValueError):
            summarize(self.rep, "valid_only")

    def test_no_threshold_columns(self):
        cols = summarize(self.rep).columns
        self.assertFalse(any(k in c for c in cols for k in ("trust", "pass", "threshold", "n_star")))


class EnvelopeTests(unittest.TestCase):
    def test_max_over_selected_conditions_per_N(self):
        frames = []
        for cond, pop, world, shifts in (("A", "S0", "small", (0.1, 0.1)), ("B", "S1", "typical", (0.3, 0.3)),
                                         ("C", "S2", "typical", (0.9, 0.9))):
            for N in (50, 100):
                frames.append(items(cond=f"{cond}{N}", N=N, shift_b=tuple(s * 50 / N for s in shifts),
                                    population=pop, world=world))
        summ = summarize(replicate_metrics(pd.concat(frames, ignore_index=True)))
        env = upper_envelope(summ, "rmse_b_mean", populations=["S0", "S1"], worlds=["small", "typical"])
        self.assertEqual(list(env["N"]), [50, 100])
        self.assertEqual(list(env["attained_by"]), ["B50", "B100"])   # the stress condition C is excluded
        self.assertAlmostEqual(env.iloc[0]["rmse_b_mean"], 0.3)
        self.assertEqual(int(env.iloc[0]["n_conditions"]), 2)

    def test_replications_for_target_mc_se(self):
        s = pd.DataFrame({"condition_id": ["c"], "rmse_b_sd": [0.03]})
        self.assertEqual(int(replications_for_mc_se(s, "rmse_b", 0.005)["replications_needed"][0]), 36)


if __name__ == "__main__":
    unittest.main()
