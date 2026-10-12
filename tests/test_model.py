"""Model structure (no sampling) and Policy D fit tiers."""
import unittest
from pathlib import Path

import numpy as np

from recovery2pl.config import load_config
from recovery2pl.model import TIERS, build_model, classify_fit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RULES = {"rhat_max": 1.01, "ess_bulk_min": 400, "ess_tail_min": 400, "max_divergences": 0,
         "gross_rhat": 1.1}


class ModelStructureTests(unittest.TestCase):
    def test_centered_difficulty_non_centered_log_discrimination(self):
        import pymc as pm
        pri = load_config(CONFIGS / "final_v1.yaml")["estimator"]["priors"]
        Y = np.random.default_rng(0).integers(0, 2, size=(20, 5)).astype("int8")
        m = build_model(Y, pri)
        free = {v.name for v in m.free_RVs}
        self.assertEqual(free, {"mu_log_a", "sigma_log_a", "mu_b", "sigma_b", "z_a", "b", "theta"})
        self.assertIn("a", {v.name for v in m.deterministics})
        self.assertTrue(np.isfinite(m.compile_logp()(m.initial_point())))
        a = pm.draw(m["a"], draws=500, random_seed=1)
        self.assertTrue(np.all(a > 0))
        b = pm.draw(m["b"], draws=20000, random_seed=1)               # prior predictive of b
        self.assertAlmostEqual(b.mean(), 0.0, delta=0.05)


class NonCenteredDiagnosticVariantTests(unittest.TestCase):
    def test_only_the_difficulty_parameterization_changes(self):
        import pymc as pm
        from recovery2pl.model import NONCENTERED
        pri = load_config(CONFIGS / "final_v1.yaml")["estimator"]["priors"]
        Y = np.random.default_rng(0).integers(0, 2, size=(20, 5)).astype("int8")
        mc, mn = build_model(Y, pri), build_model(Y, pri, NONCENTERED)
        fc, fn = {v.name for v in mc.free_RVs}, {v.name for v in mn.free_RVs}
        self.assertEqual(fc ^ fn, {"b", "z_b"})                       # b <-> z_b swapped, nothing else
        self.assertIn("z_a", fn)                                       # log a unchanged (non-centered)
        bc = pm.draw(mc["b"], draws=20000, random_seed=1)              # same prior on b either way
        bn = pm.draw(mn["b"], draws=20000, random_seed=1)
        self.assertAlmostEqual(bc.mean(), bn.mean(), delta=0.03)
        self.assertAlmostEqual(bc.std(), bn.std(), delta=0.03)
        with self.assertRaises(ValueError):
            build_model(Y, pri, "half_centered")


def fit(**kw):
    d = {"error": None, "est_a": np.ones(3), "est_b": np.zeros(3), "rhat_max": 1.0,
         "ess_bulk_min": 1000.0, "ess_tail_min": 1000.0, "divergences": 0}
    d.update(kw)
    return d


class PolicyDTierTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(classify_fit(fit(), RULES), "clean")
        self.assertEqual(classify_fit(fit(rhat_max=1.02), RULES), "mixing_flag")
        self.assertEqual(classify_fit(fit(ess_bulk_min=399.0), RULES), "mixing_flag")
        self.assertEqual(classify_fit(fit(ess_tail_min=10.0), RULES), "mixing_flag")
        self.assertEqual(classify_fit(fit(divergences=1), RULES), "divergent")
        self.assertEqual(classify_fit(fit(divergences=3, rhat_max=1.05), RULES), "divergent")
        self.assertEqual(classify_fit(fit(rhat_max=1.11), RULES), "gross")
        self.assertEqual(classify_fit(fit(rhat_max=1.11, divergences=5), RULES), "gross")
        self.assertEqual(classify_fit(fit(error="ValueError: boom"), RULES), "gross")
        self.assertEqual(classify_fit(fit(est_a=np.array([1.0, np.nan, 1.0])), RULES), "gross")
        self.assertEqual(classify_fit({"error": None}, RULES), "gross")
        self.assertEqual(classify_fit(fit(rhat_max=float("nan")), RULES), "gross")
        self.assertEqual(TIERS, ("clean", "mixing_flag", "divergent", "gross"))

    def test_boundaries_are_inclusive_as_documented(self):
        self.assertEqual(classify_fit(fit(rhat_max=1.01, ess_bulk_min=400.0, ess_tail_min=400.0), RULES),
                         "clean")
        self.assertEqual(classify_fit(fit(rhat_max=1.1), RULES), "mixing_flag")


if __name__ == "__main__":
    unittest.main()
