"""Student-population conditions: S1 (difficulty shift) and S2 (shift + standardized skew-normal)."""
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import skew, skewnorm

from recovery2pl.config import load_config
from recovery2pl.simulate import (Condition, draw_theta, generate_bank, simulate_replicate,
                                  skewnormal_moments)
from recovery2pl.units import resolve

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHIFT = -1.5


def cfg():
    return load_config(CONFIGS / "final_v1.yaml")


class PairedItemBankTests(unittest.TestCase):
    def test_s1_and_s2_banks_are_the_s0_bank_translated_by_minus_1_5(self):
        c = cfg()
        for N, J in ((50, 25), (300, 50), (500, 25)):
            for rep in (0, 1, 49):                      # S2 has 50 replicates (r000-r049)
                s0, s1, s2 = (resolve(c, [f"N{N}_J{J}_typical_rho0.3_full{suffix}__r{rep:03d}"])[0]
                              for suffix in ("", "_S1", "_S2"))
                d0, d1, d2 = (simulate_replicate(c, u.condition, u.replicate) for u in (s0, s1, s2))
                for d in (d1, d2):
                    np.testing.assert_array_equal(d["a"], d0["a"])              # a identical
                    np.testing.assert_array_equal(d["b"], d0["b"] + SHIFT)      # exact translation
                    np.testing.assert_allclose(d["b"] - d0["b"], SHIFT, rtol=0, atol=1e-12)
                self.assertNotEqual(d0["sampler_seed"], d1["sampler_seed"])     # own fit seeds
                self.assertNotEqual(d1["sampler_seed"], d2["sampler_seed"])

    def test_bank_bounds_are_not_reapplied_after_the_shift(self):
        """The shift is a pure translation: a bank whose lowest b sits near the lower acceptance
        bound is shifted past it unchanged (the extreme world is used because its b are widest)."""
        c = cfg()
        gen = c["item_generator"]
        for rep in range(200):
            cond = Condition(50, 50, "extreme", 0.75, 1.5, 0.3, "S1", SHIFT, "normal")
            a, b_bank, _ = generate_bank(c, cond, rep)
            if b_bank.min() + SHIFT < gen["accept_beta"][0]:
                b = simulate_replicate(c, cond, rep)["b"]
                np.testing.assert_array_equal(b, b_bank + SHIFT)
                self.assertLess(b.min(), gen["accept_beta"][0])
                return
        self.fail("no extreme bank came within 1.5 of the lower bound in 200 replicates")

    def test_shifted_banks_are_easier_for_the_students(self):
        c = cfg()
        u0, u1 = resolve(c, ["N500_J50_typical_rho0.3_full__r000", "N500_J50_typical_rho0.3_full_S1__r000"])
        p0 = simulate_replicate(c, u0.condition, 0)["Y"].mean()
        p1 = simulate_replicate(c, u1.condition, 0)["Y"].mean()
        self.assertTrue(0.40 < p0 < 0.60, p0)
        self.assertTrue(0.70 < p1 < 0.85, p1)


class AbilityDistributionTests(unittest.TestCase):
    def test_skewnormal_moments_are_exact(self):
        mean, sd = skewnormal_moments(-4.0)
        m, v, s = skewnorm.stats(-4.0, moments="mvs")
        self.assertAlmostEqual(mean, float(m), places=12)
        self.assertAlmostEqual(sd, float(np.sqrt(v)), places=12)
        self.assertAlmostEqual(float(s), -0.7844, places=3)          # target skewness ~ -0.78

    def test_s2_abilities_are_standardized_skew_normal(self):
        theta = draw_theta(400_000, np.random.default_rng(7), "skewnormal_standardized", -4.0)
        self.assertAlmostEqual(theta.mean(), 0.0, delta=0.005)
        self.assertAlmostEqual(theta.std(), 1.0, delta=0.005)
        self.assertAlmostEqual(skew(theta), -0.784, delta=0.02)

    def test_s0_s1_abilities_are_standard_normal(self):
        theta = draw_theta(400_000, np.random.default_rng(8), "normal")
        self.assertAlmostEqual(theta.mean(), 0.0, delta=0.005)
        self.assertAlmostEqual(theta.std(), 1.0, delta=0.005)
        self.assertAlmostEqual(skew(theta), 0.0, delta=0.02)

    def test_s2_unit_uses_skewed_abilities_deterministically(self):
        c = cfg()
        u = resolve(c, ["N500_J25_typical_rho0.3_full_S2__r003"])[0]
        t1 = simulate_replicate(c, u.condition, 3)["theta"]
        t2 = simulate_replicate(c, u.condition, 3)["theta"]
        np.testing.assert_array_equal(t1, t2)
        self.assertLess(np.median(t1) - np.mean(t1), 0.25)
        self.assertGreater(np.median(t1), np.mean(t1))               # negative skew: median > mean

    def test_unknown_theta_kind_raises(self):
        with self.assertRaises(ValueError):
            draw_theta(5, np.random.default_rng(0), "uniform")


if __name__ == "__main__":
    unittest.main()
