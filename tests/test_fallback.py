"""Frozen fit protocol: centered-b first, non-centered-b fallback for divergent/gross fits, canonical
fit chosen by Policy-D tier only (ties keep centered). Fits are scripted fakes (no MCMC)."""
import copy
import inspect
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recovery2pl.config import load_config, simulation_hash, unit_scientific_hash
from recovery2pl.fallback import (CENTERED, NONCENTERED, RULE, TIER_ORDER, select_canonical,
                                  triggers)
from recovery2pl.metrics import fallback_report, replicate_metrics, summarize
from recovery2pl.runner import UnitFailure, run_unit, run_units, unit_paths, unit_status
from recovery2pl.simulate import simulate_replicate
from recovery2pl.units import enumerate_units

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def fallback_config():
    c = load_config(CONFIGS / "smoke.yaml")
    c["estimator"]["fallback"] = RULE
    return c


def scripted(tiers: dict, quality: dict = None, errors: dict = None, calls: list = None):
    """fit_fn whose result per parameterization is scripted. `quality` sets how far estimates are
    from the truth (0 = perfect) so recovery error can be made to disagree with the tier."""
    quality = quality or {}
    errors = errors or {}

    def fit(Y, config, seed, kind):
        if calls is not None:
            calls.append((kind, seed, Y.copy()))
        if kind in errors:
            return {"error": errors[kind], "tier": "gross", "seconds": 0.1, "backend": "fake"}
        cfg_unit = fit.unit
        sim = simulate_replicate(config, cfg_unit.condition, cfg_unit.replicate)
        off = quality.get(kind, 0.1)
        J = Y.shape[1]
        est_a, est_b = sim["a"] * np.exp(off), sim["b"] + off
        return {"tier": tiers[kind], "error": None, "seconds": 1.0 if kind == CENTERED else 2.0,
                "backend": "fake", "est_a": est_a, "lo_a": est_a - 0.5, "hi_a": est_a + 0.5,
                "sd_a": np.full(J, 0.2), "rhat_a": np.ones(J), "est_b": est_b, "lo_b": est_b - 0.5,
                "hi_b": est_b + 0.5, "sd_b": np.full(J, 0.2), "rhat_b": np.ones(J),
                "est_log_a": np.log(est_a), "rhat_max": 1.0, "ess_bulk_min": 900.0,
                "ess_tail_min": 900.0, "rhat_max_group": 1.0, "ess_bulk_min_group": 900.0,
                "theta_rhat_max": 1.0, "theta_ess_bulk_min": 900.0,
                "divergences": 5 if tiers[kind] == "divergent" else 0, "divergence_share": 0.0,
                "ebfmi_min": 0.6, "treedepth_hits": 0, "hyper": {}}
    return fit


class SelectionRuleTests(unittest.TestCase):
    def test_rule_table(self):
        for c, n in itertools.product(TIER_ORDER, TIER_ORDER):
            if not triggers(c):
                continue
            label, reason = select_canonical(c, n)
            better = TIER_ORDER.index(n) < TIER_ORDER.index(c)
            self.assertEqual(label, "noncentered_b" if better else "centered_b", (c, n))
            self.assertTrue(reason)
        for c in ("clean", "mixing_flag"):
            self.assertFalse(triggers(c))
            self.assertEqual(select_canonical(c, None)[0], "centered_b")
        with self.assertRaises(ValueError):
            select_canonical("divergent", None)
        with self.assertRaises(ValueError):
            select_canonical("bad", "clean")

    def test_selection_can_only_see_two_tiers(self):
        self.assertEqual(list(inspect.signature(select_canonical).parameters),
                         ["initial_tier", "fallback_tier"])


class RunnerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = fallback_config()
        self.unit = enumerate_units(self.cfg)[0]
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_case(self, tiers, quality=None, out=None):
        fit = scripted(tiers, quality)
        fit.unit = self.unit
        out = out or self.out
        d = run_unit(self.cfg, self.unit, out, fit_fn=fit)
        rows = pd.read_parquet(unit_paths(out, self.cfg["study_id"], self.unit)[0])
        return d, rows

    def test_branches(self):
        cases = [  # centered tier, non-centered tier, expect fallback, expected canonical
            ("clean", "clean", False, "centered_b"),
            ("mixing_flag", "clean", False, "centered_b"),
            ("divergent", "clean", True, "noncentered_b"),
            ("gross", "mixing_flag", True, "noncentered_b"),
            ("divergent", "divergent", True, "centered_b"),
            ("gross", "gross", True, "centered_b"),
            ("divergent", "gross", True, "centered_b"),
        ]
        for i, (ct, nt, trig, canon) in enumerate(cases):
            with self.subTest(case=(ct, nt)):
                d, rows = self.run_case({CENTERED: ct, NONCENTERED: nt}, out=self.out / str(i))
                p = d["fit_protocol"]
                self.assertEqual(p["initial_parameterization"], "centered_b")
                self.assertEqual(p["initial_tier"], ct)
                self.assertEqual(p["fallback_triggered"], trig)
                self.assertEqual(p["fallback_parameterization"], "noncentered_b" if trig else None)
                self.assertEqual(p["fallback_tier"], nt if trig else None)
                self.assertEqual(p["canonical_parameterization"], canon)
                self.assertEqual(d["tier"], nt if canon == "noncentered_b" else ct)
                self.assertEqual(p["canonical_tier"], d["tier"])
                self.assertTrue(p["selection_reason"])
                self.assertEqual(set(d["attempts"]), {"centered_b", "noncentered_b"} if trig else {"centered_b"})
                self.assertEqual((rows["canonical_parameterization"] == canon).all(), True)

    def test_choice_ignores_recovery_error(self):
        # non-centered better tier but MUCH worse estimates -> still canonical
        d, _ = self.run_case({CENTERED: "divergent", NONCENTERED: "clean"},
                             {CENTERED: 0.0, NONCENTERED: 2.0}, out=self.out / "a")
        self.assertEqual(d["fit_protocol"]["canonical_parameterization"], "noncentered_b")
        # equal tiers but non-centered perfect -> centered kept (tie)
        d, _ = self.run_case({CENTERED: "divergent", NONCENTERED: "divergent"},
                             {CENTERED: 2.0, NONCENTERED: 0.0}, out=self.out / "b")
        self.assertEqual(d["fit_protocol"]["canonical_parameterization"], "centered_b")

    def test_both_attempts_preserved_and_metrics_use_only_the_canonical_fit(self):
        d, rows = self.run_case({CENTERED: "divergent", NONCENTERED: "clean"},
                                {CENTERED: 0.3, NONCENTERED: 0.05})
        for p in ("centered_b", "noncentered_b"):
            for k in ("est_a", "est_b", "lo_b", "hi_b", "est_log_a"):
                self.assertIn(f"{p}__{k}", rows.columns)
            a = d["attempts"][p]
            for k in ("tier", "seconds", "rhat_max", "ess_bulk_min", "divergences", "runner"):
                self.assertIn(k, a)
        np.testing.assert_array_equal(rows["est_b"], rows["noncentered_b__est_b"])
        self.assertFalse(np.allclose(rows["est_b"], rows["centered_b__est_b"]))
        self.assertAlmostEqual(d["seconds_total"], 3.0)
        rep = replicate_metrics(rows).iloc[0]
        self.assertAlmostEqual(rep["rmse_b"], 0.05, places=10)                  # canonical only
        self.assertEqual(rep["canonical_parameterization"], "noncentered_b")

    def test_simulation_is_done_once_and_both_attempts_get_the_same_data_and_seed(self):
        calls = []
        fit = scripted({CENTERED: "gross", NONCENTERED: "clean"}, calls=calls)
        fit.unit = self.unit
        d = run_unit(self.cfg, self.unit, self.out, fit_fn=fit)
        self.assertEqual([c[0] for c in calls], [CENTERED, NONCENTERED])
        np.testing.assert_array_equal(calls[0][2], calls[1][2])
        self.assertEqual(calls[0][1], calls[1][1])
        sim = simulate_replicate(self.cfg, self.unit.condition, self.unit.replicate)
        np.testing.assert_array_equal(calls[0][2], sim["Y"])
        rows = pd.read_parquet(unit_paths(self.out, self.cfg["study_id"], self.unit)[0])
        np.testing.assert_array_equal(rows["true_a"], sim["a"])
        np.testing.assert_array_equal(rows["true_b"], sim["b"])
        self.assertEqual(d["sampler_seed"], sim["sampler_seed"])

    def test_hash_protocol_and_simulation_identity(self):
        plain = copy.deepcopy(self.cfg)
        plain["estimator"].pop("fallback")
        c = self.unit.condition
        self.assertNotEqual(unit_scientific_hash(self.cfg, c), unit_scientific_hash(plain, c))
        self.assertEqual(simulation_hash(self.cfg, c), simulation_hash(plain, c))
        d, _ = self.run_case({CENTERED: "clean", NONCENTERED: "clean"})
        self.assertEqual(d["scientific_hash"], unit_scientific_hash(self.cfg, c))
        self.assertEqual(d["simulation_hash"], simulation_hash(self.cfg, c))
        with self.assertRaises(RuntimeError):                    # centered-only protocol != this one
            unit_status(plain, self.unit, self.out)

    def test_program_failures_are_not_gross_fits(self):
        for failing in (CENTERED, NONCENTERED):
            out = self.out / failing
            fit = scripted({CENTERED: "divergent", NONCENTERED: "clean"}, errors={failing: "boom"})
            fit.unit = self.unit
            with self.assertRaises(UnitFailure):
                run_unit(self.cfg, self.unit, out, fit_fn=fit)
            pq, js = unit_paths(out, self.cfg["study_id"], self.unit)
            self.assertFalse(pq.exists() or js.exists())                    # nothing canonical written
            self.assertEqual(unit_status(self.cfg, self.unit, out), "pending")   # retryable
            recs = list((out / self.cfg["study_id"] / "failures").rglob("*.json"))
            self.assertEqual(len(recs), 1)
            rec = json.loads(recs[0].read_text())
            self.assertEqual(rec["failed_attempt"], "centered_b" if failing == CENTERED else "noncentered_b")
            self.assertEqual(set(rec["attempts"]), {"centered_b"} if failing == CENTERED
                             else {"centered_b", "noncentered_b"})
        fit = scripted({CENTERED: "divergent", NONCENTERED: "clean"}, errors={CENTERED: "boom"})
        fit.unit = self.unit
        res = run_units(self.cfg, [self.unit], self.out / "batch", fit_fn=fit, log=lambda *_: None)
        self.assertEqual(res["sampler_error_units"], [self.unit.unit_id])
        self.assertEqual(res["ran"], 0)


class FallbackAggregationTests(unittest.TestCase):
    def test_rescue_counts(self):
        cfg = fallback_config()
        cfg["blocks"][0]["replications"] = 5
        units = [u for u in enumerate_units(cfg) if u.condition.population == "S0"]
        script = [("clean", "clean"), ("divergent", "clean"), ("gross", "mixing_flag"),
                  ("divergent", "divergent"), ("mixing_flag", "clean")]
        with tempfile.TemporaryDirectory() as d:
            frames = []
            for u, (ct, nt) in zip(units, script):
                fit = scripted({CENTERED: ct, NONCENTERED: nt})
                fit.unit = u
                run_unit(cfg, u, d, fit_fn=fit)
                frames.append(pd.read_parquet(unit_paths(d, cfg["study_id"], u)[0]))
            rep = replicate_metrics(pd.concat(frames, ignore_index=True))
        fr = fallback_report(rep).iloc[0]
        self.assertEqual((fr["units"], fr["n_fallback_triggered"], fr["n_fallback_rescued"],
                          fr["n_fallback_unresolved"]), (5, 3, 2, 1))
        self.assertEqual((fr["n_canonical_centered"], fr["n_canonical_noncentered"]), (3, 2))
        self.assertAlmostEqual(fr["fallback_trigger_rate"], 0.6)
        self.assertEqual((fr["initial_divergent"], fr["initial_gross"], fr["canonical_clean"],
                          fr["canonical_mixing_flag"], fr["canonical_divergent"]), (2, 1, 2, 2, 1))
        s = summarize(rep).iloc[0]
        self.assertEqual((s["n_fallback_triggered"], s["n_fallback_rescued"]), (3, 2))
        self.assertEqual(s["n_used"], 5)                     # primary: every non-gross canonical fit


if __name__ == "__main__":
    unittest.main()
