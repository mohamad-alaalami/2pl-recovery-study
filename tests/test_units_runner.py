"""Work-unit ids, sharding, and the runner's atomic, idempotent, order-independent outputs.
The fit is replaced by a deterministic fake, so no MCMC runs here."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recovery2pl.cli import cmd_aggregate
from recovery2pl.config import load_config
from recovery2pl.runner import run_unit, run_units, unit_paths, unit_status
from recovery2pl.units import enumerate_units, parse_unit_id, resolve, shard

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def fake_fit(Y, config, seed, kind="hierarchical_2pl_centered_b"):
    """Deterministic stand-in for the sampler: estimates are a function of Y and the seed only."""
    rng = np.random.default_rng(seed)
    J = Y.shape[1]
    p = (Y.mean(axis=0) + 0.5) / 2
    est_b = -np.log(p / (1 - p)) + rng.normal(0, 0.01, J)
    est_a = np.exp(rng.normal(0, 0.05, J))
    return {"tier": "clean", "error": None, "seconds": 0.0, "backend": "fake",
            "est_a": est_a, "lo_a": est_a - 0.5, "hi_a": est_a + 0.5, "sd_a": np.full(J, 0.2),
            "est_b": est_b, "lo_b": est_b - 0.5, "hi_b": est_b + 0.5, "sd_b": np.full(J, 0.2),
            "rhat_a": np.ones(J), "rhat_b": np.ones(J), "est_log_a": np.log(est_a),
            "rhat_max": 1.0, "ess_bulk_min": 1000.0, "ess_tail_min": 1000.0, "rhat_max_group": 1.0,
            "ess_bulk_min_group": 1000.0, "theta_rhat_max": 1.0, "theta_ess_bulk_min": 1000.0,
            "divergences": 0, "divergence_share": 0.0, "ebfmi_min": 1.0, "treedepth_hits": 0,
            "hyper": {"mu_b": 0.0}}


class UnitTests(unittest.TestCase):
    def test_unit_id_roundtrip_and_resolution(self):
        cfg = load_config(CONFIGS / "final_v1.yaml")
        units = enumerate_units(cfg)
        for u in (units[0], units[4321], units[-1]):
            cid, rep = parse_unit_id(u.unit_id)
            self.assertEqual((cid, rep), (u.condition.condition_id, u.replicate))
            self.assertEqual(resolve(cfg, [u.unit_id])[0], u)
        with self.assertRaises(KeyError):
            resolve(cfg, ["N51_J25_typical_rho0.3_full__r000"])
        with self.assertRaises(ValueError):
            parse_unit_id("no-separator")

    def test_enumeration_is_deterministic(self):
        cfg = load_config(CONFIGS / "final_v1.yaml")
        self.assertEqual([u.unit_id for u in enumerate_units(cfg)],
                         [u.unit_id for u in enumerate_units(copy.deepcopy(cfg))])

    def test_shards_partition_every_unit_exactly_once(self):
        cfg = load_config(CONFIGS / "final_v1.yaml")
        units = [u.unit_id for u in enumerate_units(cfg)]
        for k in (1, 7, 64):
            parts = [[u.unit_id for u in shard(enumerate_units(cfg), i, k)] for i in range(k)]
            flat = [x for p in parts for x in p]
            self.assertEqual(sorted(flat), sorted(units))
            self.assertEqual(len(flat), len(set(flat)))
        with self.assertRaises(ValueError):
            shard(units, 3, 3)


class PlanTests(unittest.TestCase):
    def test_balanced_plan_partitions_deterministically_and_balances_cost(self):
        from recovery2pl.units import estimated_cost, plan
        for cfg_name, k in (("final_v1.yaml", 64), ("precheck_s1s2.yaml", 6)):
            cfg = load_config(CONFIGS / cfg_name)
            units = enumerate_units(cfg)
            p1, p2 = plan(units, k, "balanced"), plan(enumerate_units(cfg), k, "balanced")
            self.assertEqual(p1, p2)                                       # deterministic
            flat = [x for s in p1 for x in s["units"]]
            self.assertEqual(sorted(flat), sorted(u.unit_id for u in units))
            self.assertEqual(len(flat), len(set(flat)))
            costs = [s["estimated_cost"] for s in p1]
            biggest = max(estimated_cost(u.condition) for u in units)
            self.assertLessEqual(max(costs) - min(costs), biggest + 1e-6)  # LPT guarantee
            rr = plan(units, k, "round_robin")
            rr_costs = [s["estimated_cost"] for s in rr]
            self.assertLessEqual(max(costs), max(rr_costs) + 1e-6)
        with self.assertRaises(ValueError):
            plan(units, 2, "random")

    def test_precheck_plan_covers_exactly_the_84_units(self):
        from recovery2pl.units import plan
        cfg = load_config(CONFIGS / "precheck_s1s2.yaml")
        p = plan(enumerate_units(cfg), 6, "balanced")
        self.assertEqual(sum(len(s["units"]) for s in p), 84)
        self.assertEqual([s["label"] for s in p], [f"s{i:03d}of006" for i in range(6)])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIGS / "smoke.yaml")
        self.units = enumerate_units(self.cfg)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_outputs_idempotence_and_rerun_of_one_unit(self):
        res = run_units(self.cfg, self.units, self.out, fit_fn=fake_fit, log=lambda *_: None)
        self.assertEqual((res["ran"], res["skipped"], res["crashed"]), (3, 0, 0))
        pq, js = unit_paths(self.out, "smoke", self.units[1])
        self.assertTrue(pq.exists() and js.exists())
        self.assertFalse(list(self.out.rglob("*.tmp")))
        diag = json.loads(js.read_text())
        self.assertEqual(diag["unit_id"], self.units[1].unit_id)
        rows = pd.read_parquet(pq)
        self.assertEqual(len(rows), self.units[1].condition.test_length)
        self.assertEqual(set(rows["population"]), {"S1"})

        # second run: everything skipped
        res = run_units(self.cfg, self.units, self.out, fit_fn=fake_fit, log=lambda *_: None)
        self.assertEqual((res["ran"], res["skipped"]), (0, 3))

        # delete one unit, rerun it alone: identical file content, others untouched
        before = {u.unit_id: pd.read_parquet(unit_paths(self.out, "smoke", u)[0]) for u in self.units}
        mtime_other = unit_paths(self.out, "smoke", self.units[0])[1].stat().st_mtime_ns
        pq.unlink()
        js.unlink()
        run_units(self.cfg, [self.units[1]], self.out, fit_fn=fake_fit, log=lambda *_: None)
        pd.testing.assert_frame_equal(pd.read_parquet(pq), before[self.units[1].unit_id])
        self.assertEqual(unit_paths(self.out, "smoke", self.units[0])[1].stat().st_mtime_ns, mtime_other)

    def test_hash_mismatch_is_never_overwritten(self):
        run_unit(self.cfg, self.units[0], self.out, fit_fn=fake_fit)
        changed = copy.deepcopy(self.cfg)
        changed["estimator"]["sampler"]["draws"] = 999
        with self.assertRaises(RuntimeError):
            unit_status(changed, self.units[0], self.out)

    def test_collect_keeps_one_canonical_result_and_logs_duplicate_executions(self):
        import shutil
        from recovery2pl.runner import collect

        def other_cpu_fit(Y, config, seed, kind="hierarchical_2pl_centered_b"):   # other CPU: different posterior draws
            d = fake_fit(Y, config, seed, kind)
            d["est_b"] = d["est_b"] + 0.05
            return d

        a, b, c = (self.out / n for n in ("artifact_a", "artifact_b", "artifact_c"))
        run_units(self.cfg, self.units[:2], a, fit_fn=fake_fit, log=lambda *_: None)
        run_units(self.cfg, self.units[1:], b, fit_fn=other_cpu_fit, log=lambda *_: None)  # unit 1 twice
        shutil.copytree(a, c)                        # a byte-identical copy of the same execution
        merged = self.out / "merged"
        res = collect([a, b, c], merged, "smoke")
        self.assertEqual(res, {"copied": 3, "same_execution_copies": 2,
                               "logged_duplicate_executions": 1, "previously_logged_duplicates": 0,
                               "failure_records": 0})
        sdir = merged / "smoke"
        self.assertEqual(len(list((sdir / "units").rglob("rep_*.json"))), 3)
        # canonical = earliest finished (ties: first source path) -> artifact_a; never averaged
        u1 = self.units[1]
        canon = pd.read_parquet(unit_paths(merged, "smoke", u1)[0])
        pd.testing.assert_frame_equal(canon, pd.read_parquet(unit_paths(a, "smoke", u1)[0]))
        log = [json.loads(x) for x in (sdir / "collect_duplicates.jsonl").read_text().splitlines()]
        self.assertEqual(len(log), 1)
        self.assertAlmostEqual(log[0]["max_abs_diff"]["est_b"], 0.05, places=9)
        self.assertEqual(log[0]["truth_max_abs_diff"], 0.0)
        self.assertEqual(len(list((sdir / "duplicates").rglob("*.json"))), 1)
        # collecting again changes nothing and does not re-log
        res2 = collect([a, b, c], merged, "smoke")
        self.assertEqual((res2["copied"], res2["logged_duplicate_executions"],
                          res2["previously_logged_duplicates"]), (0, 0, 1))
        pd.testing.assert_frame_equal(pd.read_parquet(unit_paths(merged, "smoke", u1)[0]), canon)

    def test_collect_rejects_different_simulated_data_or_hash(self):
        from recovery2pl.runner import collect
        a, b = self.out / "artifact_a", self.out / "artifact_b"
        run_units(self.cfg, self.units[:1], a, fit_fn=fake_fit, log=lambda *_: None)
        run_units(self.cfg, self.units[:1], b, fit_fn=fake_fit, log=lambda *_: None)
        pq, js = unit_paths(b, "smoke", self.units[0])
        rows = pd.read_parquet(pq)
        rows["true_b"] += 1e-6
        rows.to_parquet(pq, index=False)
        with self.assertRaises(RuntimeError):
            collect([a, b], self.out / "merged1", "smoke")
        d = json.loads(js.read_text())
        d["scientific_hash"] = "0" * 16
        js.write_text(json.dumps(d))
        with self.assertRaises(RuntimeError):
            collect([a, b], self.out / "merged2", "smoke")

    def test_unit_json_records_runner_metadata(self):
        d = run_unit(self.cfg, self.units[0], self.out, fit_fn=fake_fit, position=1)
        self.assertEqual(d["position_in_batch"], 1)
        self.assertIn("cpu_model", d["runner"])
        self.assertTrue(d["runner"]["cpu_model"])

    def test_aggregate_writes_replicate_metrics_summaries_and_envelopes(self):
        run_units(self.cfg, self.units, self.out, fit_fn=fake_fit, log=lambda *_: None)
        res = cmd_aggregate(self.cfg, self.out)
        self.assertEqual(res["units"], 3)
        sdir = self.out / "smoke"
        for name in ("replicate_metrics.csv", "summary_primary.csv", "summary_no_divergent.csv",
                     "summary_strict.csv", "envelope_main_rmse_b.csv", "fit_diagnostics.csv"):
            self.assertTrue((sdir / name).exists(), name)
        rep = pd.read_csv(sdir / "replicate_metrics.csv")
        self.assertEqual(len(rep), 3)
        env = pd.read_csv(sdir / "envelope_main_rmse_b.csv")
        self.assertEqual(int(env["n_conditions"].iloc[0]), 2)          # S0 + S1, S2 excluded


if __name__ == "__main__":
    unittest.main()
