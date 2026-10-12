"""Config validation, unit enumeration and the exact unit counts of the shipped configs."""
import copy
import unittest
from collections import Counter
from pathlib import Path

from recovery2pl.config import ConfigError, load_config, unit_scientific_hash, validate_config
from recovery2pl.units import enumerate_units

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def final():
    return load_config(CONFIGS / "final_v1.yaml")


class ConfigValidationTests(unittest.TestCase):
    def test_shipped_configs_are_valid(self):
        for name in ("final_v1.yaml", "precheck_s1s2.yaml", "smoke.yaml",
                     "diagnostic_s1_small_lowN_noncentered_b.yaml"):
            load_config(CONFIGS / name)

    def test_rejections(self):
        mutations = [
            lambda d: d.pop("base_seed"),
            lambda d: d.update(extra=1),
            lambda d: d["blocks"][0]["N"].append(50),                          # duplicate N
            lambda d: d["blocks"][0]["rho"].append(1.0),                       # rho out of range
            lambda d: d["blocks"][0].update(population="S9"),                  # unknown population
            lambda d: d["blocks"][0]["worlds"].append("huge"),                 # unknown world
            lambda d: d["blocks"][1].update(name="S0_main"),                   # duplicate block name
            lambda d: d["blocks"].append(dict(d["blocks"][0], name="dup")),    # duplicate conditions
            lambda d: d["worlds"]["small"].update(tau_beta=0),
            lambda d: d["populations"]["S2"]["theta"].update(kind="mystery"),
            lambda d: d["populations"]["S2"]["theta"].pop("shape"),
            lambda d: d["estimator"].update(kind="something_else"),
            lambda d: d["estimator"]["priors"].update(sigma_b_scale=-1),
            lambda d: d["estimator"]["sampler"].update(interval=1.2),
            lambda d: d["diagnostics"].update(gross_rhat=1.0),                 # below rhat_max
            lambda d: d["diagnostics"].update(max_divergences=-1),
            lambda d: d["reporting"]["envelopes"][0].update(populations=["S7"]),
            lambda d: d["execution"].update(chain_cores=0),
            lambda d: d["item_generator"].update(center="mean"),
            lambda d: d["item_generator"].update(accept_alpha=[-0.1, 4.0]),
        ]
        for i, mutate in enumerate(mutations):
            d = copy.deepcopy(final())
            mutate(d)
            with self.assertRaises(ConfigError, msg=f"mutation {i}"):
                validate_config(d)


class UnitCountTests(unittest.TestCase):
    def test_final_design_has_9660_units(self):
        units = enumerate_units(final())
        self.assertEqual(len(units), 9660)
        by_block = Counter(u.block for u in units)
        self.assertEqual(by_block, {"S0_main": 5600, "S0_extreme": 560, "S1": 2800, "S2": 700})
        self.assertEqual(len({u.unit_id for u in units}), 9660)            # ids unique

    def test_final_design_cells(self):
        conds = {u.condition.condition_id: u.condition for u in enumerate_units(final())}
        self.assertEqual(len(conds), 56 + 28 + 28 + 14)
        self.assertEqual(sorted({c.N for c in conds.values()}), [50, 75, 100, 150, 200, 300, 500])
        s1 = [c for c in conds.values() if c.population == "S1"]
        s2 = [c for c in conds.values() if c.population == "S2"]
        self.assertEqual({(c.world, c.rho) for c in s1}, {("small", 0.3), ("typical", 0.3)})
        self.assertEqual({(c.world, c.rho) for c in s2}, {("typical", 0.3)})
        self.assertEqual({c.test_length for c in s1 + s2}, {25, 50})

    def test_precheck_has_84_units_identical_in_science_to_final(self):
        pre = load_config(CONFIGS / "precheck_s1s2.yaml")
        units = enumerate_units(pre)
        self.assertEqual(len(units), 84)
        self.assertEqual(len({u.condition.condition_id for u in units}), 42)
        fin = final()
        fin_units = {u.unit_id: u for u in enumerate_units(fin)}
        for u in units:                                     # same unit id and same scientific hash
            self.assertIn(u.unit_id, fin_units)
            self.assertEqual(unit_scientific_hash(pre, u.condition),
                             unit_scientific_hash(fin, fin_units[u.unit_id].condition))

    def test_noncentered_diagnostic_reuses_the_canonical_datasets_but_never_their_hash(self):
        import numpy as np
        from recovery2pl.simulate import simulate_replicate
        diag = load_config(CONFIGS / "diagnostic_s1_small_lowN_noncentered_b.yaml")
        pre = load_config(CONFIGS / "precheck_s1s2.yaml")
        du = enumerate_units(diag)
        self.assertEqual(len(du), 12)
        pu = {u.unit_id: u for u in enumerate_units(pre)}
        fu = {u.unit_id for u in enumerate_units(final())}
        for u in du:
            self.assertEqual((u.condition.population, u.condition.world, u.condition.rho), ("S1", "small", 0.3))
            self.assertIn(u.unit_id, pu)
            self.assertIn(u.unit_id, fu)
            a = simulate_replicate(diag, u.condition, u.replicate)
            b = simulate_replicate(pre, pu[u.unit_id].condition, u.replicate)
            for k in ("a", "b", "theta", "Y"):
                np.testing.assert_array_equal(a[k], b[k])                   # identical dataset
            self.assertEqual(a["sampler_seed"], b["sampler_seed"])
            self.assertNotEqual(unit_scientific_hash(diag, u.condition),     # never mergeable
                                unit_scientific_hash(pre, pu[u.unit_id].condition))
        self.assertEqual({k: v for k, v in diag["estimator"].items() if k != "kind"},
                         {k: v for k, v in pre["estimator"].items() if k != "kind"})

    def test_hash_ignores_layout_but_not_science(self):
        c = final()
        u = enumerate_units(c)[0]
        d = copy.deepcopy(c)
        d["blocks"][0]["replications"] = 7
        d["execution"]["chain_cores"] = 1
        d["study_id"] = "other"
        self.assertEqual(unit_scientific_hash(c, u.condition), unit_scientific_hash(d, u.condition))
        d["estimator"]["sampler"]["draws"] = 999
        self.assertNotEqual(unit_scientific_hash(c, u.condition), unit_scientific_hash(d, u.condition))


if __name__ == "__main__":
    unittest.main()
