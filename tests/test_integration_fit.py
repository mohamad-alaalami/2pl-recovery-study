"""Real (tiny) MCMC fit, run twice from the same unit. Slow: enabled with RECOVERY2PL_SLOW=1.

Requires a C++ compiler for PyTensor. Simulated inputs must be bit-identical; posterior summaries
must agree within a tight numerical tolerance (MCMC results are not required to be byte-identical
across platforms or compilers)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recovery2pl.config import load_config
from recovery2pl.runner import run_unit, unit_paths
from recovery2pl.units import enumerate_units

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TOL = 1e-8


@unittest.skipUnless(os.environ.get("RECOVERY2PL_SLOW") == "1", "set RECOVERY2PL_SLOW=1 to run")
class TinyFitReproducibilityTests(unittest.TestCase):
    def test_same_unit_twice(self):
        cfg = load_config(CONFIGS / "smoke.yaml")
        unit = enumerate_units(cfg)[0]
        outs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                diag = run_unit(cfg, unit, d)
                outs.append((diag, pd.read_parquet(unit_paths(d, "smoke", unit)[0])))
        (d1, r1), (d2, r2) = outs
        self.assertEqual(d1["backend"], "c")
        self.assertIsNone(d1["error"])
        for col in ("true_a", "true_b"):
            np.testing.assert_array_equal(r1[col], r2[col])
        self.assertEqual(d1["sampler_seed"], d2["sampler_seed"])
        for col in ("est_a", "est_b", "lo_a", "hi_a", "lo_b", "hi_b", "est_log_a"):
            np.testing.assert_allclose(r1[col], r2[col], rtol=0, atol=TOL, err_msg=col)
        for k in ("rhat_max", "ess_bulk_min", "ess_tail_min"):
            self.assertAlmostEqual(d1[k], d2[k], delta=TOL)
        self.assertEqual((d1["divergences"], d1["tier"]), (d2["divergences"], d2["tier"]))


if __name__ == "__main__":
    unittest.main()
