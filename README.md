# 2PL parameter-recovery study

How does recovery error of 2PL item **difficulty (b)** and **discrimination (a)** change as the
number of responses per item **N** grows?

This is a Monte Carlo parameter-recovery study:
- true item parameters (a, b) and student abilities θ are simulated;
- a hierarchical Bayesian 2PL is fitted to the **responses only**;
- the estimates are compared with the generating values.

The output is a set of error-vs-N curves with Monte Carlo uncertainty, one per experimental
condition. The code contains **no** "trustworthy" cutoff: what error is acceptable is left to the
user of the curves.

## Design

| Factor | Levels |
|---|---|
| Responses per item N (full exposure: every student answers every item) | 50, 75, 100, 150, 200, 300, 500 |
| Test length J | 25, 50 |
| Item heterogeneity (SD of raw α, SD of β) | small (0.10, 0.40); typical (0.25, 0.90); extreme (0.75, 1.50) |
| corr(α, β) ρ | 0, 0.3 |

Student populations, applied after the item bank is generated:

| Population | Abilities θ | Item-difficulty shift |
|---|---|---|
| S0 reference | N(0, 1) | 0 |
| S1 easy bank (off target) | N(0, 1) | −1.5 θ-SD (pure translation of the S0 bank) |
| S2 easy bank + skewed abilities | skew-normal, shape −4, standardized to mean 0, SD 1 (skewness ≈ −0.78) | −1.5 |

Blocks in `configs/final_v1.yaml`:

| Block | Conditions | Replicates | Units |
|---|---|---|---|
| S0, small + typical | 2 worlds × 7 N × 2 J × 2 ρ | 100 | 5,600 |
| S0, extreme (stress) | 1 × 7 × 2 × 2 | 20 | 560 |
| S1, small + typical | 2 × 7 × 2 × ρ = 0.3 | 100 | 2,800 |
| S2, typical (stress) | 1 × 7 × 2 × ρ = 0.3 | 50 | 700 |
| **Total** | | | **9,660** |

- **Item bank:** a port of the simulation code of König, Spoden & Frey (2022), *Behavior Research
  Methods* 55, 3965–3983 (OSF project m3zaq, `Scripts/Simulation/cauchy1.R`). Each replicate's
  sample SDs and correlation match the targets exactly, and the bank is accepted only if every
  parameter lies within the source's bounds.
- **Fitted model:** a hierarchical 2PL fitted with NUTS (PyMC).
  - θ ~ N(0, 1), which identifies the scale;
  - non-centered log a = μ + σ·z;
  - centered b ~ N(μ_b, σ_b);
  - half-normal hierarchical SDs;
  - 4 chains × (1,000 tune + 1,000 draws), target_accept 0.95.
- **Fit tiers.** Every fit is kept and classified:

  | Tier | Rule |
  |---|---|
  | gross | sampler error, non-finite estimates, or R-hat > 1.1 |
  | divergent | any divergence |
  | mixing_flag | R-hat > 1.01, or bulk/tail ESS < 400 |
  | clean | otherwise |

  - The primary results use every non-gross fit.
  - Sensitivity summaries use "no divergent" fits and "clean only" fits.
- **Limitations:**
  - full exposure only;
  - unidimensional 2PL;
  - simulated item banks.

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # Windows: .venv\Scripts\python
.venv/bin/python -m pip install -e .
```

**A C++ compiler is required.** PyTensor (PyMC's backend) compiles to C. Without a compiler it
falls back to a pure-Python mode that is orders of magnitude slower, so real fits refuse to run.
- Linux/macOS: g++ (or clang) on PATH.
- Windows: install a MinGW-w64 g++ (MSVC is not supported) and set `RECOVERY2PL_MINGW_BIN` to its
  `bin` directory.

## Test

```
python -m unittest discover -s tests -v                  # fast; no MCMC
RECOVERY2PL_SLOW=1 python -m unittest tests.test_integration_fit -v   # one tiny real fit, twice
```

## Run

A **work unit** is one condition × one replicate, with id `{condition_id}__r{replicate:03d}`. For
example: `N100_J25_typical_rho0.3_full__r007`, or `N100_J25_typical_rho0.3_full_S1__r007` for S1.
A unit's data and seeds depend only on that unit, so units can run in any order, on any machine,
and a failed unit can be rerun alone.

```
python -m recovery2pl list-units --config configs/final_v1.yaml            # 9,660 ids
python -m recovery2pl shard --config configs/final_v1.yaml --index 0 --of 64
python -m recovery2pl show-unit --config configs/final_v1.yaml --unit N50_J25_typical_rho0.3_full__r000
python -m recovery2pl run --config configs/smoke.yaml --shard 0/1 --out results
python -m recovery2pl run --config configs/final_v1.yaml --units-file my_units.txt --out results
python -m recovery2pl aggregate --config configs/smoke.yaml --out results
```

Set numerical-library thread caps in the environment before starting a run, e.g.
`OMP_NUM_THREADS=1`. Sampler chains run in parallel within a unit (`execution.chain_cores`).

### Parallel shards

```
python -m recovery2pl plan --config configs/precheck_s1s2.yaml --shards 12 --strategy balanced --json plan.json
python -m recovery2pl collect --config configs/precheck_s1s2.yaml --from downloaded_artifacts --out results
```

- `plan` writes a deterministic shard assignment.
  - `balanced`: longest-job-first packing on a provisional relative cost model,
    `units.estimated_cost`, to be recalibrated from measured timings.
  - `round_robin` is the alternative.
- `collect` merges unit files from many shard outputs. A unit present twice (e.g. after a retry)
  must have the same scientific hash and simulated parameters, and estimates within 1e-8;
  otherwise it raises.
- Shard assignment never affects results.

### GitHub Actions

**`tests.yml`** runs on every push. It:
- installs the pinned dependencies;
- checks that PyTensor has a C++ compiler;
- runs the fast tests and the tiny real fit twice.

**`run-units.yml`** is manually dispatched, with inputs config, shards, strategy, optional
explicit units and max_parallel. Its jobs:
1. **plan:** writes the shard plan.
2. **run:** one matrix job per shard. Each job:
   - runs its units;
   - records runner, CPU and compiler info;
   - uploads `results-<study_id>-<shard>-a<attempt>`;
   - fails on an infrastructure crash or a sampler exception. The unit is still written, tiered
     gross.
3. **merge:** collects every shard, fails if any planned unit is missing, aggregates, and
   uploads `merged-<study_id>-a<attempt>`.

## Outputs

Under `results/<study_id>/`:

| File | Content |
|---|---|
| `units/<condition_id>/rep_XXX.parquet` | one row per item: true and estimated a and b, 95% interval, posterior SD, per-item R-hat, condition fields, fit tier |
| `units/<condition_id>/rep_XXX.json` | fit diagnostics, tier, seeds, scientific hash, software versions. Written last: it marks the unit complete |
| `replicate_metrics.csv` | one row per unit: RMSE, bias, coverage, interval width, share of items within the configured tolerances, correlation, rank correlation, shrinkage slope, for b, a and log a |
| `summary_{primary,no_divergent,strict}.csv` | per condition, for every metric: mean, SD across replicates, Monte Carlo SE (SD/√R), median, 5th/10th/90th/95th percentiles, pooled RMSE, tier counts |
| `envelope_<name>_<metric>.csv` | per N, the maximum over a configured set of conditions of the condition mean, and which condition attained it |

Both unit files are written atomically. A unit whose stored scientific hash differs from the
current config is never overwritten.

## Reproducibility

- **Simulated data are bit-identical** for a given unit on any machine with the pinned
  dependencies. The tests check S0 datasets against frozen SHA-256 digests.
- **MCMC results** for a given unit and seed agree within numerical tolerance on the same
  software stack. They are not guaranteed to be byte-identical across platforms or compilers.

## Licensing

Not yet decided.
