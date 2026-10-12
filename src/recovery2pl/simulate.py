"""Data-generating process. Deterministic seeds, independent of execution order.

    P(Y_si = 1) = sigmoid(a_i (theta_s - b_i)),   a_i > 0

Item bank (`koenig2022`): a port of the simulation code of Koenig, Spoden & Frey (2022), "Robustness
of the performance of the optimized hierarchical two-parameter logistic IRT model for small-sample
item calibration", Behavior Research Methods 55:3965-3983 (code: OSF m3zaq,
Scripts/Simulation/cauchy1.R). Per replicate:
  1. Z: J independent truncated-normal pairs, location (alpha_location, beta_location), limits
     aux_lower/aux_upper. As in the source, the tau matrix is passed as the COVARIANCE of this
     auxiliary draw (auxiliary SDs = sqrt(tau)); it only shapes Z, because step 2 whitens it away.
  2. xi = Z chol(cov(Z))^-1 chol(Sigma), Sigma = diag(tau) Omega diag(tau): the replicate's SAMPLE
     SDs are exactly (tau_alpha, tau_beta) and its SAMPLE correlation exactly rho, on the raw alpha
     scale.
  3. alpha = xi_1 - mean + E[TN(alpha_location, tau_alpha, aux limits)], beta likewise.
  4. whole-replicate rejection until all alpha lie in accept_alpha and all beta in accept_beta
     (open intervals). Rejection never alters step 2-3's exact SDs and correlation.

Student population (config `populations`), applied AFTER the bank is generated and validated:
  - difficulty_shift d: b <- b + d (a pure targeting translation). The bank's bounds are NOT
    re-applied after shifting, so the shifted bank is exactly the reference bank moved by d.
    Abilities stay standardized, so d is measured in ability SDs and the fitted model's
    theta ~ N(0, 1) identification is unchanged.
  - theta kind `normal`: theta ~ N(0, 1).
  - theta kind `skewnormal_standardized` (shape s): X ~ SkewNormal(s), theta = (X - E X) / SD X with
    the exact analytic moments, so theta has mean 0, SD 1 and the skew-normal's skewness.

Seeding (SHA-256 of explicit strings):
  - item bank: (generator, test length, world, rho, replicate) -- shared by every N and every
    population, so all N recover the same true items and populations are paired at item level;
  - abilities and responses: (condition id, replicate);
  - sampler: (condition id, replicate).
The reference population keeps the bare condition id N{N}_J{J}_{world}_rho{rho}_full; other
populations append _{population}.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, skewnorm, truncnorm

EXPOSURE = "full"           # every simulated student answers every simulated item
REFERENCE_SHIFT = 0.0


@dataclass(frozen=True)
class Condition:
    N: int
    test_length: int
    world: str
    tau_a: float               # SD of raw alpha
    tau_b: float               # SD of beta (before any population shift)
    rho: float                 # corr(alpha, beta), raw scale
    population: str
    difficulty_shift: float
    theta_kind: str
    theta_shape: float = 0.0   # skew-normal shape (only for skewnormal_standardized)

    @property
    def is_reference(self) -> bool:
        return self.difficulty_shift == REFERENCE_SHIFT and self.theta_kind == "normal"

    @property
    def condition_id(self) -> str:
        base = f"N{self.N}_J{self.test_length}_{self.world}_rho{self.rho:g}_{EXPOSURE}"
        return base if self.is_reference else f"{base}_{self.population}"

    @property
    def item_family(self) -> str:
        """Everything the generated (unshifted) item bank depends on: not N, not population."""
        return f"koenig2022_J{self.test_length}_{self.world}_rho{self.rho:g}"


def derive_seed(base_seed: int, *parts) -> int:
    text = "|".join([str(base_seed), *map(str, parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2 ** 63)


# ---------------------------------------------------------------- item bank (Koenig et al. 2022)

def truncated_normal_mean(mu: float, sd: float, lower: float, upper: float) -> float:
    """E[X] for X ~ N(mu, sd) truncated to [lower, upper] (the source's mean.tnorm)."""
    lo, hi = (lower - mu) / sd, (upper - mu) / sd
    return mu + sd * (norm.pdf(lo) - norm.pdf(hi)) / (norm.cdf(hi) - norm.cdf(lo))


def koenig2022_items(J: int, tau_alpha: float, tau_beta: float, rho: float, gen: dict,
                     rng: np.random.Generator) -> tuple:
    """Returns (alpha, beta, tries). See the module docstring for the four steps."""
    loc = np.array([gen["alpha_location"], gen["beta_location"]], dtype=float)
    lower, upper = np.array(gen["aux_lower"], float), np.array(gen["aux_upper"], float)
    tau = np.array([tau_alpha, tau_beta], float)
    aux_sd = np.sqrt(tau)                                   # source quirk: sigma = diag(tau)
    sigma = np.diag(tau) @ np.array([[1.0, rho], [rho, 1.0]]) @ np.diag(tau)
    L = np.linalg.cholesky(sigma).T                        # upper factor, as R's chol()
    centers = [truncated_normal_mean(loc[k], tau[k], lower[k], upper[k]) for k in (0, 1)]
    if gen["center"] != "truncated_mean":
        raise ValueError(f"unsupported center {gen['center']!r}")
    (a_lo, a_hi), (b_lo, b_hi) = gen["accept_alpha"], gen["accept_beta"]
    for tries in range(1, gen["max_tries"] + 1):
        Z = np.column_stack([
            truncnorm.rvs((lower[k] - loc[k]) / aux_sd[k], (upper[k] - loc[k]) / aux_sd[k],
                          loc=loc[k], scale=aux_sd[k], size=J, random_state=rng) for k in (0, 1)])
        H = np.linalg.cholesky(np.cov(Z, rowvar=False)).T  # upper factor of the sample cov
        xi = Z @ np.linalg.inv(H) @ L
        alpha = xi[:, 0] - xi[:, 0].mean() + centers[0]
        beta = xi[:, 1] - xi[:, 1].mean() + centers[1]
        if alpha.min() > a_lo and alpha.max() < a_hi and beta.min() > b_lo and beta.max() < b_hi:
            return alpha, beta, tries
    raise RuntimeError(f"koenig2022: no replicate within bounds after {gen['max_tries']} tries "
                       f"(J={J}, tau=({tau_alpha}, {tau_beta}), rho={rho})")


# ---------------------------------------------------------------- students and responses

def skewnormal_moments(shape: float) -> tuple:
    """Exact mean and SD of the standard skew-normal with the given shape."""
    delta = shape / np.sqrt(1.0 + shape ** 2)
    mean = delta * np.sqrt(2.0 / np.pi)
    sd = np.sqrt(1.0 - 2.0 * delta ** 2 / np.pi)
    return float(mean), float(sd)


def draw_theta(N: int, rng: np.random.Generator, kind: str, shape: float = 0.0) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(N)
    if kind == "skewnormal_standardized":
        mean, sd = skewnormal_moments(shape)
        return (skewnorm.rvs(shape, size=N, random_state=rng) - mean) / sd
    raise ValueError(f"unknown theta kind {kind!r}")


def simulate_responses(a: np.ndarray, b: np.ndarray, theta: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """Full exposure: an N x J int8 matrix of 2PL responses."""
    p = 1.0 / (1.0 + np.exp(-(a[None, :] * (theta[:, None] - b[None, :]))))
    return (rng.random(p.shape) < p).astype(np.int8)


def generate_bank(config: dict, cond: Condition, rep: int) -> tuple:
    """The validated, UNSHIFTED item bank (identical for every N and population)."""
    rng = np.random.default_rng(derive_seed(config["base_seed"], "items", cond.item_family, rep))
    return koenig2022_items(cond.test_length, cond.tau_a, cond.tau_b, cond.rho,
                            config["item_generator"], rng)


def simulate_replicate(config: dict, cond: Condition, rep: int) -> dict:
    base = config["base_seed"]
    a, b_bank, tries = generate_bank(config, cond, rep)
    b = b_bank + cond.difficulty_shift           # pure translation; bank bounds are not re-applied
    rng = np.random.default_rng(derive_seed(base, "responses", cond.condition_id, rep))
    theta = draw_theta(cond.N, rng, cond.theta_kind, cond.theta_shape)
    Y = simulate_responses(a, b, theta, rng)
    return {"a": a, "b": b, "theta": theta, "Y": Y, "item_tries": tries,
            "sampler_seed": derive_seed(base, "sampler", cond.condition_id, rep)}
