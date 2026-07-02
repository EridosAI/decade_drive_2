"""
core/consistency.py
===================

Input-driven CONSISTENCY / echo-state-property (ESP) gate for Experiment A — the
operating-point criterion that REPLACES the autonomous origin-instability assert
inherited from 0.1/0.2 (the wrong axis for an input-DRIVEN reservoir; see
results/A/methods_gates.md).

The test (Lymburn et al. 2019; the replica-convergence form 0.3 used, converging to
~5e-4): drive two replicas of the SAME reservoir (identical omega, L, m, input u)
from DIFFERENT initial conditions. If the reservoir has the echo-state property, the
input washes out the initial condition and the two replica state trajectories
CONVERGE — i.e. the largest CONDITIONAL Lyapunov exponent (the divergence rate of
nearby trajectories ON the driven orbit) is negative. We measure convergence
directly as the post-washout normalised replica distance:

    d = RMS_t,i |z_a - z_b|  /  RMS_t,i |z_a|        (over the usable window)

ESP holds when d is tiny (deep convergence, ~1e-3); it is O(1) when the reservoir
keeps memory of its initial condition (no fading memory). The split is sharply
bimodal, so a single threshold cleanly gates.

Why this and not autonomous origin-stability / autonomous lambda_max:
* The input keeps the reservoir active even when the autonomous origin is stable
  (amplitude-death regime), so origin-instability is neither necessary nor
  sufficient for a good driven reservoir.
* ESP depends on BOTH K and beta: strong input can RESTORE consistency an
  origin-unstable point lacks, and over-drive (e.g. beta=8) can BREAK it. So the
  gate is evaluated across the (K, beta) grid, not K alone.

Points that FAIL the gate are flagged and EXCLUDED from the performance map: their
"performance" is initial-condition-dependent and therefore meaningless.
"""
from __future__ import annotations

import numpy as np

from core.reservoir import ResSpec

__all__ = ["replica_spec", "consistency_distance", "ESP_EPS"]

# Convergence threshold. Clearly-ESP points sit far below this (integration-error /
# deep-convergence floor); clearly-failed points are O(1). 0.3 reached ~5e-4.
ESP_EPS = 1e-2


def replica_spec(spec: ResSpec, z0_seed: int) -> ResSpec:
    """A replica of `spec` with the SAME omega/L/m but an INDEPENDENT random
    on-limit-cycle initial state (different z0). Used as the second replica for the
    consistency gate."""
    rng = np.random.default_rng(z0_seed)
    lam = 0.1  # core.reservoir.LAM
    z0 = np.sqrt(lam) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(spec.omega)))
    return ResSpec(spec.omega, spec.L, spec.m, z0)


def consistency_distance(Xa: np.ndarray, Xb: np.ndarray, sl: slice) -> float:
    """Post-washout normalised replica distance d (see module docstring). Xa, Xb are
    (L, N) complex state trajectories of two replicas (same input, different init);
    sl selects the post-washout usable window. ESP holds iff d < ESP_EPS."""
    a = Xa[sl]
    b = Xb[sl]
    num = np.sqrt(np.mean(np.abs(a - b) ** 2))
    den = np.sqrt(np.mean(np.abs(a) ** 2)) + 1e-12
    return float(num / den)
