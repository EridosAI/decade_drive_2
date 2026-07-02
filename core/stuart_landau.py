"""
core/stuart_landau.py
=====================

The JAX/diffrax Stuart-Landau network ODE -- the program's core nonlinear
dynamics module, reused by Experiments A and C.

Model
-----
    zdot_i = (lambda + i*omega_i - |z_i|^2) z_i + K(t) * sum_j A_ij (z_j - z_i)

with K(t) = K0 + eps*cos(Omega t). Conventions identical to core/magnus.py and
core/lyapunov.py: the coupling enters through the graph Laplacian, because
sum_j A_ij (z_j - z_i) = -(L z)_i, so

    zdot = (lambda + i*omega - |z|^2) * z  -  K(t) * (L @ z) .

Setting eps=0 gives the autonomous case; K0=eps=0 fully decouples the nodes.

Limit cycle
-----------
Each uncoupled node has an attracting limit cycle |z_i| = sqrt(lambda) with phase
advancing at omega_i: started exactly on it (K=0, no input/noise), the closed form
is z_i(t) = sqrt(lambda) * exp(i (omega_i t + phi_i0)). The amplitude is ATTRACTING
(self-correcting) while the phase is MARGINAL (no restoring force) -- so phase drift
is the binding integrator-accuracy criterion and amplitude/norm drift is a near-free
secondary check (see Experiment 0.2 / results/0/solver_benchmark.md).

Stiffness reframing
-------------------
The Jacobian eigenvalues on the limit cycle are -lambda +- i*omega: the REAL parts
are O(lambda) ~ 0.1 for every oscillator, the spread lives entirely in the
IMAGINARY parts (1 .. 1e4). Classical stiffness is a spread in real parts -- there
is none. This is a MULTIRATE / HIGHLY-OSCILLATORY problem, so high-order EXPLICIT
solvers are expected to win; implicit (Kvaerno5) is kept only as a hypothesis to
falsify.

Complex vs real-2N state
------------------------
diffrax integrates the complex state directly; its PID error norm handles complex
dtype (verified on the analytical anchor in tests/test_stuart_landau.py, all four
candidate solvers incl. implicit Kvaerno5). The complex path is the default. A real
2N representation [Re z, Im z] is provided (`sl_vector_field_real`, real_repr=True)
as a documented fallback should an adaptive norm ever misbehave on complex dtype.

Scope (deferred risk -- recorded)
---------------------------------
A deterministic input term (beta*u(t)*m_i) would be fine here. The NOISE term
(sigma*xi) turns this into an SDE requiring diffrax's SDE machinery
(ControlTerm/VirtualBrownianTree + an SDE solver) with its own, LOWER, strong-order
question. Experiment 0.2 validates the DETERMINISTIC integrator only; the high-order
recommendation does NOT transfer to noisy runs. Deferred to A.

Precision note
--------------
JAX defaults to float32. float64 requires `jax.config.update("jax_enable_x64", True)`
at startup (before array creation). With x64 enabled, precision is then selected per
run by the dtype of z0 (complex64 vs complex128). Use `enable_x64()` below.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import diffrax


__all__ = [
    "SLParams",
    "enable_x64",
    "sl_vector_field",
    "sl_vector_field_real",
    "limit_cycle_state",
    "SOLVERS",
    "integrate_sl",
]


def enable_x64() -> None:
    """Enable float64. Must be called at startup, before any array is created."""
    jax.config.update("jax_enable_x64", True)


class SLParams(NamedTuple):
    """Parameters for the Stuart-Landau network vector field (a diffrax args pytree).

    lam   : scalar linear gain (lambda)
    omega : (N,) natural frequencies
    L     : (N, N) real graph Laplacian (coupling enters as -K(t) L)
    K0    : baseline coupling
    eps   : drive amplitude   (eps=0 -> autonomous)
    Omega : drive frequency
    """
    lam: float
    omega: jnp.ndarray
    L: jnp.ndarray
    K0: float
    eps: float
    Omega: float


def sl_vector_field(t, z, p: SLParams):
    """Complex Stuart-Landau vector field f(t, z, args) for diffrax ODETerm.

    zdot = (lambda + i*omega - |z|^2) z - K(t) (L z),  K(t)=K0+eps cos(Omega t).
    """
    K = p.K0 + p.eps * jnp.cos(p.Omega * t)
    return (p.lam + 1j * p.omega - jnp.abs(z) ** 2) * z - K * (p.L @ z)


def sl_vector_field_real(t, y, p: SLParams):
    """Real 2N representation: y = [Re z, Im z]. Documented fallback if a solver's
    adaptive error norm ever misbehaves on complex dtype (the complex path is the
    validated default)."""
    N = y.shape[0] // 2
    z = y[:N] + 1j * y[N:]
    dz = sl_vector_field(t, z, p)
    return jnp.concatenate([dz.real, dz.imag])


def limit_cycle_state(lam: float, phases) -> jnp.ndarray:
    """On-limit-cycle initial state z_i = sqrt(lambda) e^{i phi_i}."""
    phases = jnp.asarray(phases)
    return jnp.sqrt(lam) * jnp.exp(1j * phases)


# Candidate solvers (instantiated per use). Explicit high-order expected to win;
# Kvaerno5 (implicit) kept as the hypothesis to falsify.
SOLVERS = {
    "Tsit5": diffrax.Tsit5,
    "Dopri5": diffrax.Dopri5,
    "Dopri8": diffrax.Dopri8,
    "Kvaerno5": diffrax.Kvaerno5,
}


def integrate_sl(params: SLParams, z0, t0, t1, ts, solver, rtol, atol,
                 dt0=None, max_steps=2 ** 20, real_repr=False):
    """Integrate the SL network with an adaptive PID controller.

    params    : SLParams (carried as diffrax args)
    z0        : (N,) complex initial state; its dtype selects precision
                (complex64 -> float32 path, complex128 -> float64 path)
    ts        : save grid (jnp array); SaveAt(ts=ts), dense-interpolated
    solver    : a diffrax solver INSTANCE (e.g. SOLVERS["Dopri8"]())
    rtol,atol : PID tolerances (the cost knob the 0.2 benchmark optimises)
    real_repr : integrate the real 2N system instead of complex state

    Returns (zs, stats): zs is (len(ts), N) complex; stats is diffrax sol.stats
    (keys incl. num_steps, num_accepted_steps, num_rejected_steps).
    """
    if dt0 is None:
        dt0 = (t1 - t0) / 1.0e4
    if real_repr:
        term = diffrax.ODETerm(sl_vector_field_real)
        y0 = jnp.concatenate([z0.real, z0.imag])
    else:
        term = diffrax.ODETerm(sl_vector_field)
        y0 = z0
    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=params,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts), max_steps=max_steps,
    )
    jax.block_until_ready(sol.ys)        # finish compute before the timer stops
    # Return HOST (numpy) arrays. diffeqsolve outputs are lazy DEVICE arrays; left
    # lazy, a downstream metric reduction re-traces and recomputes the whole solve
    # (wrong benchmark timing, wasted work). Materialising to host once gives a
    # stable concrete array and keeps all metric code in plain numpy. At 0.2/A/C
    # save sizes the copy is negligible. (If A/C later need on-device results for
    # batched GPU post-processing, add an opt-in flag; host is right for 0.2.)
    if real_repr:
        N = y0.shape[0] // 2
        zs = np.asarray(sol.ys[:, :N]) + 1j * np.asarray(sol.ys[:, N:])
    else:
        zs = np.asarray(sol.ys)
    return zs, sol.stats
