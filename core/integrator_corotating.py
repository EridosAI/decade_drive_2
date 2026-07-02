"""
core/integrator_corotating.py
=============================

Co-rotating-frame ("integrating-factor") integrator for the driven Stuart-Landau
network -- Experiment C's fast hot loop, parked since 0.3 and built/validated here
as C's Gate 0. The accuracy reference is diffrax Dopri8/float64 (0.2); the Floquet
cross-check is 0.1's monodromy_qr (core/lyapunov.py).

The problem it solves
---------------------
The full driven substrate is

    z_i' = (lambda + i omega_i - |z_i|^2) z_i - K(t) (L z)_i  [+ beta u m_i + sigma xi_i]
    K(t) = K0 + eps cos(Omega t).

The DOMINANT term is the fast self-rotation i omega_i z_i (omega up to 1e4). It is not
classically stiff (real parts ~ -lambda) but it is HIGHLY OSCILLATORY, so a low-order
explicit method must take ~thousands of steps per fast period (0.3: RK4 needed ~2048).

Co-rotating substitution
------------------------
Let v_i = z_i e^{-i omega_i t}  (so z_i = v_i e^{i omega_i t}, and |z_i| = |v_i|).
The i omega_i term cancels analytically, leaving the SLOW remainder

    v_i' = (lambda - |v_i|^2) v_i  -  K(t) [ Ltilde(t) v ]_i  [+ beta u m_i e^{-i omega_i t} ...]

where Ltilde(t) = R(t)^{-1} L R(t),  R(t) = diag(e^{i omega_i t}), i.e.

    [Ltilde(t) v]_i = e^{-i omega_i t} * ( L @ (e^{i omega_i t} * v) )_i .

Equivalently: rotate v UP to the lab frame (z = e^{i omega t} * v), apply L, rotate the
result BACK (* e^{-i omega t}). No Ltilde matrix is ever formed -- two elementwise phase
multiplies and one matvec, O(N^2) like the original field, but the integrated variable v
is slow.

Why it is fast -- and the MEASURED limit at multi-decade (read this)
-------------------------------------------------------------------
* The fast O(omega_max) self-rotation is GONE. The residual time dependence is the
  coupling beat phases e^{i(omega_j-omega_i)t} with amplitude O(K).
* The big speedup the frame promises is real ONLY when the frequency SPREAD (the largest
  coupling difference |omega_i-omega_j|) is small relative to omega_max -- i.e. CLUSTERED
  oscillators at high absolute frequency. Then the step resolves the small differences
  while the lab frame would resolve the large absolute omega.
* MEASURED FINDING (validation 8, results/C/engine_benchmark.md): for a MULTI-DECADE
  log-uniform spectrum omega in [1, 10^d], the largest difference ~ omega_max - omega_min
  ~ omega_max. Under-resolving it does NOT just miss a tiny ripple -- it CORRUPTS the
  amplitude envelope too (RK4 aliasing of an O(K)-amplitude oscillation), ~5% at coarse
  steps. So at multi-decade the step must resolve ~omega_max anyway, and the wall-time
  speedup over Dopri8 is MODEST (the RK4-cheap-step / no-adaptivity factor, ~single
  digits), NOT the ~200x the pre-spec saw on a 2-osc gap. The spec anticipated this as
  the dense-spectrum risk; it is now retired by measurement, not assumed.
* ON resonance (Omega ~ |omega_i-omega_j|, the Arnold-tongue / combination resonance C
  is built to detect), the beat times the drive produces a near-DC RESONANT transfer.
  A step that averages away the fast ripple averages away the resonance too -> confident
  WRONG answer. The step MUST resolve the drive Omega and any near-resonant difference
  frequency. See `recommended_h` and the resonance gate (validation 3).

Method (cheapest sufficient, per spec): plain fixed-step RK4 in a lax.scan, float64,
vmap-batchable over seeds and (K0, eps, Omega) grid points. No ETD phi-function machinery
-- once the diagonal D is removed by the frame change there is no fast LINEAR term left to
exponentiate (the residual fast content is inside the nonlinear coupling, which ETD does
not help). Escalation path (RK8 -> Magnus exponential integrator) is documented in the
spec and only taken if validation forces it.

What this module provides
-------------------------
* integrate_corotating(_batch) -- nonlinear trajectory of v, returned ROTATED BACK to z
  so it is directly comparable to core.stuart_landau.integrate_sl output.
* monodromy_corotating -- linearised variational propagation (Jacobian-apply closure)
  with Wolf QR-renorm in the co-rotating frame; reproduces 0.1's Floquet exponents
  (validation 5). Origin generator supplied; the driven-limit-cycle Jacobian is a
  one-closure swap for C.
* integrate_corotating_sde -- additive-noise Euler-Maruyama in the frame (validation 7).
* recommended_h -- the documented step-size rule.

Precision: float64 required (call enable_x64() before any array is built). Inherited from
core.stuart_landau.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from core.stuart_landau import enable_x64  # re-export for callers

LAM_DEFAULT = 0.1

__all__ = [
    "enable_x64",
    "recommended_h",
    "n_steps_for",
    "co_field",
    "rk4_step",
    "integrate_corotating",
    "integrate_corotating_batch",
    "co_field_input",
    "rk4_step_input",
    "integrate_corotating_input",
    "integrate_corotating_input_batch",
    "co_field_input_comb",
    "rk4_step_input_comb",
    "integrate_corotating_input_comb",
    "integrate_corotating_input_comb_batch",
    "co_generator_origin",
    "monodromy_corotating",
    "limitcycle_generator_real",
    "monodromy_limitcycle",
    "leading_multiplier",
    "driven_reference",
    "frozen_reference",
    "integrate_corotating_sde",
]


# --------------------------------------------------------------------------- #
# Step-size rule (the central correctness knob -- documented per spec)
# --------------------------------------------------------------------------- #
def recommended_h(Omega: float, resolve_freqs=(), spp_safe: int = 16) -> float:
    """Fixed RK4 step h = (2*pi / f_resolve) / spp_safe, f_resolve = max(Omega, max|resolve_freqs|).

    STEP-SIZE RULE (validation 3 + 8, measured -- not assumed):
    The step must resolve (a) the drive Omega, and (b) every coupling DIFFERENCE
    frequency Delta_ij = |omega_i-omega_j| whose contribution exceeds tolerance. Pass
    those in `resolve_freqs`.

    Two cases, both measured:
      * Near a resonance (|Delta - n*Omega| small): that Delta MUST be in resolve_freqs
        or the resonant transfer is averaged away (a confident wrong answer). The
        resonance gate (validation 3) set spp_safe >= 16.
      * For ACCURACY on a multi-decade log-uniform spectrum: under-resolving the LARGEST
        difference (~omega_max) corrupts the amplitude envelope too (~5% at coarse steps),
        not just the phase -- so pass the largest difference (~omega_max) in resolve_freqs.
        This is why the multi-decade speedup is modest (see the module header).

    Off resonance with genuinely CLUSTERED frequencies (spread << omega_max), pass only
    the small in-cluster differences -- there the frame's speedup is large.

    spp_safe = samples per fastest RESOLVED period; 16 is the validated default.
    """
    fr = float(Omega)
    if len(resolve_freqs):
        fr = max(fr, float(np.max(np.abs(np.asarray(resolve_freqs)))))
    if fr <= 0.0:
        raise ValueError("f_resolve must be > 0 (need a drive or a resolve frequency)")
    return (2.0 * np.pi / fr) / float(spp_safe)


def n_steps_for(t0: float, t1: float, h: float) -> int:
    """Number of fixed RK4 steps to cover [t0, t1] at step h (rounded)."""
    return int(round((t1 - t0) / h))


# --------------------------------------------------------------------------- #
# Nonlinear vector field in the co-rotating frame
# --------------------------------------------------------------------------- #
def co_field(t, v, omega, L, K0, eps, Omega, lam):
    """Co-rotating-frame field v' = f(t, v) for the bare driven SL network.

        f = (lam - |v|^2) v  -  K(t) * e^{-i omega t} (x) ( L @ (e^{i omega t} (x) v) )
        K(t) = K0 + eps cos(Omega t).

    The fast self-rotation i omega is absent by construction. `(x)` is elementwise.
    Shapes: v (N,), omega (N,), L (N,N) -> (N,) complex.
    """
    K = K0 + eps * jnp.cos(Omega * t)
    rot = jnp.exp(1j * omega * t)            # e^{i omega t} : lift v -> lab frame z
    z = rot * v
    Lz = L @ z
    coupling = -K * (jnp.conj(rot) * Lz)     # -K(t) e^{-i omega t} (x) (L z)
    return (lam - jnp.abs(v) ** 2) * v + coupling


def rk4_step(v, t, h, omega, L, K0, eps, Omega, lam):
    """One classical RK4 step of co_field from (t, v) by h."""
    k1 = co_field(t, v, omega, L, K0, eps, Omega, lam)
    k2 = co_field(t + 0.5 * h, v + 0.5 * h * k1, omega, L, K0, eps, Omega, lam)
    k3 = co_field(t + 0.5 * h, v + 0.5 * h * k2, omega, L, K0, eps, Omega, lam)
    k4 = co_field(t + h, v + h * k3, omega, L, K0, eps, Omega, lam)
    return v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --------------------------------------------------------------------------- #
# Single-trajectory integrator (jit-compiled; vmap via _batch below)
# --------------------------------------------------------------------------- #
def _integrate_one(omega, L, z0, K0, eps, Omega, lam, t0, h, n_save, save_every):
    """Integrate one trajectory; return (ts, zs) with zs ROTATED BACK to the lab frame.

    Static (python int) args: n_save, save_every. Total steps = n_save*save_every.
    Saves AFTER each block of `save_every` steps, at ts[k] = t0 + (k+1)*save_every*h.
    The returned zs = vs * e^{i omega ts} matches core.stuart_landau.integrate_sl.
    """
    v0 = z0.astype(jnp.complex128) * jnp.exp(-1j * omega * t0)   # z0 -> v0 at t0

    def block(carry, _):
        v, t = carry

        def body(c, _):
            v, t = c
            v = rk4_step(v, t, h, omega, L, K0, eps, Omega, lam)
            return (v, t + h), None

        (v, t), _ = jax.lax.scan(body, (v, t), None, length=save_every)
        return (v, t), v

    (_, _), vs = jax.lax.scan(block, (v0, t0), None, length=n_save)
    ts = t0 + (jnp.arange(n_save) + 1.0) * (save_every * h)
    zs = vs * jnp.exp(1j * omega[None, :] * ts[:, None])
    return ts, zs


def integrate_corotating(omega, L, z0, K0, eps, Omega, t0, h, n_save, save_every,
                         lam: float = LAM_DEFAULT):
    """Host-facing single-trajectory wrapper. Returns (ts, zs) as numpy host arrays;
    zs is (n_save, N) complex in the LAB frame (rotated back). Inputs may be numpy."""
    f = jax.jit(_integrate_one, static_argnums=(9, 10))
    ts, zs = f(jnp.asarray(omega, dtype=float), jnp.asarray(L, dtype=float),
               jnp.asarray(z0, dtype=jnp.complex128),
               jnp.float64(K0), jnp.float64(eps), jnp.float64(Omega),
               jnp.float64(lam), jnp.float64(t0), jnp.float64(h),
               int(n_save), int(save_every))
    jax.block_until_ready(zs)
    return np.asarray(ts), np.asarray(zs)


def integrate_corotating_batch(omegas, Ls, z0s, K0s, epss, Omegas, t0, h,
                               n_save, save_every, lam: float = LAM_DEFAULT,
                               return_ts: bool = True):
    """vmap-batched integrator over B trajectories (seeds x (K0,eps,Omega) grid points).

    omegas (B,N), Ls (B,N,N), z0s (B,N) complex; K0s/epss/Omegas (B,). h, n_save,
    save_every shared (one span). Returns (ts (n_save,), zs (B, n_save, N)) host arrays.
    The 0.2/GATE-3 finding (single-traj launch-bound under WSL2; batching amortises) is
    why this is the throughput path.
    """
    def one(omega, L, z0, K0, eps, Omega):
        return _integrate_one(omega, L, z0, K0, eps, Omega, lam, t0, h,
                              int(n_save), int(save_every))

    f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0)))
    ts, zs = f(jnp.asarray(omegas, dtype=float), jnp.asarray(Ls, dtype=float),
               jnp.asarray(z0s, dtype=jnp.complex128),
               jnp.asarray(K0s, dtype=float), jnp.asarray(epss, dtype=float),
               jnp.asarray(Omegas, dtype=float))
    jax.block_until_ready(zs)
    if return_ts:
        return np.asarray(ts[0]), np.asarray(zs)
    return np.asarray(zs)


# --------------------------------------------------------------------------- #
# Input-driven field + integrator (C-build prerequisite 1)
# --------------------------------------------------------------------------- #
# C is a RESERVOIR experiment: the substrate is driven by the task input
# u(t) as well as the slow parametric coupling drive K(t). A (core/reservoir.py)
# integrated  zdot = (lam+i omega-|z|^2)z - K(Lz) + beta u(t) m  with Dopri8 in the
# LAB frame; C must reproduce that in the co-rotating frame (the C-reproduces-A
# bridge, tests/test_C_prereq.py) and then switch the parametric drive on. The lab
# input term beta u m maps into the frame by the same e^{-i omega t} rotation as the
# coupling: v' gets  + beta u(t) m_i e^{-i omega_i t}.
#
# Input convention (inherited from A, GATE 1): u(t) is held PIECEWISE-CONSTANT over
# dt_in and the state is sampled stroboscopically at the END of each dt_in block
# (ts[k] = t0 + (k+1) dt_in), matching core.reservoir.run_reservoir exactly so the
# two integrators are a single-variable comparison. Each dt_in block is integrated
# with `n_sub` fixed RK4 substeps (h = dt_in/n_sub); within a block u is constant so
# the only sub-block time dependence is the (slow, in-frame) drift, coupling beats,
# and the K(t) drive -- the input jump lands exactly on a block boundary, never
# inside an RK4 step.
def co_field_input(t, v, u, omega, L, m, K0, eps, Omega, beta, lam):
    """Co-rotating-frame field WITH input, v' = f(t, v; u):

        f = (lam - |v|^2) v
            - K(t) e^{-i omega t} (x) ( L @ (e^{i omega t} (x) v) )
            + beta u m (x) e^{-i omega t} ,
        K(t) = K0 + eps cos(Omega t).

    `u` is the (scalar) input value for the current hold block; `m` (N,) complex is
    the input projection. With eps=0 this is the co-rotating image of A's driven SL
    field (core.reservoir._det_field) -- the bridge case. `(x)` is elementwise.
    """
    K = K0 + eps * jnp.cos(Omega * t)
    rot = jnp.exp(1j * omega * t)            # e^{i omega t}
    z = rot * v
    Lz = L @ z
    coupling = -K * (jnp.conj(rot) * Lz)
    drive_in = beta * u * m * jnp.conj(rot)  # beta u m_i e^{-i omega_i t}
    return (lam - jnp.abs(v) ** 2) * v + coupling + drive_in


def rk4_step_input(v, t, h, u, omega, L, m, K0, eps, Omega, beta, lam):
    """One classical RK4 step of co_field_input from (t, v) by h, u held constant."""
    k1 = co_field_input(t, v, u, omega, L, m, K0, eps, Omega, beta, lam)
    k2 = co_field_input(t + 0.5 * h, v + 0.5 * h * k1, u, omega, L, m, K0, eps, Omega, beta, lam)
    k3 = co_field_input(t + 0.5 * h, v + 0.5 * h * k2, u, omega, L, m, K0, eps, Omega, beta, lam)
    k4 = co_field_input(t + h, v + h * k3, u, omega, L, m, K0, eps, Omega, beta, lam)
    return v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_one_input(omega, L, m, z0, u_seq, K0, eps, Omega, beta, lam, t0, h, n_sub):
    """Input-driven single trajectory; return (ts, zs) ROTATED BACK to the lab frame.

    Static (python int) arg: n_sub (RK4 substeps per dt_in hold block). The number
    of save points equals len(u_seq); the block scan consumes u_seq one value per
    block. dt_in = n_sub*h is implied by the caller. ts[k] = t0 + (k+1)*n_sub*h.
    """
    v0 = z0.astype(jnp.complex128) * jnp.exp(-1j * omega * t0)

    def block(carry, u):
        v, t = carry

        def body(c, _):
            v, t = c
            v = rk4_step_input(v, t, h, u, omega, L, m, K0, eps, Omega, beta, lam)
            return (v, t + h), None

        (v, t), _ = jax.lax.scan(body, (v, t), None, length=n_sub)
        return (v, t), v

    (_, _), vs = jax.lax.scan(block, (v0, t0), u_seq)
    n_save = u_seq.shape[0]
    ts = t0 + (jnp.arange(n_save) + 1.0) * (n_sub * h)
    zs = vs * jnp.exp(1j * omega[None, :] * ts[:, None])
    return ts, zs


def integrate_corotating_input(omega, L, m, z0, u_seq, K0, eps, Omega, beta,
                               dt_in, n_sub, t0: float = 0.0,
                               lam: float = LAM_DEFAULT):
    """Host-facing input-driven single-trajectory wrapper (lab-frame zs).

    u_seq (L_samp,) real input held over dt_in; the state is saved at the END of each
    dt_in block (ts = (k+1) dt_in), so X = zs matches core.reservoir.run_reservoir's
    sampling. n_sub fixed RK4 substeps per block, h = dt_in/n_sub. Returns
    (ts (L_samp,), zs (L_samp, N) complex) host arrays.
    """
    n_sub = int(n_sub)
    h = float(dt_in) / n_sub
    f = jax.jit(_integrate_one_input, static_argnums=(12,))
    ts, zs = f(jnp.asarray(omega, dtype=float), jnp.asarray(L, dtype=float),
               jnp.asarray(m, dtype=jnp.complex128),
               jnp.asarray(z0, dtype=jnp.complex128),
               jnp.asarray(u_seq, dtype=float),
               jnp.float64(K0), jnp.float64(eps), jnp.float64(Omega),
               jnp.float64(beta), jnp.float64(lam), jnp.float64(t0),
               jnp.float64(h), n_sub)
    jax.block_until_ready(zs)
    return np.asarray(ts), np.asarray(zs)


def integrate_corotating_input_batch(omegas, Ls, ms, z0s, u_seqs, K0s, epss, Omegas,
                                     betas, dt_in, n_sub, t0: float = 0.0,
                                     lam: float = LAM_DEFAULT, return_ts: bool = True):
    """vmap-batched input-driven integrator over B trajectories (seeds x grid points).

    omegas (B,N), Ls (B,N,N), ms (B,N) complex, z0s (B,N) complex, u_seqs (B,L_samp);
    K0s/epss/Omegas/betas (B,). dt_in, n_sub shared (one span/hold rate). Returns
    (ts (L_samp,), zs (B, L_samp, N)) host arrays. Batching is the throughput path
    (0.2/GATE-3: single-traj launch-bound under WSL2).
    """
    n_sub = int(n_sub)
    h = float(dt_in) / n_sub

    def one(omega, L, m, z0, u_seq, K0, eps, Omega, beta):
        return _integrate_one_input(omega, L, m, z0, u_seq, K0, eps, Omega, beta,
                                    lam, t0, jnp.float64(h), n_sub)

    f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0)))
    ts, zs = f(jnp.asarray(omegas, dtype=float), jnp.asarray(Ls, dtype=float),
               jnp.asarray(ms, dtype=jnp.complex128),
               jnp.asarray(z0s, dtype=jnp.complex128),
               jnp.asarray(u_seqs, dtype=float),
               jnp.asarray(K0s, dtype=float), jnp.asarray(epss, dtype=float),
               jnp.asarray(Omegas, dtype=float), jnp.asarray(betas, dtype=float))
    jax.block_until_ready(zs)
    if return_ts:
        return np.asarray(ts[0]), np.asarray(zs)
    return np.asarray(zs)


# --------------------------------------------------------------------------- #
# Multi-tone (comb) parametric drive -- Experiment D Phase-2 (cross-band routing
# knob). Identical to the single-cosine input path above EXCEPT K(t) is now a comb
#   K(t) = K0 + sum_m eps_vec[m] cos(Omega_vec[m] t + phi_vec[m])
# (the single SOURCE waveform core.nulls.comb_K; matched comb / off-resonance comb /
# matched-variance broadband are all instances). For M=1, phi=0 this reproduces
# co_field_input to float64 (the Phase-2 sandbox asserts the bridge). The single-tone
# path above is left byte-identical so Gate-0 / Phase-1 still reproduce exactly.
# The eps/Omega/phi tone vectors are (M,) (traced); n_sub stays static.
# --------------------------------------------------------------------------- #
def co_field_input_comb(t, v, u, omega, L, m, K0, eps_vec, Omega_vec, phi_vec, beta, lam):
    """Co-rotating-frame field WITH input and a COMB parametric drive:

        f = (lam - |v|^2) v
            - K(t) e^{-i omega t} (x) ( L @ (e^{i omega t} (x) v) )
            + beta u m (x) e^{-i omega t} ,
        K(t) = K0 + sum_m eps_vec[m] cos(Omega_vec[m] t + phi_vec[m]).

    eps_vec/Omega_vec/phi_vec are (M,). M=1, phi=0 -> co_field_input. `(x)` elementwise."""
    K = K0 + jnp.sum(eps_vec * jnp.cos(Omega_vec * t + phi_vec))
    rot = jnp.exp(1j * omega * t)            # e^{i omega t}
    z = rot * v
    Lz = L @ z
    coupling = -K * (jnp.conj(rot) * Lz)
    drive_in = beta * u * m * jnp.conj(rot)  # beta u m_i e^{-i omega_i t}
    return (lam - jnp.abs(v) ** 2) * v + coupling + drive_in


def rk4_step_input_comb(v, t, h, u, omega, L, m, K0, eps_vec, Omega_vec, phi_vec, beta, lam):
    """One classical RK4 step of co_field_input_comb from (t, v) by h, u held constant."""
    k1 = co_field_input_comb(t, v, u, omega, L, m, K0, eps_vec, Omega_vec, phi_vec, beta, lam)
    k2 = co_field_input_comb(t + 0.5 * h, v + 0.5 * h * k1, u, omega, L, m, K0, eps_vec,
                             Omega_vec, phi_vec, beta, lam)
    k3 = co_field_input_comb(t + 0.5 * h, v + 0.5 * h * k2, u, omega, L, m, K0, eps_vec,
                             Omega_vec, phi_vec, beta, lam)
    k4 = co_field_input_comb(t + h, v + h * k3, u, omega, L, m, K0, eps_vec, Omega_vec,
                             phi_vec, beta, lam)
    return v + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _integrate_one_input_comb(omega, L, m, z0, u_seq, K0, eps_vec, Omega_vec, phi_vec,
                              beta, lam, t0, h, n_sub):
    """Comb-drive input trajectory; return (ts, zs) ROTATED BACK to the lab frame.
    Mirrors _integrate_one_input (static n_sub; one u value per saved block)."""
    v0 = z0.astype(jnp.complex128) * jnp.exp(-1j * omega * t0)

    def block(carry, u):
        v, t = carry

        def body(c, _):
            v, t = c
            v = rk4_step_input_comb(v, t, h, u, omega, L, m, K0, eps_vec, Omega_vec,
                                    phi_vec, beta, lam)
            return (v, t + h), None

        (v, t), _ = jax.lax.scan(body, (v, t), None, length=n_sub)
        return (v, t), v

    (_, _), vs = jax.lax.scan(block, (v0, t0), u_seq)
    n_save = u_seq.shape[0]
    ts = t0 + (jnp.arange(n_save) + 1.0) * (n_sub * h)
    zs = vs * jnp.exp(1j * omega[None, :] * ts[:, None])
    return ts, zs


def integrate_corotating_input_comb(omega, L, m, z0, u_seq, K0, eps_vec, Omega_vec,
                                    phi_vec, beta, dt_in, n_sub, t0: float = 0.0,
                                    lam: float = LAM_DEFAULT):
    """Host-facing comb-drive single-trajectory wrapper (lab-frame zs). Sampling matches
    integrate_corotating_input (state saved at the END of each dt_in block)."""
    n_sub = int(n_sub)
    h = float(dt_in) / n_sub
    f = jax.jit(_integrate_one_input_comb, static_argnums=(13,))
    ts, zs = f(jnp.asarray(omega, dtype=float), jnp.asarray(L, dtype=float),
               jnp.asarray(m, dtype=jnp.complex128),
               jnp.asarray(z0, dtype=jnp.complex128),
               jnp.asarray(u_seq, dtype=float),
               jnp.float64(K0), jnp.asarray(eps_vec, dtype=float),
               jnp.asarray(Omega_vec, dtype=float), jnp.asarray(phi_vec, dtype=float),
               jnp.float64(beta), jnp.float64(lam), jnp.float64(t0),
               jnp.float64(h), n_sub)
    jax.block_until_ready(zs)
    return np.asarray(ts), np.asarray(zs)


def integrate_corotating_input_comb_batch(omegas, Ls, ms, z0s, u_seqs, K0s, eps_vecs,
                                          Omega_vecs, phi_vecs, betas, dt_in, n_sub,
                                          t0: float = 0.0, lam: float = LAM_DEFAULT,
                                          return_ts: bool = True):
    """vmap-batched comb-drive input integrator over B trajectories. Shapes: omegas (B,N),
    Ls (B,N,N), ms (B,N) complex, z0s (B,N) complex, u_seqs (B,L_samp); K0s/betas (B,);
    eps_vecs/Omega_vecs/phi_vecs (B,M) -- one comb per trajectory (M shared; pad inert
    tones with core.nulls.pad_comb). dt_in, n_sub shared. Returns (ts (L_samp,),
    zs (B, L_samp, N)) host arrays. Batching is the throughput path (WSL2 launch-bound)."""
    n_sub = int(n_sub)
    h = float(dt_in) / n_sub

    def one(omega, L, m, z0, u_seq, K0, eps_vec, Omega_vec, phi_vec, beta):
        return _integrate_one_input_comb(omega, L, m, z0, u_seq, K0, eps_vec, Omega_vec,
                                         phi_vec, beta, lam, t0, jnp.float64(h), n_sub)

    f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
    ts, zs = f(jnp.asarray(omegas, dtype=float), jnp.asarray(Ls, dtype=float),
               jnp.asarray(ms, dtype=jnp.complex128),
               jnp.asarray(z0s, dtype=jnp.complex128),
               jnp.asarray(u_seqs, dtype=float),
               jnp.asarray(K0s, dtype=float), jnp.asarray(eps_vecs, dtype=float),
               jnp.asarray(Omega_vecs, dtype=float), jnp.asarray(phi_vecs, dtype=float),
               jnp.asarray(betas, dtype=float))
    jax.block_until_ready(zs)
    if return_ts:
        return np.asarray(ts[0]), np.asarray(zs)
    return np.asarray(zs)


# --------------------------------------------------------------------------- #
# Linearised variational propagation + Floquet (validation 5)
# --------------------------------------------------------------------------- #
def co_generator_origin(t, V, omega, L, K0, eps, Omega, lam):
    """In-frame linear generator about the ORIGIN applied to a matrix/vector V:

        Gtilde(t) V = lam V  -  K(t) * e^{-i omega t} (x) ( L @ (e^{i omega t} (x) V) ).

    This is the co-rotating image of 0.1's G(t) = D - K(t)L linearised about z=0: the
    change of variables v = R^{-1} z gives  v' = [lam I - K(t) Ltilde(t)] v  (the i omega
    in D cancels exactly, leaving lam I). It is the origin-linearised case 0.1 validates.
    For C's driven-limit-cycle Jacobian, swap this closure for one that adds the
    -(2|v*|^2 V + v*^2 conj(V)) saturation terms along the trajectory v*(t).

    V may be (N,) or (N,k)/(N,N); phases broadcast over columns.
    """
    K = K0 + eps * jnp.cos(Omega * t)
    rot = jnp.exp(1j * omega * t)
    if V.ndim == 2:
        rot_c = rot[:, None]
    else:
        rot_c = rot
    return lam * V - K * (jnp.conj(rot_c) * (L @ (rot_c * V)))


def _rk4_step_mat(V, t, h, gen: Callable, args):
    """RK4 step for the LINEAR matrix ODE V' = gen(t, V; args)."""
    k1 = gen(t, V, *args)
    k2 = gen(t + 0.5 * h, V + 0.5 * h * k1, *args)
    k3 = gen(t + 0.5 * h, V + 0.5 * h * k2, *args)
    k4 = gen(t + h, V + h * k3, *args)
    return V + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def monodromy_corotating(omega, L, K0, eps, Omega, lam=LAM_DEFAULT,
                         n_steps_per_period: int = 4000, n_periods: int = 1,
                         m_reorth: int = 1, gen: Callable = co_generator_origin):
    """One-period (or many-period) monodromy of the linearised driven system, computed
    IN the co-rotating frame with Wolf-style QR-renorm, then mapped back to the lab frame.

    The in-frame fundamental matrix obeys  Vtilde' = Gtilde(t) Vtilde,  Vtilde(0)=I.
    Because z = R(t) v with R(t)=diag(e^{i omega t}) UNITARY, growth is frame-identical
    (|z_i|=|v_i|), so the QR log-growth (real parts of the Floquet exponents / Lyapunov
    spectrum) is the SAME in both frames. The lab-frame one-period monodromy is recovered
    exactly by  M_lab = R(T) @ M_frame  (R(0)=I), so its eigenvalues (the Floquet
    multipliers) match 0.1's monodromy_qr.

    Returns (M_lab, log_growth):
      M_lab      : (N,N) reconstructed lab-frame map over n_periods*T (may overflow on a
                   growing case over many periods -- log_growth is the safe output).
      log_growth : (N,) Wolf accumulator; log_growth/(n_periods*T) = Floquet real parts.

    NumPy host loop (N <= ~30 for 0.1's cases; the QR-in-loop is trivial there and keeps
    the reference obviously-correct). The JAX/vmap version for C's N=200 scale reuses the
    same gen closure inside lax.scan -- left for the C build where its cost is measured.
    """
    omega = np.asarray(omega, dtype=float)
    L = np.asarray(L, dtype=float)
    N = omega.shape[0]
    T = 2.0 * np.pi / Omega
    h = T / n_steps_per_period
    total = n_steps_per_period * n_periods

    g_omega = jnp.asarray(omega)
    g_L = jnp.asarray(L)
    args = (g_omega, g_L, jnp.float64(K0), jnp.float64(eps), jnp.float64(Omega),
            jnp.float64(lam))

    @jax.jit
    def step_block(V, t0_block):
        def body(c, _):
            V, t = c
            V = _rk4_step_mat(V, t, jnp.float64(h), gen, args)
            return (V, t + h), None
        (V, _), _ = jax.lax.scan(body, (V, t0_block), None, length=m_reorth)
        return V

    Q = jnp.eye(N, dtype=jnp.complex128)
    log_growth = np.zeros(N)
    R_factors = []
    t = 0.0
    n_blocks = total // m_reorth
    for b in range(n_blocks):
        raw = step_block(Q, jnp.float64(t))     # propagate the carried frame
        raw = np.asarray(jax.block_until_ready(raw))
        Qn, R = np.linalg.qr(raw)
        d = np.diag(R)
        ph = d / np.abs(d)                       # gauge: force diag(R) real positive
        Qn = Qn * ph[np.newaxis, :]
        R = ph.conj()[:, np.newaxis] * R
        log_growth += np.log(np.abs(np.diag(R)))
        R_factors.append(R)
        Q = jnp.asarray(Qn)
        t += m_reorth * h

    with np.errstate(over="ignore", invalid="ignore"):
        R_tot = np.eye(N, dtype=complex)
        for R in R_factors:
            R_tot = R @ R_tot
        M_frame = np.asarray(Q) @ R_tot
        RT = np.diag(np.exp(1j * omega * (n_periods * T)))   # R(n_periods*T)
        M_lab = RT @ M_frame
    return M_lab, log_growth


# --------------------------------------------------------------------------- #
# Driven-limit-cycle Floquet diagnostic (C-build prerequisite 2)
# --------------------------------------------------------------------------- #
# C's Floquet EDGE must linearise about the DRIVEN LIMIT CYCLE, not the origin.
# 0.1 found the origin-linearised system is HOLOMORPHIC (no z<->zbar channel), so it
# shows only the COMBINATION resonance (Omega ~ |omega_i-omega_j|) and is BLIND to the
# 2 wbar parametric (Mathieu) tongue. The full nonlinear field linearised about a
# NONZERO reference v*(t) has the Wirtinger anti-holomorphic term, which opens the
# z<->zbar channel and lets the parametric tongue appear. Linearising about the origin
# would silently MISS half the bifurcation structure C is built to predict (verified:
# a non-rotating fixed-point reference gives |mult|=1.0000 flat in Omega -> a confident
# false negative). DECISIONS (Jason): operating-point reference = Option B (relaxed
# in-frame attractor over one drive period, with the window-to-window multiplier spread
# REPORTED as the T-periodicity health check); Option A (frozen fixed point) kept only
# as a BLIND NULL/cross-check; the unit test uses Option C (a commensurate phase-locked
# small case with exact T-periodicity).
#
# Derivation (per-oscillator frame, delta z = R(t) delta v, R=diag(e^{i omega t})):
#   lab Wirtinger:  d/dt delta z = A(t) delta z + B(t) conj(delta z),
#       A = diag(lam + i omega - 2|z*|^2) - K(t) L,   B = diag(-z*^2).
#   In the frame the i omega and the e^{2 i omega t} of z*^2 cancel exactly, leaving
#       d/dt delta v = Gtil_lc(t) delta v  -  diag(v*^2) conj(delta v),
#       Gtil_lc(t) = diag(lam - 2|v*|^2) - K(t) Ltilde(t),  Ltilde = R^{-1} L R.
#   v*=0 recovers the origin generator lam I - K Ltilde (co_generator_origin) -- the
#   built-in consistency check (monodromy_limitcycle with v*=0 == origin monodromy).
# Because of the conj(delta v) term the system is REAL-linear, so the monodromy is
# computed in the real-2N representation [Re delta v; Im delta v].


def limitcycle_generator_real(t, vstar, omega, L, K0, eps, Omega, lam):
    """Real-2N generator J(t) (2N x 2N) of the driven-limit-cycle variational system,
    linearised about the in-frame reference `vstar` (N,) complex at time `t`.

    From  d/dt delta v = G delta v + diag(d) conj(delta v),
        G = diag(lam - 2|vstar|^2) - K(t) Ltilde(t)   (complex N x N),
        d = -vstar^2                                   (complex N, conjugate channel),
    write delta v = a + i b. With G = Gr + i Gi and d = dr + i di:
        a' = (Gr + diag(dr)) a + (-Gi + diag(di)) b
        b' = (Gi + diag(di)) a + ( Gr - diag(dr)) b
    so J = [[Gr+Dr, Di-Gi], [Gi+Di, Gr-Dr]], Dr=diag(dr), Di=diag(di). v*=0 gives the
    holomorphic origin block [[Gr,-Gi],[Gi,Gr]] (real form of lam I - K Ltilde).
    """
    omega = np.asarray(omega, dtype=float)
    L = np.asarray(L, dtype=float)
    vstar = np.asarray(vstar, dtype=complex)
    N = omega.shape[0]
    K = K0 + eps * np.cos(Omega * t)
    rot = np.exp(1j * omega * t)
    Ltil = np.conj(rot)[:, None] * L * rot[None, :]      # R^{-1} L R
    G = np.diag(lam - 2.0 * np.abs(vstar) ** 2) - K * Ltil
    d = -vstar ** 2
    Gr, Gi = G.real, G.imag
    Dr, Di = np.diag(d.real), np.diag(d.imag)
    top = np.hstack([Gr + Dr, Di - Gi])
    bot = np.hstack([Gi + Di, Gr - Dr])
    return np.vstack([top, bot])


def _rk4_real_step(V, t, h, vstar0, vstarh, vstar1, omega, L, K0, eps, Omega, lam):
    """One RK4 step of the real-2N matrix ODE V' = J(t) V over [t, t+h]. The reference
    is supplied at the substep ENDPOINTS and MIDPOINT (vstar0=v*(t), vstarh=v*(t+h/2),
    vstar1=v*(t+h)) so the linearisation is about the true trajectory at each stage."""
    J0 = limitcycle_generator_real(t, vstar0, omega, L, K0, eps, Omega, lam)
    Jh = limitcycle_generator_real(t + 0.5 * h, vstarh, omega, L, K0, eps, Omega, lam)
    J1 = limitcycle_generator_real(t + h, vstar1, omega, L, K0, eps, Omega, lam)
    k1 = J0 @ V
    k2 = Jh @ (V + 0.5 * h * k1)
    k3 = Jh @ (V + 0.5 * h * k2)
    k4 = J1 @ (V + h * k3)
    return V + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _real_rotation(omega, tau):
    """Real-2N representation of the complex diagonal R(tau)=diag(e^{i omega_i tau}),
    acting on [Re v; Im v]: per oscillator the 2x2 rotation [[cos,-sin],[sin,cos]],
    assembled in the [a; b] block ordering as [[C, -S],[S, C]]. Orthogonal (a rotation),
    so it preserves growth -- it only rotates the individual Floquet multipliers."""
    omega = np.asarray(omega, dtype=float)
    C = np.diag(np.cos(omega * tau))
    S = np.diag(np.sin(omega * tau))
    return np.block([[C, -S], [S, C]])


def monodromy_limitcycle(omega, L, K0, eps, Omega, lam, vstar_half, t0_window,
                         n_spp, m_reorth: int = 8):
    """One-period LAB-frame monodromy of the driven-limit-cycle variational system,
    linearised about the supplied reference, with Wolf QR-renorm.

    The variational equation is integrated in the co-rotating frame (slow; real-2N
    because of the conj(delta v) channel), then mapped to the LAB frame -- the Floquet
    MULTIPLIERS are eigenvalues of the lab monodromy, not the frame map. With
    delta z = R(t) delta v, R(t)=diag(e^{i omega t}), the lab monodromy over the window
    [t0, t0+T] is  M_lab = R(t0+T) M_frame R(t0)^{-1}  (t0-independent eigenvalues, the
    gauge invariance of Floquet multipliers). This mirrors monodromy_corotating's
    M_lab = R(T) M_frame. The rotations are orthogonal, so the REAL Floquet exponents
    (log_growth/T) are frame-identical.

    vstar_half : (2*n_spp+1, N) complex reference v*(t) sampled at the HALF grid
                 t0_window + j*(T/(2 n_spp)), j=0..2 n_spp (endpoints + midpoints of the
                 n_spp RK4 substeps that tile one drive period T=2*pi/Omega).
    t0_window  : absolute start time of the window (K(t), Ltilde(t) use absolute t).

    Returns (M_lab, log_growth, mult):
      M_lab      : (2N, 2N) real lab-frame one-period monodromy.
      log_growth : (2N,) Wolf accumulator; log_growth/T = real Floquet exponents.
      mult       : (2N,) eigenvalues of M_lab (the Floquet multipliers; complex-conjugate
                   pairs in the real representation). |mult|max=1 is the bifurcation edge.
    """
    omega = np.asarray(omega, dtype=float)
    vstar_half = np.asarray(vstar_half, dtype=complex)
    N = omega.shape[0]
    T = 2.0 * np.pi / Omega
    h = T / n_spp
    if vstar_half.shape[0] != 2 * n_spp + 1:
        raise ValueError(f"vstar_half must be (2*n_spp+1, N); got {vstar_half.shape}")

    Q = np.eye(2 * N)
    log_growth = np.zeros(2 * N)
    R_factors = []
    block = np.eye(2 * N)
    in_block = 0
    for k in range(n_spp):
        t_k = t0_window + k * h
        # accumulate this substep's propagator into the carried block (reorth below)
        block = _rk4_real_step(block, t_k, h, vstar_half[2 * k], vstar_half[2 * k + 1],
                               vstar_half[2 * k + 2], omega, L, K0, eps, Omega, lam)
        in_block += 1
        if in_block == m_reorth or k == n_spp - 1:
            raw = block @ Q
            Q, R = np.linalg.qr(raw)
            d = np.diag(R)
            ph = np.sign(d)
            ph[ph == 0] = 1.0
            Q = Q * ph[np.newaxis, :]
            R = ph[:, np.newaxis] * R
            log_growth += np.log(np.abs(np.diag(R)))
            R_factors.append(R)
            block = np.eye(2 * N)
            in_block = 0
    with np.errstate(over="ignore", invalid="ignore"):
        R_tot = np.eye(2 * N)
        for R in R_factors:
            R_tot = R @ R_tot
        M_frame = Q @ R_tot
        # Map frame -> lab: M_lab = R(t0+T) M_frame R(t0)^{-1} (R orthogonal: inv = T).
        RtT = _real_rotation(omega, t0_window + T)
        Rt0 = _real_rotation(omega, t0_window)
        M_lab = RtT @ M_frame @ Rt0.T
    mult = np.linalg.eigvals(M_lab)
    return M_lab, log_growth, mult


def leading_multiplier(mult) -> float:
    """Leading Floquet multiplier magnitude max|mult| (the bifurcation-edge indicator;
    crosses 1 at the driven-limit-cycle edge)."""
    return float(np.max(np.abs(np.asarray(mult))))


def driven_reference(omega, L, K0, eps, Omega, lam=LAM_DEFAULT, z0=None,
                     n_relax: int = 60, n_windows: int = 1, n_spp: int = 400):
    """Compute the in-frame driven reference v*(t) (Option B: the relaxed nonlinear
    attractor) on the half grid over `n_windows` consecutive drive periods.

    Procedure: integrate the BARE driven field (co_field; NO input) from an on-shell
    state for `n_relax` periods to reach the attractor, then record v*(t) on the half
    grid (resolution T/(2 n_spp)) over the next `n_windows` periods. The drive phase /
    absolute time is preserved throughout (K(t), Ltilde(t) need absolute t).

    Returns dict:
      windows   : list of (2*n_spp+1, N) complex half-grid references, one per window.
      t0        : list of absolute window start times (n_relax+w)*T.
      omega,... : echoed for convenience.
      lock_rate : (N,) measured frame-rotation rate of v* (d arg v*/dt) over window 0 --
                  |lock_rate|>0 means the oscillator is PHASE-LOCKED away from its own
                  omega (v* rotates in the frame), which is what opens the parametric
                  channel. ~0 means a frozen fixed point (Option-A-like, blind).
      amp       : (N,) mean |v*| over window 0.
    """
    omega = np.asarray(omega, dtype=float)
    N = omega.shape[0]
    T = 2.0 * np.pi / Omega
    hh = T / (2 * n_spp)                         # half-grid step
    if z0 is None:
        z0 = np.sqrt(lam) * np.ones(N, dtype=complex)
    z0 = np.asarray(z0, dtype=complex)

    total_half = (n_relax + n_windows) * 2 * n_spp
    ts, zs = integrate_corotating(omega, L, z0, K0, eps, Omega, t0=0.0, h=hh,
                                  n_save=total_half, save_every=1, lam=lam)
    # v at the saved points t=(k+1)hh; prepend v(0)=z0 to get the j=0..total_half grid.
    v_saved = zs * np.exp(-1j * omega[None, :] * ts[:, None])
    v_half = np.vstack([z0[None, :], v_saved])    # (total_half+1, N), v_half[j]=v(j*hh)

    windows, t0s = [], []
    for w in range(n_windows):
        start = (n_relax + w) * 2 * n_spp         # half-grid index of window start
        windows.append(v_half[start:start + 2 * n_spp + 1])
        t0s.append((n_relax + w) * T)

    # lock diagnostic over window 0 (endpoints, step h=2*hh between substep endpoints)
    w0 = windows[0]
    h = T / n_spp
    phase = np.unwrap(np.angle(w0[::2]), axis=0)   # at substep endpoints
    lock_rate = (phase[-1] - phase[0]) / (n_spp * h)
    amp = np.abs(w0[::2]).mean(axis=0)
    return dict(windows=windows, t0=t0s, omega=omega, L=np.asarray(L), K0=K0, eps=eps,
                Omega=Omega, lam=lam, n_spp=n_spp, lock_rate=lock_rate, amp=amp)


def frozen_reference(vstar_point, n_spp: int):
    """Option-A blind-null reference: a single in-frame state `vstar_point` (N,) FROZEN
    (constant) across the half grid (2*n_spp+1, N). Used as a cross-check -- a
    non-rotating fixed point has no z<->zbar oscillation, so its monodromy is BLIND to
    the parametric tongue (|mult| flat in Omega); structure that appears under the
    relaxed reference but not here is genuinely parametric, not a coupling-sweep
    artifact."""
    vstar_point = np.asarray(vstar_point, dtype=complex)
    return np.repeat(vstar_point[None, :], 2 * n_spp + 1, axis=0)


# --------------------------------------------------------------------------- #
# Additive-noise Euler-Maruyama in the frame (validation 7)
# --------------------------------------------------------------------------- #
def integrate_corotating_sde(omega, L, z0, K0, eps, Omega, t0, h, n_save, save_every,
                             sigma, key, lam: float = LAM_DEFAULT):
    """Additive-noise trajectory in the co-rotating frame via Euler-Maruyama.

    The lab-frame noise sigma*xi_i (circular complex white) becomes sigma*xi_i*e^{-i omega t}
    in the frame; circular complex noise is rotation-invariant in distribution, so the
    increment is still circular complex with the SAME sigma -- the phase factor can be
    dropped for the DISTRIBUTION of the increment. We add sqrt(h)*sigma*(N(0,1)+iN(0,1))/sqrt(2)
    per step (unit per-component variance sigma^2). Drift is the co_field; additive noise
    => Ito=Stratonovich so plain Euler-Maruyama is correct (no Milstein).

    The point (validation 7): the frame removes the fast i omega from the DRIFT, so the
    Euler base order no longer needs ~256 steps/fast-period (A's finding in the lab frame)
    -- the step is set by the slow remainder. Returns (ts, zs) host arrays (lab frame).
    """
    omega = jnp.asarray(omega, dtype=float)
    L = jnp.asarray(L, dtype=float)
    N = omega.shape[0]
    v0 = jnp.asarray(z0, dtype=jnp.complex128) * jnp.exp(-1j * omega * t0)
    sh = jnp.sqrt(jnp.float64(h))
    total = int(n_save) * int(save_every)
    keys = jax.random.split(key, total)

    def body(carry, k):
        v, t = carry
        v = rk4_step(v, t, jnp.float64(h), omega, L, jnp.float64(K0),
                     jnp.float64(eps), jnp.float64(Omega), jnp.float64(lam))
        kr, ki = jax.random.split(k)
        dW = (jax.random.normal(kr, (N,)) + 1j * jax.random.normal(ki, (N,))) / jnp.sqrt(2.0)
        v = v + sigma * sh * dW
        return (v, t + jnp.float64(h)), v

    @jax.jit
    def run():
        (_, _), vs = jax.lax.scan(body, (v0, jnp.float64(t0)), keys)
        return vs

    vs = jax.block_until_ready(run())
    vs = np.asarray(vs)
    ts = t0 + (np.arange(total) + 1.0) * h
    zs = vs * np.exp(1j * np.asarray(omega)[None, :] * ts[:, None])
    # subsample to the save grid (every save_every steps)
    sel = (np.arange(int(n_save)) + 1) * int(save_every) - 1
    return ts[sel], zs[sel]
