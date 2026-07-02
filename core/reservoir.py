"""
core/reservoir.py
=================

Batched Stuart-Landau reservoir harness for Experiment A (the GATE-3 deliverable;
reused by B/C). AUTONOMOUS baseline -- there is NO slow parametric drive here (the
drive is Experiment C). The only time dependence is the piecewise-constant input
u(t); coupling K is a constant sweep parameter (eps=0).

System (spec A; deterministic primary, sigma=0):

    zdot_i = (lambda + i omega_i - |z_i|^2) z_i  -  K (L z)_i  +  beta u(t) m_i
             [ + sigma xi_i(t)  -- ABLATION only, see run_reservoir_sde ]

Design decisions carried from Experiment 0 (asserted / documented here):

* OPERATING POINT (0.1/0.2): the coupled origin G0 = D - K L must be UNSTABLE
  (max Re eig > 0) or the state collapses to z=0; and (omega-set, K) must be away
  from an EXCEPTIONAL POINT (eigenvalue gap of G0 well above 0). Both are real
  failure modes; `assert_operating_point` bakes them in and the sweep skips/flags
  violators. Mean graph degree is held ~constant across N via p = mean_deg/(N-1)
  (fixed p over-connects large N and damps the origin stable).

* INPUT TIMESCALE / STROBOSCOPIC SAMPLING (0.3 aliasing lesson, GATE 1): input is
  held piecewise-constant over dt_in and the state is sampled on the SAME
  stroboscopic grid. dt_in is set to resolve the FAST band: ~`spp` (default 8)
  samples per fastest oscillator period (Nyquist). `dt_in_for` computes it. The
  KNOWN TENSION A studies: one input rate cannot serve a 4-decade span -- resolving
  the fast band starves the slow band of periods for fixed sample budget. This is
  predictions 4/5 and the GATE-3 4-decade compute fork.

* INTEGRATION (0.2): diffrax Dopri8, float64, adaptive PID. Trajectories are
  vmap-batched into ONE diffeqsolve (the GPU win is batching, not single-traj
  speed: B~30 ~ B=1 wall). Metrics computed on host numpy.

* NOISE (GATE 2, deferred from 0.2): sigma=0 is the PRIMARY baseline; sigma=0.01 is
  an ABLATION. The noise is ADDITIVE (sigma independent of z) -> Ito = Stratonovich,
  Euler-Maruyama suffices (NO high-strong-order solver). Complex xi -> real 2N
  Brownian (Re, Im independent). `run_reservoir_sde` implements it in the real-2N
  representation; validated on one small case at GATE 2 before any noise sweep.
"""
from __future__ import annotations

from typing import NamedTuple
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import diffrax

from core.stuart_landau import SLParams
from core.magnus import graph_laplacian, diagonal_block

__all__ = [
    "dt_in_for",
    "washout_samples",
    "build_graph",
    "build_encoding",
    "build_system",
    "assert_operating_point",
    "run_reservoir",
    "run_reservoir_grid",
    "run_reservoir_sde",
    "ResSpec",
]

LAM = 0.1                       # bifurcation parameter (fixed across A, spec)


# --------------------------------------------------------------------------- #
# Sampling / washout timescales (GATE 1)
# --------------------------------------------------------------------------- #
def dt_in_for(omega_max: float, spp: int = 8) -> float:
    """Input-hold / stroboscopic-sample step resolving the FAST band: `spp`
    samples per fastest oscillator period (Nyquist, spec >= ~8). The fastest
    period is 2*pi/omega_max, so dt_in = 2*pi/(omega_max*spp)."""
    return 2.0 * np.pi / (omega_max * spp)


def washout_samples(omega_min: float, omega_max: float, dt_in: float,
                    n_slow: int = 5, n_fast: int = 100) -> int:
    """Washout length in SAMPLES = (n_slow slow periods + n_fast fast periods)
    / dt_in (spec: 5 slow + 100 fast)."""
    t_wash = n_slow * (2 * np.pi / omega_min) + n_fast * (2 * np.pi / omega_max)
    return int(np.ceil(t_wash / dt_in))


# --------------------------------------------------------------------------- #
# Graph topologies (baseline ER + ablation variants)
# --------------------------------------------------------------------------- #
def build_graph(kind: str, N: int, mean_deg: float, rng: np.random.Generator,
                ws_p: float = 0.1, n_modules: int = 4,
                inter_frac: float = 0.05) -> np.ndarray:
    """Symmetric 0/1 adjacency (no self-loops) at ~`mean_deg` average degree.

    kind = 'er'  : Erdos-Renyi (baseline), edge prob p = mean_deg/(N-1).
    kind = 'ws'  : Watts-Strogatz small-world -- ring lattice of degree mean_deg,
                   each edge rewired with prob ws_p.
    kind = 'modular' : `n_modules` block-diagonal ER clusters (dense intra) plus
                   sparse inter-module edges (fraction `inter_frac` of intra prob).
    """
    A = np.zeros((N, N))
    if kind == "er":
        p = mean_deg / (N - 1)
        # Threshold FIRST then take the strict upper triangle (thresholding a
        # triu'd matrix would count the zeroed lower triangle as edges).
        E = np.triu((rng.uniform(size=(N, N)) < p).astype(float), 1)
        A = E + E.T
    elif kind == "ws":
        k = int(round(mean_deg))
        k -= k % 2  # even: k/2 neighbours each side
        for i in range(N):
            for j in range(1, k // 2 + 1):
                A[i, (i + j) % N] = 1.0
                A[(i + j) % N, i] = 1.0
        # rewire
        for i in range(N):
            for j in range(1, k // 2 + 1):
                if rng.uniform() < ws_p:
                    nb = (i + j) % N
                    A[i, nb] = A[nb, i] = 0.0
                    t = rng.integers(N)
                    while t == i or A[i, t]:
                        t = rng.integers(N)
                    A[i, t] = A[t, i] = 1.0
    elif kind == "modular":
        sizes = [N // n_modules] * n_modules
        sizes[-1] += N - sum(sizes)
        p_intra = mean_deg / (max(sizes[0], 2) - 1)
        p_inter = inter_frac * p_intra
        # Sparse inter-module background, dense intra-module blocks; threshold
        # then take the strict upper triangle (same fix as the ER branch).
        base = (rng.uniform(size=(N, N)) < p_inter).astype(float)
        start = 0
        for s in sizes:
            blk = slice(start, start + s)
            base[blk, blk] = (rng.uniform(size=(s, s)) < p_intra).astype(float)
            start += s
        E = np.triu(base, 1)
        A = E + E.T
    else:
        raise ValueError(f"unknown graph kind {kind!r}")
    np.fill_diagonal(A, 0.0)
    return A


# --------------------------------------------------------------------------- #
# Input encodings (baseline random + ablation variants)
# --------------------------------------------------------------------------- #
def build_encoding(kind: str, omega: np.ndarray, rng: np.random.Generator,
                   span_decades: float) -> np.ndarray:
    """Complex input-projection vector m_i (length N).

    kind = 'random'    : iid complex normal (baseline).
    kind = 'band'      : frequency-band-aware -- independent random projection per
                         DECADE, each band L2-normalised to equal injected power
                         (prediction 5: random under-drives the fast band at wide
                         spans; equal-power-per-band compensates).
    kind = 'flat'      : spectrally-flat white mask -- random +-1 +- i signs
                         (photonic-RC binary-mask convention).
    """
    N = len(omega)
    if kind == "random":
        m = rng.normal(size=N) + 1j * rng.normal(size=N)
    elif kind == "band":
        m = rng.normal(size=N) + 1j * rng.normal(size=N)
        # band index by decade of omega (log10, since omega_min ~ 1)
        band = np.clip(np.floor(np.log10(omega + 1e-12)).astype(int), 0,
                       int(np.ceil(span_decades)))
        for b in np.unique(band):
            mask = band == b
            nrm = np.sqrt(np.sum(np.abs(m[mask]) ** 2)) + 1e-12
            m[mask] = m[mask] / nrm * np.sqrt(mask.sum())  # equal power per band
    elif kind == "flat":
        m = (rng.choice([-1.0, 1.0], size=N)
             + 1j * rng.choice([-1.0, 1.0], size=N)) / np.sqrt(2.0)
    else:
        raise ValueError(f"unknown encoding kind {kind!r}")
    return m


# --------------------------------------------------------------------------- #
# System builder + operating-point guards
# --------------------------------------------------------------------------- #
class ResSpec(NamedTuple):
    """One reservoir instance (host numpy)."""
    omega: np.ndarray          # (N,) natural frequencies
    L: np.ndarray              # (N, N) graph Laplacian
    m: np.ndarray              # (N,) complex input projection
    z0: np.ndarray             # (N,) complex initial state (on limit cycle)


def build_system(seed: int, N: int, span_decades: float, mean_deg: float = 10.0,
                 graph: str = "er", encoding: str = "random") -> ResSpec:
    """Build one reservoir: log-uniform omega over [1, 10^span], graph Laplacian,
    input projection, on-limit-cycle random-phase initial state."""
    rng = np.random.default_rng(seed)
    omega = np.logspace(0.0, span_decades, N)
    A = build_graph(graph, N, mean_deg, rng)
    L = graph_laplacian(A)
    m = build_encoding(encoding, omega, rng, span_decades)
    z0 = np.sqrt(LAM) * np.exp(1j * rng.uniform(0, 2 * np.pi, N))
    return ResSpec(omega, L, m, z0)


def assert_operating_point(spec: ResSpec, K: float, kappa_max: float = 1e3):
    """Bake in the 0.1/0.2 operating-point constraints. Returns (max_re, kappa).
    Raises AssertionError on violation; the sweep catches and flags the point.

    EP guard -- METHODS DECISION (GATE, Experiment A): 0.1 used the raw eigenvalue
    GAP of G0 as the exceptional-point proxy (disc(G0) not near 0). That proxy
    FAILS on a multi-decade frequency COMB: log-uniform omega pack the eigenvalues
    of the near-diagonal G0 densely (min gap ~ omega-spacing, shrinking with N and
    narrowing span), so the gap-proxy flags dense-but-perfectly-diagonalisable
    points as 'near-EP'. An exceptional point is where EIGENVECTORS coalesce
    (defectiveness), measured by the eigenvector condition number kappa(V) ->
    infinity at an EP. Here kappa(V) ~ 1-3 across all spans/N (verified), so we
    guard on kappa(V) < kappa_max instead of the raw gap.
    """
    G0 = diagonal_block(LAM, spec.omega) - K * spec.L
    ev, V = np.linalg.eig(G0)
    max_re = float(ev.real.max())
    kappa = float(np.linalg.cond(V))
    assert max_re > 0, f"origin STABLE (max Re={max_re:.4f}) -> state collapses"
    assert kappa < kappa_max, f"near EXCEPTIONAL POINT (kappa(V)={kappa:.1f})"
    return max_re, kappa


# --------------------------------------------------------------------------- #
# Deterministic batched runner (GATE 3) -- diffrax Dopri8, vmap over batch
# --------------------------------------------------------------------------- #
def _det_field(t, z, args):
    """SL + piecewise-constant input. No slow drive (K constant = K0)."""
    p, u_seq, dt_in, m, beta = args
    idx = jnp.clip(jnp.floor(t / dt_in).astype(jnp.int32), 0, u_seq.shape[0] - 1)
    inp = beta * u_seq[idx] * m
    return (p.lam + 1j * p.omega - jnp.abs(z) ** 2) * z - p.K0 * (p.L @ z) + inp


@lru_cache(maxsize=None)
def _build_det_solver(L_samp: int, rtol: float, atol: float):
    """Build + JIT a batched solver for a fixed (L_samp, rtol, atol). Cached so the
    sweep compiles ONCE per span (not per K/beta/seed point): K, beta, dt_in are
    TRACED runtime args (changing them does NOT recompile); only L_samp/rtol/atol
    (static) and the batch shapes key the compilation. Without this every
    run_reservoir call rebuilt a fresh jit(lambda) -> a recompile per sweep point
    (~hundreds of compiles, the GATE-1 flat-timing artifact)."""
    def _one(omega, L, z0, u_seq, m, K, beta, dt_in):
        p = SLParams(lam=LAM, omega=omega, L=L, K0=K, eps=0.0, Omega=0.0)
        t1 = L_samp * dt_in
        ts = (jnp.arange(L_samp) + 1.0) * dt_in
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_det_field), diffrax.Dopri8(), 0.0, t1, dt_in, z0,
            args=(p, u_seq, dt_in, m, beta),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts), max_steps=50_000_000)
        return sol.ys

    @jax.jit
    def solver(om, Ls, z0, uu, mm, K, beta, dt_in):
        return jax.vmap(_one, in_axes=(0, 0, 0, 0, 0, None, None, None))(
            om, Ls, z0, uu, mm, K, beta, dt_in)
    return solver


def run_reservoir(specs, u_seqs, K, beta, dt_in, rtol=1e-7, atol=1e-9):
    """Batched deterministic reservoir run over a list of (ResSpec, input) pairs.

    specs   : list of ResSpec (all SAME N) -- the batch (seeds x params)
    u_seqs  : list of (L,) input arrays (one per spec)
    K, beta : scalar coupling / input strength (shared across the batch)
    dt_in   : stroboscopic step (shared)
    rtol,atol : PID tolerances (0.2: 1e-7 for A's <=1e5 fast cycles; tighten for
                the 4-decade tail per the per-period drift rate)

    Returns X : (B, L, N) complex host array of state samples.
    """
    L_samp = len(u_seqs[0])
    om = jnp.asarray(np.stack([s.omega for s in specs]))
    Ls = jnp.asarray(np.stack([s.L for s in specs]))
    z0 = jnp.asarray(np.stack([s.z0 for s in specs]))
    mm = jnp.asarray(np.stack([s.m for s in specs]))
    uu = jnp.asarray(np.stack(u_seqs))
    solver = _build_det_solver(L_samp, rtol, atol)
    X = jax.block_until_ready(
        solver(om, Ls, z0, uu, mm, jnp.float64(K), jnp.float64(beta),
               jnp.float64(dt_in)))
    return np.asarray(X)


@lru_cache(maxsize=None)
def _build_grid_solver(L_samp: int, rtol: float, atol: float):
    """Like _build_det_solver but with K and beta BATCHED (per-trajectory), so a
    whole sweep grid (K x beta x seed) at one span runs in ONE vmap. GATE 3 found
    batching amortises at N=200 (B4/B1~0.9 -- still launch-bound under WSL2), so
    batching across param points, not just seeds, is ~free until VRAM limits; this
    is what makes the spans-1..3 stage-1 sweep ~minutes instead of ~hours."""
    def _one(omega, L, z0, u_seq, m, K, beta, dt_in):
        p = SLParams(lam=LAM, omega=omega, L=L, K0=K, eps=0.0, Omega=0.0)
        t1 = L_samp * dt_in
        ts = (jnp.arange(L_samp) + 1.0) * dt_in
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_det_field), diffrax.Dopri8(), 0.0, t1, dt_in, z0,
            args=(p, u_seq, dt_in, m, beta),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts), max_steps=50_000_000)
        return sol.ys

    @jax.jit
    def solver(om, Ls, z0, uu, mm, Ks, betas, dt_in):
        return jax.vmap(_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
            om, Ls, z0, uu, mm, Ks, betas, dt_in)
    return solver


def run_reservoir_grid(specs, u_seqs, Ks, betas, dt_in, rtol=1e-7, atol=1e-9,
                       chunk=48, reduce_fn=None):
    """Batched run with PER-TRAJECTORY coupling Ks and input strength betas.

    specs, u_seqs : length-B lists (one per trajectory)
    Ks, betas     : length-B arrays (per-trajectory K and beta)
    dt_in         : shared stroboscopic step (one span)
    chunk         : max trajectories per vmap call (VRAM bound; the amortised
                    sweet spot is "as large as fits"). Chunks loop on host.
    reduce_fn     : optional callback (X_chunk, start_index) -> list of per-
                    trajectory results. If given, each chunk's raw X is reduced
                    and DISCARDED (the full (B,L,N) array is never held -- needed
                    at wide spans where L~1e4-1e5 makes the full grid many GB);
                    returns the flat list of reduced results. If None, returns the
                    concatenated (B, L, N) complex host array.
    """
    B = len(specs)
    L_samp = len(u_seqs[0])
    solver = _build_grid_solver(L_samp, rtol, atol)
    outs = []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        om = jnp.asarray(np.stack([sp.omega for sp in specs[s:e]]))
        Ls = jnp.asarray(np.stack([sp.L for sp in specs[s:e]]))
        z0 = jnp.asarray(np.stack([sp.z0 for sp in specs[s:e]]))
        mm = jnp.asarray(np.stack([sp.m for sp in specs[s:e]]))
        uu = jnp.asarray(np.stack(u_seqs[s:e]))
        Kc = jnp.asarray(np.asarray(Ks[s:e], dtype=float))
        Bc = jnp.asarray(np.asarray(betas[s:e], dtype=float))
        X = jax.block_until_ready(
            solver(om, Ls, z0, uu, mm, Kc, Bc, jnp.float64(dt_in)))
        if reduce_fn is None:
            outs.append(np.asarray(X))
        else:
            outs.extend(reduce_fn(np.asarray(X), s))
    return outs if reduce_fn is not None else np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
# Stochastic runner (GATE 2) -- additive complex noise, Euler-Maruyama, real-2N
# --------------------------------------------------------------------------- #
def _drift_real(t, y, args):
    """Real-2N drift y=[Re z, Im z]: SL + input."""
    p, u_seq, dt_in, m, beta = args
    N = y.shape[0] // 2
    z = y[:N] + 1j * y[N:]
    idx = jnp.clip(jnp.floor(t / dt_in).astype(jnp.int32), 0, u_seq.shape[0] - 1)
    inp = beta * u_seq[idx] * m
    dz = (p.lam + 1j * p.omega - jnp.abs(z) ** 2) * z - p.K0 * (p.L @ z) + inp
    return jnp.concatenate([dz.real, dz.imag])


def run_reservoir_sde(specs, u_seqs, K, beta, dt_in, sigma, key,
                      n_substep: int = 4):
    """Batched additive-noise reservoir run (the GATE-2 ablation path).

    ADDITIVE complex white noise sigma*xi: in real-2N coords the diffusion is the
    constant sigma*I_{2N} driven by 2N independent Brownian motions (Re, Im
    independent). Ito = Stratonovich for additive noise, so Euler-Maruyama
    (diffrax.Euler) is correct and sufficient -- NO Milstein / high strong order.
    Fixed step dt_in/n_substep (the fast band must be resolved; an SDE solver is
    fixed-step). VirtualBrownianTree gives a reproducible, refineable path.

    Returns X : (B, L, N) complex host array.
    """
    L_samp = len(u_seqs[0])
    t1 = float(L_samp) * dt_in
    dt = dt_in / n_substep
    keys = jax.random.split(key, len(specs))

    def solve_one(omega, L, z0, u_seq, m, k):
        p = SLParams(lam=LAM, omega=omega, L=L, K0=K, eps=0.0, Omega=0.0)
        N = omega.shape[0]
        y0 = jnp.concatenate([z0.real, z0.imag])
        ts = (jnp.arange(L_samp) + 1.0) * dt_in
        bm = diffrax.VirtualBrownianTree(
            0.0, t1, tol=dt / 2.0, shape=(2 * N,), key=k)
        # Additive diagonal diffusion sigma*I: ControlTerm with a matrix vf.
        diffusion = diffrax.ControlTerm(
            lambda t, y, args: sigma * jnp.eye(2 * N), bm)
        terms = diffrax.MultiTerm(diffrax.ODETerm(_drift_real), diffusion)
        sol = diffrax.diffeqsolve(
            terms, diffrax.Euler(), 0.0, t1, dt, y0,
            args=(p, u_seq, dt_in, m, beta),
            saveat=diffrax.SaveAt(ts=ts), max_steps=None)
        Y = sol.ys
        return Y[:, :N] + 1j * Y[:, N:]

    solve = jax.jit(jax.vmap(solve_one, in_axes=(0, 0, 0, 0, 0, 0)))
    om = jnp.asarray(np.stack([s.omega for s in specs]))
    Ls = jnp.asarray(np.stack([s.L for s in specs]))
    z0 = jnp.asarray(np.stack([s.z0 for s in specs]))
    mm = jnp.asarray(np.stack([s.m for s in specs]))
    uu = jnp.asarray(np.stack(u_seqs))
    X = jax.block_until_ready(solve(om, Ls, z0, uu, mm, keys))
    return np.asarray(X)
