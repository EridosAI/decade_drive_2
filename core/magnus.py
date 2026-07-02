"""
core/magnus.py
==============

Analytical Floquet effective-generator for the *linearised* Stuart-Landau
network under a parametric coupling drive.

Conceptual setup (see spec, Experiment 0 / Conceptual Background)
----------------------------------------------------------------
The full network is

    z_i' = (lambda + i*omega_i - |z_i|^2) z_i + K(t) * sum_j A_ij (z_j - z_i) + ...

For the Floquet machinery we drop the nonlinear saturation (-|z|^2 z), noise,
and input, leaving a *linear* time-periodic system

    z' = G(t) z ,        G(t) = D - K(t) L ,
    D  = diag(lambda + i*omega_i) ,
    L  = graph Laplacian = deg(A) - A ,
    K(t) = K0 + eps*cos(Omega t).

Because  sum_j A_ij (z_j - z_i) = -(L z)_i , the coupling enters as  -K(t) L .

The one-period propagator is  M = T-ordered exp( int_0^T G dt' ) = exp(Phi),
with Phi the Magnus series. We want an *effective time-independent generator*
G_F with  M = exp(G_F * T) , i.e. G_F = Phi / T. Its eigenvalues are the
Floquet exponents; their real parts are the driven analogue of Lyapunov
exponents (edge-of-chaos -> edge-of-Floquet-regime), and bifurcations of G_F
are the predicted performance-peak boundaries in Experiment C.

Two roads to G_F, both implemented here:
  (1) closed-form high-frequency (van Vleck) expansion  -> effective_generator_hf
  (2) direct numerical quadrature of the Magnus integrals -> magnus_terms_quadrature
They are cross-checked against each other and against hand calculations in
tests/test_magnus.py (Experiment 0.4).

Why van Vleck rather than the t0=0 Floquet-Magnus form?
-------------------------------------------------------
The Floquet-Magnus effective Hamiltonian depends on the drive's starting phase
t0; Eckardt & Anisimovas (New J. Phys. 17, 093039, 2015; arXiv:1502.06477)
show this t0-dependence "can lead to artifactual symmetry breaking". For C we
predict *bifurcations* (eigenvalue crossings) and want them independent of an
arbitrary choice of where the cosine starts. The van Vleck effective generator
is t0-independent. Its eigenvalues equal those of any Floquet-Magnus form (the
Floquet multipliers are gauge-invariant), so the two agree on every quantity we
test or predict.

Key analytic results (derived in the module docstring of the tests):
  G_F^(0) = D - K0 L                         (drive average; cosine averages out)
  G_F^(1) = i * sum_{m>=1} (1/(m*Omega)) [G_m, G_{-m}]      ( == 0 for a pure cosine,
            because G_{+1} = G_{-1} = -(eps/2) L  commute )
  G_F^(2) = -(1/(2 Omega^2)) sum_{m>=1} (1/m^2) ( [[G_m,G0],G_{-m}] + [[G_{-m},G0],G_m] )
          = (eps^2 / (4 Omega^2)) [[D, L], L]            (single-cosine drive)

That last expression is exactly the spec's "order eps^2/Omega^2 correction to the
couplings plus a frequency renormalisation": [D,L]_{ij} = i(omega_i-omega_j) L_{ij},
so the correction vanishes for frequency-matched oscillators and grows with
frequency mismatch.

NOTE on scope: the closed-form path is exact for a *single-harmonic* drive
(only m = 0, +-1 Fourier modes). Multi-harmonic drives (e.g. the square-wave
null in C) bring in van Vleck cross terms (m != m'); these are NOT yet
implemented and the code raises rather than silently returning a wrong answer.

This module is plain NumPy on purpose: it is the small-dense-matrix *analytical*
layer (N up to a few hundred for the linearised network). The heavy ODE work
(stuart_landau, lyapunov/Floquet-monodromy) is where JAX/diffrax belongs.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "commutator",
    "graph_laplacian",
    "diagonal_block",
    "build_generator_modes",
    "effective_generator_hf",
    "magnus_terms_quadrature",
    "effective_generator_quadrature",
    "one_period_map_from_modes",
]


# --------------------------------------------------------------------------- #
# Basic linear-algebra helpers
# --------------------------------------------------------------------------- #
def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[A, B] = A B - B A."""
    return A @ B - B @ A


def graph_laplacian(A: np.ndarray) -> np.ndarray:
    """Combinatorial graph Laplacian L = deg(A) - A from an adjacency matrix A.

    Coupling term sum_j A_ij (z_j - z_i) equals -(L z)_i, so the linearised
    generator is G(t) = D - K(t) L.
    """
    A = np.asarray(A)
    deg = np.diag(A.sum(axis=1))
    return deg - A


def diagonal_block(lam: float, omegas: np.ndarray) -> np.ndarray:
    """D = diag(lambda + i*omega_i)."""
    omegas = np.asarray(omegas, dtype=float)
    return np.diag(lam + 1j * omegas)


# --------------------------------------------------------------------------- #
# Fourier modes of the generator  G(t) = sum_m G_m exp(i m Omega t)
# --------------------------------------------------------------------------- #
def build_generator_modes(D: np.ndarray, L: np.ndarray,
                          K0: float, eps: float,
                          phi: float = 0.0) -> dict[int, np.ndarray]:
    """Fourier modes of G(t) = D - (K0 + eps cos(Omega t + phi)) L.

    cos(Omega t + phi) = (e^{iphi} e^{iOmega t} + e^{-iphi} e^{-iOmega t})/2, so

        G_0    = D - K0 L
        G_{+1} = -(eps/2) e^{+i phi} L
        G_{-1} = -(eps/2) e^{-i phi} L

    `phi` is the drive's starting phase (equivalently a time origin t0 = phi/Omega).
    For physics it is irrelevant which phase the cosine "starts" at; the van Vleck
    effective generator (effective_generator_hf) is phi-independent by construction,
    whereas the t0=0 Floquet-Magnus generator (effective_generator_quadrature) is
    NOT. That contrast is exploited as a regression guard in tests/test_magnus.py
    and is the reason this layer uses van Vleck (see module docstring).

    Returned as {m: G_m} including only nonzero modes.
    """
    G0 = D - K0 * L
    Gp1 = -0.5 * eps * np.exp(1j * phi) * L
    Gm1 = -0.5 * eps * np.exp(-1j * phi) * L
    return {0: G0, 1: Gp1, -1: Gm1}


# --------------------------------------------------------------------------- #
# (1) Closed-form van Vleck high-frequency effective generator
# --------------------------------------------------------------------------- #
def effective_generator_hf(modes: dict[int, np.ndarray], Omega: float,
                           order: int = 2) -> np.ndarray:
    """t0-independent van Vleck effective generator G_F, to given `order` in 1/Omega.

    order=0:  G_0                                  (drive average)
    order=1:  + i sum_{m>=1} (1/(m Omega)) [G_m, G_{-m}]
    order=2:  - (1/(2 Omega^2)) sum_{m>=1} (1/m^2) ( [[G_m,G0],G_{-m}]
                                                     + [[G_{-m},G0],G_m] )

    Exact for single-harmonic drive (modes subset of {-1,0,+1}). For drives with
    higher harmonics, order>=2 also needs van Vleck cross terms (m != m'); those
    are not implemented and this function raises if such modes are present.
    """
    if 0 not in modes:
        raise ValueError("modes must contain the m=0 (average) component")
    G0 = modes[0]
    GF = G0.astype(complex).copy()

    pos = sorted(m for m in modes if m > 0)
    if order >= 1:
        for m in pos:
            Gm = modes[m]
            Gmm = modes.get(-m)
            if Gmm is None:
                raise ValueError(f"missing conjugate mode -{m}")
            GF += 1j / (m * Omega) * commutator(Gm, Gmm)

    if order >= 2:
        if any(abs(m) > 1 for m in modes):
            raise NotImplementedError(
                "order>=2 closed form is implemented for single-harmonic drive "
                "only (m in {-1,0,1}); higher harmonics need van Vleck cross "
                "terms. Use magnus_terms_quadrature for multi-harmonic drives."
            )
        for m in pos:
            Gm = modes[m]
            Gmm = modes[-m]
            term = (commutator(commutator(Gm, G0), Gmm)
                    + commutator(commutator(Gmm, G0), Gm))
            GF += -1.0 / (2.0 * m * m * Omega ** 2) * term

    if order >= 3:
        raise NotImplementedError("van Vleck expansion implemented to order 2.")
    return GF


# --------------------------------------------------------------------------- #
# (2) Direct numerical quadrature of the Magnus series (independent reference)
# --------------------------------------------------------------------------- #
def _generator_on_grid(modes: dict[int, np.ndarray], Omega: float,
                       t: np.ndarray) -> np.ndarray:
    """G(t) evaluated on a time grid. Returns array of shape (len(t), N, N)."""
    N = modes[0].shape[0]
    G = np.zeros((len(t), N, N), dtype=complex)
    for m, Gm in modes.items():
        G += np.exp(1j * m * Omega * t)[:, None, None] * Gm[None, :, :]
    return G


def _cumulative_integral(F: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral C(t_k) = int_0^{t_k} F(s) ds for a
    matrix-valued F sampled on grid t. Returns same shape as F, C[0]=0."""
    C = np.zeros_like(F)
    dt = np.diff(t)
    incr = 0.5 * (F[1:] + F[:-1]) * dt[:, None, None]
    C[1:] = np.cumsum(incr, axis=0)
    return C


def magnus_terms_quadrature(modes: dict[int, np.ndarray], Omega: float,
                            n_grid: int = 4001) -> dict[int, np.ndarray]:
    """Magnus terms Phi_1, Phi_2, Phi_3 by direct quadrature of their integral
    definitions over one period T = 2*pi/Omega.

        Phi_1 = int_0^T G dt
        Phi_2 = (1/2) int_0^T [ G(t1), int_0^{t1} G dt2 ] dt1
        Phi_3 = (1/6) int_0^T ( [G(t1), Q(t1)] + R(t1) ) dt1
                Q(t1) = int_0^{t1} [G(t2), int_0^{t2} G dt3] dt2
                R(t1) = int_0^{t1} [ int_0^{t2} G dt3 , [G(t2), G(t1)] ] dt2

    These are the unambiguous textbook definitions (no memorised expansion
    coefficient), used to cross-check the closed-form path. Floquet-Magnus
    form (t0=0); its *eigenvalues* match the van Vleck generator's.

    Returns {1: Phi_1, 2: Phi_2, 3: Phi_3}.
    """
    T = 2.0 * np.pi / Omega
    t = np.linspace(0.0, T, n_grid)
    G = _generator_on_grid(modes, Omega, t)          # (n, N, N)
    N = G.shape[1]

    # Phi_1
    Phi1 = np.trapezoid(G, t, axis=0)

    # cumulative C(t) = int_0^t G ds
    C = _cumulative_integral(G, t)

    # Phi_2 = (1/2) int_0^T [G(t1), C(t1)] dt1
    comm2 = G @ C - C @ G                            # [G(t1), C(t1)] pointwise
    Phi2 = 0.5 * np.trapezoid(comm2, t, axis=0)

    # Phi_3, first half: (1/6) int [G(t1), Q(t1)] dt1, Q = int_0^{t1} [G(t2),C(t2)] dt2
    P = G @ C - C @ G                                # [G(t2), C(t2)]  (== comm2)
    Q = _cumulative_integral(P, t)
    half1 = G @ Q - Q @ G                            # [G(t1), Q(t1)]

    # Phi_3, second half: (1/6) int R(t1) dt1,
    # R(t1) = int_0^{t1} [C(t2), [G(t2), G(t1)]] dt2  (G(t1) fixed inside inner int)
    # Build pointwise; small N so an explicit t1-loop is cheap and clear.
    half2 = np.zeros((len(t), N, N), dtype=complex)
    for k in range(len(t)):
        G_t1 = G[k]
        # inner integrand over t2 in [0, t1_k]:  [C(t2), [G(t2), G_t1]]
        inner_comm = G[:k + 1] @ G_t1 - G_t1 @ G[:k + 1]      # [G(t2), G(t1)]
        integrand = (C[:k + 1] @ inner_comm - inner_comm @ C[:k + 1])
        if k == 0:
            half2[k] = 0.0
        else:
            half2[k] = np.trapezoid(integrand, t[:k + 1], axis=0)

    Phi3 = (1.0 / 6.0) * np.trapezoid(half1 + half2, t, axis=0)

    return {1: Phi1, 2: Phi2, 3: Phi3}


def effective_generator_quadrature(modes: dict[int, np.ndarray], Omega: float,
                                   order: int = 3, n_grid: int = 4001) -> np.ndarray:
    """G_F from the quadratured Magnus series: (Phi_1 + ... + Phi_order)/T."""
    terms = magnus_terms_quadrature(modes, Omega, n_grid=n_grid)
    T = 2.0 * np.pi / Omega
    Phi = sum(terms[k] for k in range(1, order + 1) if k in terms)
    return Phi / T


def one_period_map_from_modes(modes: dict[int, np.ndarray], Omega: float,
                              order: int = 3, n_grid: int = 4001) -> np.ndarray:
    """Approximate one-period map M = exp(Phi_1 + ... + Phi_order) via quadrature.

    (This is the *Magnus* approximation to M. The exact M comes from integrating
    the variational ODE directly — that lives in lyapunov.py / Experiment 0.1.)
    """
    terms = magnus_terms_quadrature(modes, Omega, n_grid=n_grid)
    Phi = sum(terms[k] for k in range(1, order + 1) if k in terms)
    return expm(Phi)
