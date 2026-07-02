"""
core/bands.py
=============

Frequency-band partitioning + band-targeted input projection for Experiment D
(cross-band routing). The substrate, operating point, integrator, ESP gate and
readout are all inherited UNCHANGED from A/C (core/reservoir, integrator_corotating,
consistency, readout); this module adds only the two D-specific primitives the
Gate-0 well-posedness probe needs and nothing else:

  1. band_indices  -- split the log-uniform omega spectrum into a SLOW band (bottom
     tertile by log-frequency), a FAST band (top tertile), and a GUARD band (middle
     tertile, excluded from BOTH injection and readout). The guard third buys genuine
     timescale separation between the injected (fast) and read-out (slow) bands so a
     positive transfer cannot be short-range leakage across an adjacent boundary.

  2. masked_encoding -- a complex input projection m supported ONLY on a chosen index
     set (zero elsewhere). This replaces A/C's uniform random projection (which drove
     every oscillator) so that u_fast enters the FAST band exclusively. With the
     existing single-stream integrate_corotating_input (scalar u, vector m), a masked
     m is all that is needed to inject one stream into one band -- no change to the
     validated hot loop.

Design rationale (Gate-0 spec): the readout already includes |z|^2, the natural
square-law demodulator; the only instrument gaps were on the INJECTION side (band
targeting) and the band PARTITION (guard). Those two live here.

Pure host numpy -- these build the (omega, m) descriptors consumed by the integrator
batch wrappers; no device code.
"""
from __future__ import annotations

import numpy as np

__all__ = ["band_indices", "masked_encoding", "band_summary"]


def band_indices(omega, guard: bool = True) -> dict:
    """Partition oscillators into slow / guard / fast bands by log-frequency tertiles.

    omega : (N,) natural frequencies (log-uniform, as built by reservoir.build_system;
            need not be pre-sorted -- we sort by value).
    guard : if True (default), the middle tertile is the GUARD band, excluded from both
            injection and readout. If False, a two-way split (no guard) with slow =
            bottom half, fast = top half (kept only as an ablation handle).

    Returns dict with int index arrays into the ORIGINAL omega ordering:
      slow, guard, fast  -- each sorted ascending by index.
    plus the (lo, hi) frequency edges of each band for the record.

    Tertiles are by RANK in log-frequency; for a log-uniform spectrum that is identical
    to equal-log-width bands. N not divisible by 3 -> the guard absorbs the remainder
    (slow and fast get floor(N/3) each), keeping the two active bands equal-sized.
    """
    omega = np.asarray(omega, dtype=float)
    N = omega.shape[0]
    order = np.argsort(omega)                 # ascending by frequency
    if guard:
        k = N // 3                            # slow = bottom k, fast = top k
        slow = order[:k]
        fast = order[N - k:]
        mid = order[k:N - k]
    else:
        k = N // 2
        slow = order[:k]
        fast = order[k:]
        mid = order[:0]
    out = dict(
        slow=np.sort(slow), guard=np.sort(mid), fast=np.sort(fast),
    )
    for name in ("slow", "guard", "fast"):
        idx = out[name]
        if len(idx):
            out[name + "_edges"] = (float(omega[idx].min()), float(omega[idx].max()))
        else:
            out[name + "_edges"] = (float("nan"), float("nan"))
    return out


def masked_encoding(omega, idx, rng, normalize: bool = False) -> np.ndarray:
    """Complex input projection m (N,) supported ONLY on the index set `idx`.

    Within the support, entries are iid complex normal -- the same distribution as
    core.reservoir.build_encoding('random'), so a band-masked run is the single-variable
    analogue of A's random encoding restricted to one band. Outside `idx`, m is EXACTLY
    zero (verified by band_summary / the gate's mask-orthogonality check).

    normalize : if True, scale the support so sum|m|^2 == len(idx) (equal injected power
                regardless of band size); default False (raw iid, matching A's 'random').
    """
    omega = np.asarray(omega, dtype=float)
    N = omega.shape[0]
    idx = np.asarray(idx, dtype=int)
    m = np.zeros(N, dtype=complex)
    sub = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))
    if normalize:
        nrm = np.sqrt(np.sum(np.abs(sub) ** 2)) + 1e-12
        sub = sub / nrm * np.sqrt(len(idx))
    m[idx] = sub
    return m


def band_summary(omega, bands: dict) -> str:
    """One-line human summary of a band partition (sizes + frequency edges + the
    fast/slow separation ratio that gates demodulation testability)."""
    slow_e = bands["slow_edges"]
    fast_e = bands["fast_edges"]
    # nearest fast / farthest slow gives the WORST-case (smallest) separation ratio.
    near_ratio = fast_e[0] / slow_e[1] if slow_e[1] > 0 else float("nan")
    span_ratio = fast_e[1] / slow_e[0] if slow_e[0] > 0 else float("nan")
    return (f"slow N={len(bands['slow'])} w in [{slow_e[0]:.3g},{slow_e[1]:.3g}] | "
            f"guard N={len(bands['guard'])} | "
            f"fast N={len(bands['fast'])} w in [{fast_e[0]:.3g},{fast_e[1]:.3g}] | "
            f"sep ratio: nearest {near_ratio:.2g}x, widest {span_ratio:.2g}x")
