"""
experiments/relay_gate3.py
==========================

Relay Gate-3: mechanism decomposition (readout-channel ablation). Per
relay_gate3_mechanism_spec.md. Stage-A only (NO chains, no new core machinery).

Determines WHICH readout channel carries the cross-band message as a function of message
band -- the |z|^2 square-law (C3's claim) vs the linear (Re z, Im z) resonant channel -- by
readout-FEATURE ablation. The ablation is FIT-TIME ONLY and already first-class in the
committed Phase-1 readout: p1.band_features(X, idx, mode) gives
    full -> [Re z, Im z, |z|^2, bias]   (FULL)
    pow  -> [|z|^2, bias]               (SQ)
    reim -> [Re z, Im z, bias]          (LIN)
so SQ/LIN are provable COLUMN-SUBSETS of FULL, fed to the SAME p1.demod_capacity (same idx,
same inner-val lambda-SELECTION per mode, same CV split, bias-protected -- each mode picks its
own ridge penalty from ENV_LAMS, the correct choice for a subset model). One integration per
(band, seed); the
three ridge fits subset the features of that ONE trajectory.

Built STRICTLY by REUSE of the committed Gate-1 / Gate-0 / Phase-1 machinery (imported). This
module modifies NOTHING in relay_gate0/1/2.py, D_phase1_routing.py, the core/ files, or any
committed artifact -- it only imports them.

Axes:
  * message band (3): SUB [0.2,0.9] (below slow resonance; the standard Phase-1 band -> ANCHOR),
    RES [2,9] (overlaps the slow tertile's natural range), SUPRA [10,28] (above slow resonance,
    inside the fast tertile range, below Nyquist -- carry the recorded carrier-comparable caveat).
  * readout ablation (3): FULL / SQ / LIN (the p1 modes above).

Pre-registration (spec, pinned): retention R_abl = r2_abl / r2_FULL per (band, seed),
intersection means; a band classifies only where FULL r2 > 0.2; channel = SQ-carried if
R_SQ>=0.9 and R_LIN<0.5, LIN-carried if R_LIN>=0.9 and R_SQ<0.5, else MIXED. Per-band, no pooling.

Modes:
  --sandbox       Stage 1. CPU-ONLY: ablation column-subset correctness, decoy protocol match,
                  seed-collision matrix vs committed families, validity-guard + classifier +
                  consequence-map logic, ESP-intersection logic, band geometry.
  --verdict-test  CPU-only synthetic exercise of decide() across consequence-map + NO-MEASUREMENT.
  --smoke         Stage 2 (separate go). FULL x SUB seed 0 must reproduce committed 0.981470
                  digit-exact; one RES and one SUPRA run complete with logging.
  --run           Stage 3 (separate go). Full battery: 3 bands x n seeds.
  --reread        Re-decide + re-render from committed gate3 recs (CPU, byte-identical assert).

STOP-and-report after --sandbox. Nothing committed. Drift-attribution is OUT of scope (Gate 4+).
"""
from __future__ import annotations

import os
import sys
import json
import argparse

if any(a in sys.argv for a in ("--sandbox", "--verdict-test", "--reread")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))                        # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root

import numpy as np                                                   # noqa: E402
import D_phase1_routing as p1                                        # noqa: E402 (jax x64 on import)
import relay_gate0 as g0                                             # noqa: E402
import relay_gate1 as g1                                             # noqa: E402

RESDIR = g0.RESDIR

# --- carried from the committed harnesses (imported, not redefined) -------------------- #
STAGE_SPAN = g0.STAGE_SPAN            # 1.5 decades (Phase-1 stage-A span)
K_PRIMARY = g0.K_PRIMARY             # 0.24 (relay operating point)
N = p1.N                             # 500
N_DEC = p1.N_DEC                     # 60 decoy-null draws (Phase-1 exact)
ESP_EPS = p1.ESP_EPS
ANCHOR, ANCHOR_SE_K, ANCHOR_FLOOR = g0.ANCHOR, g0.ANCHOR_SE_K, g0.ANCHOR_FLOOR   # 0.986 / 2 / 0.02
MIN_PAIRS = g0.MIN_PAIRS             # >=2 ESP-honest seeds or underpowered (target >=5)
DECOY_ELEVATED = g1.DECOY_ELEVATED   # decoy bar 0.2
REF_TABLE = g0.REF_TABLE             # committed b0f7664 per-seed rows: {(span,K): {seed:(r2_d0,esp)}}
SEED_MAX = 9                         # REF_TABLE / committed Phase-1 covers seeds 0..9

# seed families (Phase-1 EXACT so FULL x SUB reproduces the committed anchor)
ENC_BASE, REP_BASE, MSG_BASE = 5000, 9000, 1000     # masked_encoding rng / replica / am message
RADE_OFFSET = 777                                   # am_input_band Rademacher carrier offset

# axis 1: message band. SUB == the committed Phase-1 band (p1.MSG_LO/HI) -> the ANCHOR replica.
BANDS = {"SUB": (p1.MSG_LO, p1.MSG_HI), "RES": (2.0, 9.0), "SUPRA": (10.0, 28.0)}
BAND_ORDER = ["SUB", "RES", "SUPRA"]
# axis 2: readout ablation -> committed p1.band_features mode
MODES = {"FULL": "full", "SQ": "pow", "LIN": "reim"}
MODE_ORDER = ["FULL", "SQ", "LIN"]
# per-band decoy bases: distinct + clear of ALL committed families (proven in sandbox check 0)
DECOY_BASE = {"SUB": 300000, "RES": 320000, "SUPRA": 340000}

# classifier thresholds (spec, pinned now -- no post-hoc moving)
R_HI = 0.9           # a channel "carries" if its retention >= 0.9 ...
R_LO = 0.5           # ... AND the other channel's retention < 0.5
FULL_R2_MIN = 0.2    # validity guard: a band classifies only where FULL r2 > this

FRAMING = ("Stage-A Phase-1 replica (N=500, span 1.5, K=0.24), message fast-tertile injected, "
           "slow-tertile readout. Ablation is FIT-TIME ONLY (subset the SAME trajectory's readout "
           "features; SQ/LIN are column-subsets of FULL). Retention R_abl=r2_abl/r2_FULL per "
           "(band,seed); a band classifies only where FULL r2>0.2. SUPRA is carrier-comparable "
           "(injection ill-posedness may contribute). No chains; drift-attribution is out of scope.")

# Anchor provenance (Jason-ratified 2026-07-09; every field VERIFIED against the artifact, not asserted).
# REF_TABLE stores 6-dp transcriptions, so "faithful" is at that stored precision; the BIT-EXACT claim is
# between phase1_routing.json's full-precision mean(seeds 0-7) and the committed gate1 record anchor mean.
ANCHOR_SRC = {
    "repo": "decade_drive (github.com/EridosAI/decade_drive)",
    "commit": "b0f7664",
    "path": "results/D/phase1_routing.json",
    "sha256": "2e739315141e88c3c5c698f88ed6f84efaae46f7257397c756f58ee4c3965590",
    "sha256_scope": "working tree AND the blob as committed at b0f7664 (identical)",
    "ref_table_cells": ("all 40 REF_TABLE cells (4 (span,K) keys x 10 seeds) faithful to the committed "
                        "record: 40/40 r2_d0 and 40/40 ESP ok_slow flags, 0 discrepancies, at the 6-dp "
                        "precision REF_TABLE stores"),
    "mean_0_7": ("phase1_routing.json full-precision mean(seeds 0-7) at (span 1.5, K=0.24) = "
                 "0.9864540115271048, BIT-EXACT to the committed gate1 record anchor mean; REF_TABLE's own "
                 "6-dp mean = 0.986454 (equal at 6 dp)"),
}
# Scope tag (Jason-ratified AMEND-2): appended to the verdict line whenever the C3-STANDS branch fires,
# so neither the json verdict nor the .md (nor a git-log quote of either) can be read stronger than the doc.
SCOPE_TAG = ('[SCOPED -- the injection supplies no linear message content; the LIN-carried branch tested '
             'only reservoir-mediated re-encoding into first-order coordinates, which did not occur. NOT '
             'an independent refutation of a linear resonant channel. See "Degeneracy and the scope of '
             'this verdict".]')

ANCHOR_PROV_MD = ("Anchor provenance: verified against decade_drive b0f7664 results/D/phase1_routing.json "
                  "(sha256 2e739315141e88c3c5c698f88ed6f84efaae46f7257397c756f58ee4c3965590; working tree "
                  "== blob at b0f7664). All 40 REF_TABLE cells (4 (span,K) x 10 seeds) faithful: 40/40 "
                  "r2_d0 + 40/40 ESP flags, 0 discrepancies, at REF_TABLE's stored 6-dp precision. The "
                  "full-precision phase1_routing mean(seeds 0-7) at (1.5, 0.24) = 0.9864540115271048 is "
                  "BIT-EXACT to the committed gate1 record anchor mean (REF_TABLE's own 6-dp mean = "
                  "0.986454).")


def _mstats(v):
    return g0._mstats(v)


def _fmt(x, spec="+.4f"):
    return g0._fmt(x, spec)


# ===================================================================================== #
#  Seed-derivation scheme (Phase-1 EXACT leaves + fresh per-band decoy bases) + collision proof
# ===================================================================================== #
def seed_scheme(i, band):
    """Seeds for message band `band`, seed i. build/enc/rep/msg reproduce Phase-1 stage-A EXACTLY
    (so FULL x SUB replays the committed anchor); the decoy base is fresh per band."""
    return {"build": i, "enc": ENC_BASE + i, "rep": REP_BASE + i, "msg": MSG_BASE + i,
            "rademacher": MSG_BASE + i + RADE_OFFSET, "decoy_base": DECOY_BASE[band]}


def _decoy_range(base, seed_max=SEED_MAX):
    return {base + i * 200 + d for i in range(seed_max + 1) for d in range(N_DEC)}


def _committed_families(seed_max=SEED_MAX):
    """All seed families used by the committed harnesses (Phase-1 + Gate-0/1/2) that a NEW Gate-3
    decoy base must clear so its nulls are genuinely never-injected and independent."""
    seeds = range(seed_max + 1)
    # committed decoy bases: Phase-1 40000; Gate-0/1/2 stage bases + e2e 80000
    committed_decoy_bases = [40000, 60000, 80000, 100000, 120000, 140000,
                             160000, 180000, 200000, 220000, 240000]
    fams = {
        "phase1/gate_decoys": set().union(*[_decoy_range(b, seed_max) for b in committed_decoy_bases]),
        "scramble_laplacian": {70000 + i for i in seeds},          # D_phase1_routing.scramble_laplacian
        "build": {i for i in seeds},
        "enc(5000)": {ENC_BASE + i for i in seeds},
        "rep(9000)/scramble_rep(9500)": {REP_BASE + i for i in seeds} | {9500 + i for i in seeds},
        "carrier(2000..2800 gate1/2)+rademacher": ({2000 + s * 100 + i for s in range(0, 9) for i in seeds}
                                                   | {2000 + s * 100 + i + RADE_OFFSET
                                                      for s in range(0, 9) for i in seeds}),
        "msg(1000)+rademacher": {MSG_BASE + i for i in seeds} | {MSG_BASE + i + RADE_OFFSET for i in seeds},
    }
    return fams


def verify_no_collision(seed_max=SEED_MAX):
    """Prove (over i=0..seed_max) the 3 Gate-3 decoy families are pairwise disjoint AND disjoint
    from every committed seed family. The Phase-1-shared leaves (build/enc/rep/msg) are REUSED by
    design (they reproduce the anchor) -- they are not new families to separate."""
    decoy_by_band = {b: _decoy_range(DECOY_BASE[b], seed_max) for b in BAND_ORDER}
    pw = {}
    for a in range(len(BAND_ORDER)):
        for c in range(a + 1, len(BAND_ORDER)):
            ba, bc = BAND_ORDER[a], BAND_ORDER[c]
            n = len(decoy_by_band[ba] & decoy_by_band[bc])
            if n:
                pw[f"{ba}^{bc}"] = n
    fams = _committed_families(seed_max)
    decoy_all = set().union(*decoy_by_band.values())
    vs_committed = {name: sorted(decoy_all & fam)[:5] for name, fam in fams.items() if decoy_all & fam}
    ok = (not pw) and (not vs_committed)
    return {"ok": bool(ok), "seed_max": seed_max,
            "decoy_bases": DECOY_BASE,
            "decoy_ranges": {b: [min(decoy_by_band[b]), max(decoy_by_band[b])] for b in BAND_ORDER},
            "pairwise_overlaps": pw, "vs_committed_overlaps": vs_committed}


def _band_decoys(band, i, L, dt_in, base=None):
    """Per-band never-injected decoys = the EXACT committed Phase-1 protocol (p1.slow_bandlimited on
    the band's [lo,hi], stride i*200+d, N_DEC draws); differs from Phase-1's committed base 40000 ONLY
    by the fresh per-band base (collision-proven distinct). Used by stage_a AND cross-checked in the
    CPU sandbox CHECK 3 against Phase-1's actual base-40000 decoys."""
    lo, hi = BANDS[band]
    b0 = DECOY_BASE[band] if base is None else base
    return [p1.slow_bandlimited(L, dt_in, lo, hi, seed=b0 + i * 200 + d) for d in range(N_DEC)]


def log_seed_scheme(log, seed_max=SEED_MAX):
    log(f"--- seed scheme (message band x seed i; build/enc/rep/msg = Phase-1 EXACT) ---")
    log(f"  {'field':>12}  value(+i)")
    sd = seed_scheme(0, "SUB")
    for k in ("build", "enc", "rep", "msg", "rademacher"):
        log(f"  {k:>12}  {sd[k]}+i")
    log(f"  {'decoy_base':>12}  per band {DECOY_BASE}")
    rep = verify_no_collision(seed_max)
    log(f"  decoy ranges (i=0..{seed_max}): {rep['decoy_ranges']}")
    log(f"  collision proof: pairwise overlaps={rep['pairwise_overlaps'] or 'none'}; "
        f"vs committed families={rep['vs_committed_overlaps'] or 'none'} -> collision-free: {rep['ok']}")
    return rep


# ===================================================================================== #
#  Stage-A run (ONE integration per (band, seed); 3 ablations on the SAME trajectory)
# ===================================================================================== #
def stage_a(band, i, geom, *, decoys=True):
    """One stage-A run for message band `band`, seed i, at K=0.24. Integrates main + ESP replica
    ONCE over the K_GRID batch (batch-shape -> anchor bit-exact) and picks K=0.24, then scores the
    slow-tertile readout under all 3 ablations via p1.demod_capacity on the SAME trajectory.
    SUB reproduces the committed Phase-1 stage-A exactly (am_input_band==am_input for [0.2,0.9])."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    lo, hi = BANDS[band]
    sd = seed_scheme(i, band)
    sp = p1.build_system(sd["build"], N, STAGE_SPAN)
    bands = p1.band_indices(sp.omega)
    # runtime-MEASURED readout widths (never asserted from memory): probe the committed band_features
    nslow = int(len(bands["slow"]))
    _probe = np.zeros((2, N), dtype=complex)
    fcols = {k: int(p1.band_features(_probe, bands["slow"], MODES[k]).shape[1]) for k in MODE_ORDER}
    m_fast = p1.masked_encoding(sp.omega, bands["fast"], np.random.default_rng(sd["enc"]))
    outside = np.concatenate([bands["slow"], bands["guard"]])
    assert float(np.abs(m_fast[outside]).max()) == 0.0                # fast-band-only injection
    s_msg, u_am = g0.am_input_band(L, dt_in, sd["msg"], lo, hi)        # SUB == p1.am_input exact
    Ks = list(p1.K_GRID)
    ki = Ks.index(K_PRIMARY)                                          # batch-shape -> anchor bit-exact
    X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u_am, Ks, dt_in, n_sub)[ki]
    rep = p1.replica_spec(sp, sd["rep"])
    Xr = p1.integrate_Ks(sp.omega, sp.L, m_fast, rep.z0, u_am, Ks, dt_in, n_sub)[ki]
    d_slow = p1.consistency_distance(X[:, bands["slow"]], Xr[:, bands["slow"]], sl)
    esp = {"d_slow": float(d_slow), "ok_slow": bool(d_slow < ESP_EPS)}
    assert sd["decoy_base"] == DECOY_BASE[band]                       # seed_scheme base == module base
    dec = _band_decoys(band, i, L, dt_in) if decoys else None
    abl = {}
    for name in MODE_ORDER:
        dm = (p1.demod_capacity(X, bands["slow"], s_msg, dec, delays, sl, MODES[name])
              if dec is not None else None)
        abl[name] = ({"r2_d0": float(dm["r2_d0"]), "decoy_p95": float(dm["decoy_p95"]),
                      "lam": float(dm["lam"]), "cap": float(dm["cap"])} if dm else None)
    mband = g0.dominant_band(s_msg[iw], dt_in)
    del X, Xr
    return {"band": band, "band_hz": [lo, hi], "seed": i, "K": K_PRIMARY, "seeds": sd,
            "n_slow": nslow, "feature_cols": fcols,
            "r2": {k: (abl[k]["r2_d0"] if abl[k] else None) for k in MODE_ORDER},
            "decoy_p95": {k: (abl[k]["decoy_p95"] if abl[k] else None) for k in MODE_ORDER},
            "esp": esp, "msg_dominant_band": [float(mband[0]), float(mband[1])], "ablation": abl}


# ===================================================================================== #
#  Retention + per-band classification + consequence map (pre-registered)
# ===================================================================================== #
def _classify(R_SQ, R_LIN):
    if R_SQ is None or R_LIN is None:
        return "below-guard"
    if R_SQ >= R_HI and R_LIN < R_LO:
        return "SQ-carried"
    if R_LIN >= R_HI and R_SQ < R_LO:
        return "LIN-carried"
    return "MIXED"


def _retention(recs, inter):
    """Per band: retention R_SQ, R_LIN = mean over the intersection of r2_abl/r2_FULL, using only
    (band,seed) with FULL r2 > FULL_R2_MIN (validity guard). Per-band classification, no pooling."""
    out = {}
    for b in BAND_ORDER:
        rows = [recs[(b, i)] for i in inter]
        full_all = [r["r2"]["FULL"] for r in rows if r["r2"]["FULL"] is not None]
        valid = [r for r in rows if r["r2"]["FULL"] is not None and r["r2"]["FULL"] > FULL_R2_MIN]
        if len(valid) < MIN_PAIRS:
            out[b] = {"n_valid": len(valid), "classifies": False, "R_SQ": None, "R_LIN": None,
                      "r2_full": _mstats(full_all) if full_all else None,
                      "classification": "below-guard"}
            continue
        R_SQ = _mstats([r["r2"]["SQ"] / r["r2"]["FULL"] for r in valid])
        R_LIN = _mstats([r["r2"]["LIN"] / r["r2"]["FULL"] for r in valid])
        out[b] = {"n_valid": len(valid), "classifies": True, "R_SQ": R_SQ, "R_LIN": R_LIN,
                  "r2_full": _mstats([r["r2"]["FULL"] for r in valid]),
                  "classification": _classify(R_SQ["mean"], R_LIN["mean"])}
    return out


def _consequence(ret):
    """Pre-registered consequence map (Jason-ratified 2026-07-09). Priority: scope-note, then OFF-BAND
    UNTESTED, then C3-stands, then report-as-is. The bands are NOT symmetric: SUB is anchor-protected;
    RES is the diagnostic off-band band (RES-dark contradicts the Gate-0 side-finding); SUPRA may
    legitimately be dark (documented ill-posedness). A scope verdict (scope-note or C3-stands) requires
    RES to classify -- an RES key, NOT a >=2-count guard (a count guard would wrongly pass SQ/BG/SQ).
    res_ok is kept explicit in the C3-stands conjunct so the four outcomes are semantically disjoint
    independent of ordering."""
    sub = ret["SUB"]["classification"]
    res = ret["RES"]["classification"]
    supra = ret["SUPRA"]["classification"]
    res_ok = ret["RES"]["classifies"]
    classified = [b for b in BAND_ORDER if ret[b]["classifies"]]
    # B1 -- C3 SCOPE NOTE (RES=LIN-carried => RES classifies):
    if sub == "SQ-carried" and res == "LIN-carried":
        return ("C3 SCOPE NOTE (observable-order): the quadratic |z|^2 observable order suffices "
                "sub-resonance (SUB=SQ), but the LINEAR (Re,Im) observable order suffices for the "
                "resonant band (RES=LIN) -- so C3's |z|^2 claim gains a scope note: |z|^2 is the "
                "sub-resonance envelope order; the linear order suffices in-band. SUPRA observable "
                f"order = {supra} (reported from the matrix, not asserted). The Phase-3 "
                "'envelope-of-envelope rate cost' erratum becomes data-backed (drafting/filing in "
                "decade_drive = Jason's call).")
    # OFF-BAND UNTESTED -- SUB clean SQ, but RES (the diagnostic off-band band) is dark:
    if sub == "SQ-carried" and not res_ok:
        return ("OFF-BAND UNTESTED -- sub-resonance |z|^2 confirmed; off-band scope untested (RES below "
                "guard, anomalous vs the Gate-0 side-finding). STOP / INVESTIGATE instrument.")
    # B2 -- C3 STANDS (gated on RES classifying, and all classified bands SQ-carried):
    if res_ok and classified and all(ret[b]["classification"] == "SQ-carried"
                                     for b in classified) and sub == "SQ-carried":
        return ("C3 STANDS unmodified (SQ-carried across all classified bands, RES included); the "
                "broadband side-finding needs another explanation (report, no new claim).")
    # B3 -- REPORT MATRIX AS-IS:
    return "REPORT MATRIX AS-IS (neither the scope-note, OFF-BAND-UNTESTED, nor C3-stands pattern)."


# ===================================================================================== #
#  Verdict engine (instruments FIRST; anchor is FULL x SUB digit-exact vs REF_TABLE)
# ===================================================================================== #
def decide(recs, seeds, ref=None):
    if ref is None:
        ref = REF_TABLE[(STAGE_SPAN, K_PRIMARY)]
    esp = {i: bool(all(recs[(b, i)]["esp"]["ok_slow"] for b in BAND_ORDER)) for i in seeds}
    inter = [i for i in seeds if esp[i]]
    # runtime-logged readout widths (from the recs, not asserted); None on synthetic recs
    fcols = next((recs[k].get("feature_cols") for k in recs if recs[k].get("feature_cols")), None)
    nslow = next((recs[k].get("n_slow") for k in recs if recs[k].get("n_slow")), None)
    out = {"framing": FRAMING, "esp_all_bands": {str(i): esp[i] for i in seeds},
           "intersection": inter, "n_intersection": len(inter),
           "feature_cols": fcols, "n_slow": nslow,
           "operationalization": {"k_primary": K_PRIMARY, "modes": MODES, "bands": BANDS,
                                  "R_HI": R_HI, "R_LO": R_LO, "full_r2_min": FULL_R2_MIN,
                                  "min_pairs": MIN_PAIRS, "decoy_bar": DECOY_ELEVATED,
                                  "anchor_window": f"max({ANCHOR_SE_K}*SE,{ANCHOR_FLOOR})",
                                  "sq_carried": "R_SQ>=%.2f and R_LIN<%.2f" % (R_HI, R_LO),
                                  "lin_carried": "R_LIN>=%.2f and R_SQ<%.2f" % (R_HI, R_LO)}}
    if len(inter) < MIN_PAIRS:
        out["verdict"] = (f"NO-MEASUREMENT (underpowered: ESP intersection across all 3 bands "
                          f"n={len(inter)} < {MIN_PAIRS} -- add seeds, do not read)")
        return out

    # ---- anchor: FULL x SUB is a Phase-1 replica -> per-seed digit-exact vs REF_TABLE ---- #
    anchor_ps = {i: recs[("SUB", i)]["r2"]["FULL"] for i in inter}
    stA = _mstats(list(anchor_ps.values()))
    window = max(ANCHOR_SE_K * stA["se"], ANCHOR_FLOOR)
    mean_ok = bool(abs(stA["mean"] - ANCHOR) <= window)
    digit = {i: {"got": round(anchor_ps[i], 6), "ref": round(float(ref[i][0]), 6),
                 "match": bool(round(anchor_ps[i], 6) == round(float(ref[i][0]), 6))}
             for i in inter if i in ref}
    digit_ok = bool(len(digit) > 0 and all(v["match"] for v in digit.values()))
    out["anchor"] = {"full_sub": stA, "target": ANCHOR, "window": window, "mean_ok": mean_ok,
                     "per_seed_digit": {str(k): v for k, v in digit.items()}, "digit_exact": digit_ok}

    # ---- per-band decoys (all bands x all modes) ---------------------------------------- #
    dcells = [((b, m), recs[(b, i)]["decoy_p95"][m]) for b in BAND_ORDER for i in inter
              for m in MODE_ORDER if recs[(b, i)]["decoy_p95"][m] is not None]
    dvals = [x for _, x in dcells]
    leak = bool(dvals and max(dvals) > DECOY_ELEVATED)
    worst = (list(max(dcells, key=lambda kv: kv[1])[0]) if dcells else None)   # argmax band x mode
    out["decoys"] = {"max_p95": (float(max(dvals)) if dvals else None), "bar": DECOY_ELEVATED,
                     "elevated": leak, "max_cell": worst,
                     "by_band_mode": {b: {m: _mstats([recs[(b, i)]["decoy_p95"][m] for i in inter
                                                      if recs[(b, i)]["decoy_p95"][m] is not None])["mean"]
                                          for m in MODE_ORDER} for b in BAND_ORDER}}

    # ---- retention matrix + classification ---------------------------------------------- #
    ret = _retention(recs, inter)
    out["retention"] = ret
    out["matrix_r2"] = {b: {m: _mstats([recs[(b, i)]["r2"][m] for i in inter
                                        if recs[(b, i)]["r2"][m] is not None])["mean"]
                            for m in MODE_ORDER} for b in BAND_ORDER}

    # ---- verdict (instruments first; pre-registered order) ------------------------------ #
    if not (mean_ok and digit_ok):
        out["verdict"] = (f"NO-MEASUREMENT (anchor miss: FULL x SUB mean {stA['mean']:.6f} vs "
                          f"{ANCHOR} +/- {window:.4f} [mean_ok={mean_ok}]; per-seed digit-exact vs "
                          f"REF_TABLE={digit_ok} -- STOP, fix, re-run)")
    elif leak:
        out["verdict"] = (f"NO-MEASUREMENT (decoy elevated: max per-band/mode p95 {max(dvals):.3f} "
                          f"> {DECOY_ELEVATED} -- leakage)")
    else:
        cls = {b: ret[b]["classification"] for b in BAND_ORDER}
        cons = _consequence(ret)
        out["consequence"] = cons
        # the pre-registered consequence text stays VERBATIM; the ratified scope tag is APPENDED to the
        # verdict line on the C3-STANDS branch only (the branch whose strength the tag re-scopes).
        tag = (" " + SCOPE_TAG) if cons.startswith("C3 STANDS") else ""
        out["verdict"] = (f"MECHANISM MATRIX -- SUB={cls['SUB']}, RES={cls['RES']}, "
                          f"SUPRA={cls['SUPRA']}. {cons}{tag}")
    return out


# ===================================================================================== #
#  Markdown record
# ===================================================================================== #
def _cols_str(v):
    """Runtime-logged per-mode readout column counts, straight from the recs (never asserted)."""
    fc, ns = v.get("feature_cols"), v.get("n_slow")
    if not fc:
        return "readout widths not logged"
    return (f"slow tertile n={ns}; FULL {fc.get('FULL')} / SQ {fc.get('SQ')} / LIN {fc.get('LIN')} columns")


def _rline(r2, dp):
    return (f"FULL {_fmt(r2['FULL'],'.4f')} | SQ {_fmt(r2['SQ'],'.4f')} | LIN {_fmt(r2['LIN'],'.4f')}"
            f"  (decoy p95 seed-means: FULL {_fmt(dp['FULL'],'.3f')}/SQ {_fmt(dp['SQ'],'.3f')}/"
            f"LIN {_fmt(dp['LIN'],'.3f')})")


def _write_md(path, v, seeds, wall, hashes, colrep, note=""):
    lines = [
        "# Relay Gate-3 -- mechanism decomposition (readout-channel ablation)", "",
        f"Spec: relay_gate3_mechanism_spec.md (sha256 {hashes['spec']}). Harness: "
        f"experiments/relay_gate3.py (sha256 {hashes['code']}).",
        f"Seeds run: {seeds}. K = {K_PRIMARY}, span {STAGE_SPAN}. Wall-clock {wall/60:.0f} min. "
        f"Seed scheme collision-free: {colrep['ok']}.",
    ] + ([note] if note else []) + [
        "", f"Framing: {FRAMING}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
        "## Instrument checks (pre-registered order)", "",
    ]
    if "anchor" in v:
        a, d = v["anchor"], v["decoys"]
        digit = a["per_seed_digit"]
        lines += [
            f"1. **Anchor (FULL x SUB == committed Phase-1 b0f7664)**: mean {a['full_sub']['mean']:.6f} "
            f"(SE {a['full_sub']['se']:.6f}, n={a['full_sub']['n']}); target {ANCHOR} +/- {a['window']:.4f} "
            f"-> mean_ok={a['mean_ok']}; per-seed digit-exact vs REF_TABLE -> {a['digit_exact']} "
            f"({sum(1 for x in digit.values() if x['match'])}/{len(digit)} seeds). {ANCHOR_PROV_MD}",
            f"2. **Per-band decoys** -- the gate statistic is the PER-CELL decoy p95: the 95th percentile "
            f"of r2 over {N_DEC} never-injected same-class decoy messages, computed independently for each "
            f"(band, seed, mode) cell. The gate takes the MAX over ALL such cells. This battery: max "
            f"{_fmt(d['max_p95'],'.3f')} at {('/'.join(d['max_cell']) if d.get('max_cell') else 'n/a')} "
            f"(margin {_fmt(DECOY_ELEVATED - d['max_p95'],'.3f')} below the {DECOY_ELEVATED} bar) -> "
            f"{'ELEVATED (leak)' if d['elevated'] else 'clean'}. Runtime-logged readout widths at span "
            f"{STAGE_SPAN} ({_cols_str(v)}). Observed IN THIS BATTERY ONLY (n={v['n_intersection']}; no "
            "general mode-ordering law is claimed): LIN's nulls sit just above zero and tightly clustered, "
            "while FULL's are mostly negative but far more dispersed and contain the single largest null of "
            "the run -- which is why the max cell above is a FULL cell. A plausible reading, NOT a "
            "hypothesis this gate tests: each mode's ridge penalty is inner-validated on the REAL message, "
            "so a mode that cannot fit it is regularized toward the mean and its decoy prediction collapses "
            "to the train mean (null just above 0), while a mode that fits well takes a small penalty and "
            "its wider feature set overfits the decoy's train split (null negative, high variance). "
            "Overfitting drives a null DOWN, not up; leakage would require features that genuinely track "
            "the decoy. The gate is sound because it maxes over ALL cells -- it assumes no mode bounds "
            "the others.",
            f"3. **ESP-honest intersection** (ESP-ok across ALL 3 bands): {v['intersection']} "
            f"(n={v['n_intersection']}).", "",
            "## Readout observable-order matrix -- WHICH OBSERVABLE ORDER SUFFICES for readout "
            "(linear Re,Im vs quadratic |z|^2); demod r2_d0 (intersection means), 3 bands x 3 ablations",
            "",
            "Reading: this is an observable-ORDER sufficiency test -- which readout order the ridge demod "
            "needs to reconstruct the message -- NOT a claim of a distinct physical transport channel. "
            "'SQ-carried' = the quadratic |z|^2 observable ALONE suffices (linear insufficient); "
            "'LIN-carried' = the linear (Re,Im) observable ALONE suffices; 'MIXED' = neither order alone "
            "suffices.", "",
        ]
        for b in BAND_ORDER:
            mr = v["matrix_r2"][b]
            dpm = d["by_band_mode"][b]
            lines.append(f"- **{b}** {BANDS[b]}: {_rline(mr, dpm)}")
        lines += ["", "## Retention + per-band observable-order sufficiency (R_abl = r2_abl / r2_FULL; "
                  f"guard FULL r2 > {FULL_R2_MIN})", "",
                  "Ratios are reported UNTOUCHED -- never clipped at 1. R_abl > 1 is expected and means the "
                  "ablated feature set matches or EXCEEDS FULL out-of-sample, because FULL's extra features "
                  "mildly overfit (pure out-of-sample variance cost when the dropped order carries nothing). "
                  "'Retention' is floor-language: the classifier's 0.9 is a FLOOR, not a cap.", ""]
        for b in BAND_ORDER:
            r = v["retention"][b]
            if r["classifies"]:
                lines.append(f"- **{b}**: R_SQ = {r['R_SQ']['mean']:+.4f} +/- {r['R_SQ']['se']:.4f}, "
                             f"R_LIN = {r['R_LIN']['mean']:+.4f} +/- {r['R_LIN']['se']:.4f} "
                             f"(n_valid={r['n_valid']}, FULL r2 {r['r2_full']['mean']:.3f}) -> "
                             f"**{r['classification']}**")
            else:
                fm = (f"{r['r2_full']['mean']:.3f}" if r.get("r2_full") else "n/a")
                lines.append(f"- **{b}**: below validity guard (n_valid={r['n_valid']} < {MIN_PAIRS} "
                             f"with FULL r2 > {FULL_R2_MIN}; FULL r2 mean {fm}) -> **below-guard**")
        lines += ["", "## Consequence", "", v.get("consequence", "(instrument miss -- see verdict)")]
        if str(v.get("consequence", "")).startswith("C3 STANDS"):
            lines += ["", "## Degeneracy and the scope of this verdict (read with the consequence above)", "",
                "1. **Injection degeneracy.** Message injected as square-law AM, u = 0.5*sqrt(s)*w "
                "(Rademacher w) => u^2 = 0.25*s exactly (max|u^2 - 0.25*s| = 2.8e-17) and, because w is "
                "white and independent of s, the input carries no coherent linear message content at any "
                "frequency (eval-window |corr(u,s)| <= 0.032 in SUB, RES and SUPRA alike; lagged "
                "cross-correlations at the noise floor). LIN-carried therefore had no input-level pathway: "
                "it remained reachable only if the reservoir's nonlinearity re-encoded the power envelope "
                "into first-order (Re/Im) slow-tertile coordinates -- slow drifts and AM sidebands in Re/Im "
                "are linearly readable, so this door was physically live. B1 -- the only C3-modifying "
                "branch -- was reachable only through it. This gate tested that door and found it shut "
                "(LIN at or below its own never-injected decoy null, 24/24 cells); it did not arbitrate "
                "between two live input channels. The AM scheme is the ratified Gate-0 instrument and is "
                "not re-litigated here.",
                "2. **What this gate does establish (real, narrow).** A power-encoded message survives "
                "cross-band routing to the slow tertile at all three bands (FULL r2 0.9865 / 0.8601 / "
                "0.6650), and the reservoir does NOT re-encode it into a linearly-readable slow-tertile "
                "form -- not even at RES, where [2,9] overlaps the slow tertile's natural range "
                "[1.00, 3.13].",
                "3. **What it does not establish.** That a live linear resonant transmission channel is "
                "impossible. A coherent (non-power) linear injection was never applied; testing it "
                "requires a different injection and is a separate gate.",
                "4. **Where the discrimination actually lives.** R_SQ >= 0.9 is near-automatic: SQ is a "
                "strict column-subset of FULL, so dropping Re/Im columns that carry nothing generically "
                "improves out-of-sample fit (R_SQ >= 1 in 23/24; lone exception SUB seed 5 at 0.9997 -- "
                "noise-level, far above the floor). All falsifiable content is in R_LIN < 0.5: LIN scores "
                "at or below its own never-injected decoy null in 24/24 cells (its real r2 is negative in "
                "20/24). Read 'SQ-carried' as 'the linear observable order is insufficient AND the message "
                "is retained on |z|^2', not as an independent selection of a quadratic channel.",
                "5. **SUPRA is corroborative, not necessary.** B2 fires on SUB + RES alone (a below-guard "
                "SUPRA still routes to C3 STANDS). On SUPRA's carrier-comparable ill-posedness: we "
                "identify no pathway by which it could produce the LIN/MIXED that alone would break B2 -- "
                "any carrier-mediated reconstruction under this power-encoding is intrinsically a |z|^2 "
                "path (argued, not categorical)."]
    else:
        lines += [f"- Early exit: ESP intersection {v['intersection']} (n={v['n_intersection']}) "
                  f"< {MIN_PAIRS}."]
    lines += ["", "## Scope", "",
              "Stage-A only, offline, one operating point (K=0.24), span 1.5. Ablation is fit-time "
              "feature-subsetting on ONE trajectory per (band, seed). Drift-attribution (WHERE the "
              "Gate-2 m0-referenced loss accrues) is OUT of scope. Gate-4 (hop-length trade) consumes "
              "this gate's channel answer."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ===================================================================================== #
#  STAGE 1 -- CPU sandbox
# ===================================================================================== #
def _rand_complex(T, n, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, n)) + 1j * rng.standard_normal((T, n)))


def sandbox(log):
    log(f"=== RELAY GATE-3 :: STAGE-1 CPU SANDBOX (no GPU; stage-A ablation) ===")
    log(f"    backend: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS','<default>')} "
        f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}'")
    log(f"    framing: {FRAMING}")
    R = {}

    # ---- CHECK 0: seed scheme + collision matrix vs committed families ------------------ #
    log("\n(0) Seed scheme + collision proof (Gate-3 decoy bases vs committed families, i=0..9)")
    colrep = log_seed_scheme(log)
    c0 = g0._check(log, "Gate-3 decoy bases pairwise-disjoint AND clear of all committed families",
                   colrep["ok"], f"bases {DECOY_BASE}; overlaps pw={colrep['pairwise_overlaps'] or 0} "
                   f"vs-committed={colrep['vs_committed_overlaps'] or 0}")
    R["check0_seed_scheme"] = {"pass": c0, "report": colrep}

    # ---- CHECK 1: ablation column-subset correctness (the load-bearing check) ----------- #
    log("\n(1) Ablation column-subset correctness -- SQ/LIN are EXACT column-subsets of FULL")
    T, nsl = 40, 5
    X = _rand_complex(T, 12, seed=1)
    idx = np.array([1, 3, 5, 7, 9])                       # a synthetic 'slow tertile'
    FULL = p1.band_features(X, idx, "full")
    POW = p1.band_features(X, idx, "pow")
    REIM = p1.band_features(X, idx, "reim")
    k = len(idx)
    # FULL layout = [Re(k), Im(k), P(k), bias(1)]
    re_cols, im_cols, p_cols, bias_col = FULL[:, :k], FULL[:, k:2 * k], FULL[:, 2 * k:3 * k], FULL[:, 3 * k:3 * k + 1]
    Zi = X[:, idx]
    layout_ok = (FULL.shape == (T, 3 * k + 1) and np.array_equal(re_cols, Zi.real)
                 and np.array_equal(im_cols, Zi.imag) and np.array_equal(p_cols, np.abs(Zi) ** 2)
                 and np.array_equal(bias_col, np.ones((T, 1))))
    pow_ok = (POW.shape == (T, k + 1) and np.array_equal(POW[:, :k], p_cols)
              and np.array_equal(POW[:, k:k + 1], bias_col))
    reim_ok = (REIM.shape == (T, 2 * k + 1) and np.array_equal(REIM[:, :2 * k], FULL[:, :2 * k])
               and np.array_equal(REIM[:, 2 * k:2 * k + 1], bias_col))
    c1 = all([
        g0._check(log, "FULL layout = [Re, Im, |z|^2, bias]", layout_ok, f"shape {FULL.shape} (k={k})"),
        g0._check(log, "SQ (pow) == FULL's |z|^2 columns + bias (exact)", pow_ok, f"shape {POW.shape}"),
        g0._check(log, "LIN (reim) == FULL's Re,Im columns + bias (exact)", reim_ok, f"shape {REIM.shape}"),
    ])
    R["check1_ablation_subset"] = {"pass": c1}

    # ---- CHECK 2: ablation ISOLATES the channel (synthetic reconstructions) ------------- #
    log("\n(2) Ablation isolation -- a |z|^2-encoded message favors SQ; a Re/Im-encoded one favors LIN")
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(STAGE_SPAN, n_msg=8)
    sl = slice(eval_start, L)
    iw = np.arange(eval_start, L)
    s_env = p1.slow_bandlimited(L, dt_in, p1.MSG_LO, p1.MSG_HI, seed=11)     # a smooth message
    decoys = [p1.slow_bandlimited(L, dt_in, p1.MSG_LO, p1.MSG_HI, seed=900000 + d) for d in range(20)]
    idx2 = np.arange(6)
    # (a) power-channel synthetic: |z_j|^2 = a_j*s_env (LINEAR in s); phase RANDOM per timestep ->
    #     Re = amp*cos(rand), Im = amp*sin(rand) are mean-zero, decorrelated from s (LIN can't read it)
    rng = np.random.default_rng(7)
    aj = rng.uniform(0.5, 1.5, len(idx2))
    phase = rng.uniform(0, 2 * np.pi, (L, len(idx2)))
    amp = np.sqrt(np.maximum(s_env[:, None] * aj[None, :], 1e-9))
    Xa = np.zeros((L, 8), dtype=complex)
    Xa[:, idx2] = amp * np.exp(1j * phase)
    fa = p1.demod_capacity(Xa, idx2, s_env, decoys, delays, sl, "full")["r2_d0"]
    pa = p1.demod_capacity(Xa, idx2, s_env, decoys, delays, sl, "pow")["r2_d0"]
    la = p1.demod_capacity(Xa, idx2, s_env, decoys, delays, sl, "reim")["r2_d0"]
    # (b) linear-channel synthetic: Re z_j = b_j*s_env (LINEAR); Im = large noise so |z|^2 =
    #     (b s)^2 + Im^2 is noise-dominated (SQ can't cleanly read s; LIN reads it off Re)
    bj = rng.uniform(0.5, 1.5, len(idx2))
    Xb = np.zeros((L, 8), dtype=complex)
    Xb[:, idx2] = (s_env[:, None] * bj[None, :]) + 1j * (2.0 * rng.standard_normal((L, len(idx2))))
    fb = p1.demod_capacity(Xb, idx2, s_env, decoys, delays, sl, "full")["r2_d0"]
    pb = p1.demod_capacity(Xb, idx2, s_env, decoys, delays, sl, "pow")["r2_d0"]
    lb = p1.demod_capacity(Xb, idx2, s_env, decoys, delays, sl, "reim")["r2_d0"]
    log(f"    power-encoded : FULL {fa:.3f} SQ {pa:.3f} LIN {la:.3f}  (expect SQ high, SQ>LIN)")
    log(f"    linear-encoded: FULL {fb:.3f} SQ {pb:.3f} LIN {lb:.3f}  (expect LIN high, LIN>SQ)")
    c2 = all([
        g0._check(log, "power-encoded message: SQ recovers (>0.9) and SQ > LIN", pa > 0.9 and pa > la + 0.1,
                  f"SQ {pa:.3f} vs LIN {la:.3f}"),
        g0._check(log, "linear-encoded message: LIN recovers (>0.9) and LIN > SQ", lb > 0.9 and lb > pb + 0.1,
                  f"LIN {lb:.3f} vs SQ {pb:.3f}"),
    ])
    R["check2_isolation"] = {"pass": c2, "power_enc": [fa, pa, la], "linear_enc": [fb, pb, lb]}

    # ---- CHECK 3: decoy protocol == Phase-1, proven vs the committed base-40000 path ---------- #
    log("\n(3) Decoy protocol -- gate3 _band_decoys(base=40000) reproduces Phase-1 run decoys byte-exact")
    gen_ok = p1.slow_bandlimited.__module__ == "D_phase1_routing"          # committed generator, not a copy
    sub_band_ok = BANDS["SUB"] == (p1.MSG_LO, p1.MSG_HI)                   # SUB band == Phase-1 message band
    # gate3's SHIPPED builder, forced to Phase-1's base 40000, must reproduce Phase-1 run()'s ACTUAL
    # decoys byte-exact -> proves generator + band + stride (i*200+d); the only gate3 change is the base.
    proto_ok = True
    for i in (0, 3, 7):
        g3 = _band_decoys("SUB", i, L, dt_in, base=40000)
        ref = [p1.slow_bandlimited(L, dt_in, p1.MSG_LO, p1.MSG_HI, seed=40000 + i * 200 + d)
               for d in range(N_DEC)]
        proto_ok &= (len(g3) == N_DEC and all(np.array_equal(a, r) for a, r in zip(g3, ref)))
    # wrong-band guard: the band arg must be honoured -> SUB vs RES decoys differ at the same (i,d)
    band_used = not np.array_equal(_band_decoys("SUB", 0, L, dt_in)[0], _band_decoys("RES", 0, L, dt_in)[0])
    # stride guard: adjacent (i,d) draw distinct seeds -> distinct decoys
    stride_ok = not np.array_equal(_band_decoys("RES", 0, L, dt_in)[0], _band_decoys("RES", 1, L, dt_in)[0])
    c3 = all([
        g0._check(log, "decoy generator is the committed p1.slow_bandlimited", gen_ok, ""),
        g0._check(log, "SUB decoy band == Phase-1 message band [0.2,0.9]", sub_band_ok, f"{BANDS['SUB']}"),
        g0._check(log, "gate3 _band_decoys(SUB,base=40000) == Phase-1 run() decoys byte-exact (gen+band+stride)",
                  proto_ok, "i in {0,3,7} x N_DEC draws vs the committed base-40000 path"),
        g0._check(log, "band arg honoured: SUB vs RES decoys differ at same (i,d)", band_used, ""),
        g0._check(log, "stride i*200+d: RES seed i=0 vs i=1 give distinct decoys", stride_ok, ""),
    ])
    R["check3_decoy"] = {"pass": c3, "gen_ok": gen_ok, "sub_band_ok": sub_band_ok, "proto_ok": proto_ok}

    # ---- CHECK 4: validity-guard + classifier + consequence map ------------------------- #
    log("\n(4) Classifier + validity-guard + consequence map (synthetic matrices)")
    c4 = all([
        g0._check(log, "SQ-carried when R_SQ>=0.9 and R_LIN<0.5", _classify(0.95, 0.30) == "SQ-carried", ""),
        g0._check(log, "LIN-carried when R_LIN>=0.9 and R_SQ<0.5", _classify(0.30, 0.95) == "LIN-carried", ""),
        g0._check(log, "MIXED when both high", _classify(0.95, 0.95) == "MIXED", ""),
        g0._check(log, "MIXED when neither clears", _classify(0.70, 0.60) == "MIXED", ""),
        g0._check(log, "consequence: SUB=SQ & RES=LIN -> C3 scope note",
                  "SCOPE NOTE" in _consequence({"SUB": {"classification": "SQ-carried", "classifies": True},
                                                "RES": {"classification": "LIN-carried", "classifies": True},
                                                "SUPRA": {"classification": "MIXED", "classifies": True}}), ""),
        g0._check(log, "consequence: SQ everywhere -> C3 STANDS",
                  "C3 STANDS" in _consequence({b: {"classification": "SQ-carried", "classifies": True}
                                               for b in BAND_ORDER}), ""),
        g0._check(log, "consequence: SUB=SQ & RES dark -> OFF-BAND UNTESTED (RES-keyed; catches SQ/BG/SQ)",
                  "OFF-BAND UNTESTED" in _consequence(
                      {"SUB": {"classification": "SQ-carried", "classifies": True},
                       "RES": {"classification": "below-guard", "classifies": False},
                       "SUPRA": {"classification": "SQ-carried", "classifies": True}}), ""),
        g0._check(log, "consequence: SUB=SQ,RES=SQ,SUPRA dark -> C3 STANDS (RES classifies; SUPRA legit dark)",
                  "C3 STANDS" in _consequence(
                      {"SUB": {"classification": "SQ-carried", "classifies": True},
                       "RES": {"classification": "SQ-carried", "classifies": True},
                       "SUPRA": {"classification": "below-guard", "classifies": False}}), ""),
        g0._check(log, "consequence: other -> report as-is",
                  "AS-IS" in _consequence({"SUB": {"classification": "LIN-carried", "classifies": True},
                                           "RES": {"classification": "MIXED", "classifies": True},
                                           "SUPRA": {"classification": "MIXED", "classifies": True}}), ""),
    ])
    # validity guard: a band with FULL r2 <= 0.2 does not classify
    fake = {(b, i): {"r2": {"FULL": (0.98 if b != "SUPRA" else 0.05), "SQ": 0.9, "LIN": 0.2},
                     "decoy_p95": {"FULL": -0.1, "SQ": -0.1, "LIN": -0.1},
                     "esp": {"ok_slow": True}} for b in BAND_ORDER for i in range(5)}
    ret_guard = _retention(fake, list(range(5)))
    c4 = c4 and g0._check(log, "validity guard drops SUPRA (FULL r2=0.05 <= 0.2)",
                          (not ret_guard["SUPRA"]["classifies"]) and ret_guard["SUB"]["classifies"],
                          f"SUPRA classifies={ret_guard['SUPRA']['classifies']}")
    R["check4_classifier"] = {"pass": c4}

    # ---- CHECK 5: ESP-honest intersection across bands ---------------------------------- #
    log("\n(5) ESP-honest intersection -- a seed failing ESP in ANY band is dropped")
    esprec = {(b, i): {"esp": {"ok_slow": True}} for b in BAND_ORDER for i in range(5)}
    esprec[("RES", 2)]["esp"]["ok_slow"] = False       # seed 2 fails ESP in RES only
    esprec[("SUPRA", 4)]["esp"]["ok_slow"] = False      # seed 4 fails ESP in SUPRA only
    inter = [i for i in range(5) if all(esprec[(b, i)]["esp"]["ok_slow"] for b in BAND_ORDER)]
    c5 = g0._check(log, "intersection = seeds ESP-ok in ALL 3 bands", inter == [0, 1, 3],
                   f"intersection={inter} (2 fails RES, 4 fails SUPRA)")
    R["check5_intersection"] = {"pass": c5, "intersection": inter}

    # ---- CHECK 6: anchor plumbing (REF_TABLE + am_input_band[SUB] BIT-IDENTICAL to am_input) --- #
    log("\n(6) Anchor plumbing -- REF_TABLE expectation + SUB message BIT-IDENTICAL to Phase-1 am_input")
    ref = REF_TABLE[(STAGE_SPAN, K_PRIMARY)]
    seed0 = round(float(ref[0][0]), 6)
    mean07 = np.mean([ref[i][0] for i in range(8)])
    # bit-identity: am_input_band([0.2,0.9], seed) MUST equal p1.am_input(seed) array-for-array (both the
    # message s AND the AM drive u) -- digit-exact 0.981470 needs IDENTICAL arrays, not math-equivalence
    # (a generic band generator could differ in RNG consumption or float-op order).
    dt_c, _W0c, _esc, Lc, _delc, _stc = p1.am_window(STAGE_SPAN, n_msg=8)
    ident = []
    for i in (0, 1, 3, 7):
        s1, u1 = g0.am_input_band(Lc, dt_c, MSG_BASE + i, p1.MSG_LO, p1.MSG_HI)
        s0, u0 = p1.am_input(Lc, dt_c, MSG_BASE + i)
        ident.append(bool(np.array_equal(s1, s0) and np.array_equal(u1, u0)))
    ident_ok = all(ident)
    c6 = all([
        g0._check(log, "REF_TABLE(1.5,0.24) seed 0 = 0.981470", seed0 == 0.981470, f"got {seed0}"),
        g0._check(log, "anchor mean(seeds 0-7) within window of 0.986", abs(mean07 - ANCHOR) <= 0.02,
                  f"mean {mean07:.6f}"),
        g0._check(log, "am_input_band([0.2,0.9],seed) BIT-IDENTICAL to p1.am_input(seed) -- s AND u arrays",
                  ident_ok, f"seeds 0/1/3/7 array_equal (s&u) = {ident}"),
    ])
    R["check6_anchor"] = {"pass": c6, "ref_seed0": seed0, "ref_mean_0_7": float(mean07),
                          "sub_msg_bit_identical": ident_ok}

    # ---- CHECK 7: band geometry (slow/fast tertile ranges vs SUB/RES/SUPRA) ------------- #
    log("\n(7) Band geometry -- slow tertile range vs RES [2,9]; SUPRA above slow, in fast")
    slo, shi = g0.slow_tertile_omega(STAGE_SPAN)
    om = p1.build_system(0, N, STAGE_SPAN).omega
    fidx = p1.band_indices(om)["fast"]
    flo, fhi = float(om[fidx].min()), float(om[fidx].max())
    log(f"    slow tertile natural range [{slo:.2f}, {shi:.2f}]; fast [{flo:.2f}, {fhi:.2f}]; "
        f"Nyquist-ish check via SUPRA hi {BANDS['SUPRA'][1]}")
    geom_ok = (BANDS["SUB"][1] < slo) and (BANDS["RES"][0] <= shi and BANDS["RES"][1] >= slo) and \
              (BANDS["SUPRA"][0] > shi and BANDS["SUPRA"][0] >= flo * 0.99)
    c7 = g0._check(log, "SUB below slow; RES overlaps slow; SUPRA above slow, inside fast", geom_ok,
                   f"SUB {BANDS['SUB']} RES {BANDS['RES']} SUPRA {BANDS['SUPRA']} vs slow[{slo:.2f},{shi:.2f}] fast[{flo:.2f},{fhi:.2f}]")
    R["check7_band_geometry"] = {"pass": c7, "slow": [slo, shi], "fast": [flo, fhi]}

    order = ["check0_seed_scheme", "check1_ablation_subset", "check2_isolation", "check3_decoy",
             "check4_classifier", "check5_intersection", "check6_anchor", "check7_band_geometry"]
    allpass = all(R[k]["pass"] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if R[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate3_sandbox.json")
    with open(outp, "w") as f:
        json.dump({"gate": "relay-gate3", "stage": "1-cpu-sandbox", "all_pass": allpass,
                   "framing": FRAMING, "checks": R}, f, indent=1, default=g1._json_default)
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


# ===================================================================================== #
#  Synthetic verdict-engine test (CPU; consequence-map + NO-MEASUREMENT branches)
# ===================================================================================== #
def _synth_recs(band_cls, seeds, anchor_full=None, decoy=-0.1, esp_fail=None):
    """Build synthetic recs realizing a target per-band channel. band_cls: {band: 'SQ'|'LIN'|'MIX'|'LOW'}.
    Retention ratios are set as fractions of FULL so SUB (FULL~0.98) still clears the 0.9 bar.
    anchor_full: dict seed->FULL r2 for SUB (default reproduces REF_TABLE so the anchor passes)."""
    ref = REF_TABLE[(STAGE_SPAN, K_PRIMARY)]
    ratios = {"SQ": (0.97, 0.20), "LIN": (0.20, 0.97), "MIX": (0.94, 0.89)}
    recs = {}
    for b in BAND_ORDER:
        for i in seeds:
            if band_cls[b] == "LOW":                        # below validity guard (FULL <= 0.2)
                full, rsq, rlin = 0.05, 0.4, 0.4
            else:
                full = (anchor_full["sub"](i) if (b == "SUB" and anchor_full)
                        else float(ref[i][0]) if b == "SUB" else 0.90)
                rsq, rlin = ratios[band_cls[b]]
            ok = not (esp_fail and (b, i) in esp_fail)
            recs[(b, i)] = {"band": b, "seed": i,
                            "r2": {"FULL": full, "SQ": full * rsq, "LIN": full * rlin},
                            "decoy_p95": {"FULL": decoy, "SQ": decoy, "LIN": decoy},
                            "esp": {"ok_slow": ok}}
    return recs


def verdict_test(log):
    log("=== VERDICT-ENGINE SYNTHETIC TEST (consequence map + NO-MEASUREMENT; CPU only) ===")
    seeds = list(range(6))
    allok = True

    def run_case(name, cls, want, **kw):
        nonlocal allok
        v = decide(_synth_recs(cls, seeds, **kw), seeds)
        ok = want in v["verdict"]
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {v['verdict'][:88]}")
        return v

    vSN = run_case("SUB=SQ,RES=LIN -> C3 scope note", {"SUB": "SQ", "RES": "LIN", "SUPRA": "MIX"}, "SCOPE NOTE")
    vST = run_case("SQ everywhere -> C3 stands", {"SUB": "SQ", "RES": "SQ", "SUPRA": "SQ"}, "C3 STANDS")
    run_case("mixed pattern -> report as-is", {"SUB": "LIN", "RES": "MIX", "SUPRA": "MIX"}, "AS-IS")
    run_case("SUPRA below guard -> classifies only SUB/RES", {"SUB": "SQ", "RES": "LIN", "SUPRA": "LOW"}, "SUPRA=below-guard")
    # OFF-BAND UNTESTED (RES-keyed): RES dark -> no scope verdict, SUPRA cannot rescue
    run_case("SQ / RES-dark / SUPRA-dark -> OFF-BAND UNTESTED", {"SUB": "SQ", "RES": "LOW", "SUPRA": "LOW"}, "OFF-BAND UNTESTED")
    run_case("SQ / RES-dark / SUPRA=SQ -> OFF-BAND UNTESTED (not C3-stands)", {"SUB": "SQ", "RES": "LOW", "SUPRA": "SQ"}, "OFF-BAND UNTESTED")
    run_case("SQ / RES=SQ / SUPRA-dark -> C3 STANDS (RES classifies)", {"SUB": "SQ", "RES": "SQ", "SUPRA": "LOW"}, "C3 STANDS")
    # ratified scope tag: appended on the C3-STANDS branch, ABSENT on the scope-note branch
    tag_ok = ("[SCOPED --" in vST["verdict"]) and ("[SCOPED --" not in vSN["verdict"])
    allok &= tag_ok
    log(f"  [{'OK' if tag_ok else 'WRONG'}] scope tag on C3-STANDS verdict line, absent on scope-note")
    # NO-MEASUREMENT branches
    run_case("anchor miss (FULL x SUB shifted low)", {"SUB": "SQ", "RES": "LIN", "SUPRA": "MIX"},
             "anchor miss", anchor_full={"sub": lambda i: 0.90})
    run_case("decoy elevated", {"SUB": "SQ", "RES": "LIN", "SUPRA": "MIX"}, "decoy elevated", decoy=0.35)
    vU = decide(_synth_recs({"SUB": "SQ", "RES": "LIN", "SUPRA": "MIX"}, seeds,
                            esp_fail={(b, i) for b in BAND_ORDER for i in range(5)}), seeds)
    okU = "underpowered" in vU["verdict"]
    allok &= okU
    log(f"  [{'OK' if okU else 'WRONG'}] ESP fail all-but-one -> underpowered: {vU['verdict'][:60]}")

    import tempfile
    for tag, vv in (("scope", vSN), ("stands", vST)):
        p = os.path.join(tempfile.gettempdir(), f"_g3_md_{tag}.md")
        try:
            _write_md(p, vv, seeds, 0.0, {"code": "selftest", "spec": "selftest"}, {"ok": True})
            has = "observable-order matrix" in open(p).read()
            os.remove(p)
            allok &= has
            log(f"  [{'OK' if has else 'WRONG'}] _write_md({tag}) renders matrix + consequence")
        except Exception as e:
            allok = False
            log(f"  [WRONG] _write_md({tag}) crashed: {e!r}")
    log(f"  VERDICT-ENGINE: {'ALL BRANCHES CORRECT' if allok else 'BRANCH ERRORS -- FIX'}")
    return allok


# ===================================================================================== #
#  STAGE 2 -- smoke (FULL x SUB seed 0 digit-exact; one RES + one SUPRA logged)
# ===================================================================================== #
def smoke(log):
    import time
    log(f"=== RELAY GATE-3 :: STAGE-2 SMOKE (seed 0; FULL x SUB anchor + RES + SUPRA) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} n_sub={n_sub}")
    ref0 = round(float(REF_TABLE[(STAGE_SPAN, K_PRIMARY)][0][0]), 6)      # 0.981470
    t0 = time.perf_counter()
    out = {}
    for b in ("SUB", "RES", "SUPRA"):
        r = stage_a(b, 0, geom)
        out[b] = r
        log(f"  {b:>5} {BANDS[b]}: FULL {r['r2']['FULL']:+.6f} SQ {r['r2']['SQ']:+.6f} "
            f"LIN {r['r2']['LIN']:+.6f} | ESP={r['esp']['ok_slow']} (d_slow {r['esp']['d_slow']:.4g}) "
            f"| decoy p95 FULL {r['decoy_p95']['FULL']:+.3f} | msg-band {r['msg_dominant_band']}")
    wall = time.perf_counter() - t0
    got0 = round(out["SUB"]["r2"]["FULL"], 6)
    anchor_ok = got0 == ref0
    log(f"\n  ANCHOR FULL x SUB seed 0 vs committed b0f7664: {out['SUB']['r2']['FULL']:.6f} vs "
        f"{ref0:.6f} -> {'MATCH' if anchor_ok else 'MISMATCH -- STOP'}")
    smoke_pass = bool(anchor_ok and out["RES"]["r2"]["FULL"] is not None
                      and out["SUPRA"]["r2"]["FULL"] is not None)
    log(f"  wall-clock {wall:.0f}s. SMOKE: {'PASS' if smoke_pass else 'FAIL -- STOP, no battery'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate3_smoke.json")
    g0._dump_json(outp, {"gate": "relay-gate3", "stage": "2-smoke", "seed": 0, "framing": FRAMING,
                         "env": g1._env_full(), "anchor_provenance": ANCHOR_SRC,
                         "anchor_ref": ref0, "anchor_got": out["SUB"]["r2"]["FULL"],
                         "anchor_match": anchor_ok, "runs": out, "wall_clock_s": wall})
    log(f"  [written -> {os.path.relpath(outp)}]  (smoke artifact; NOT committed)")
    return smoke_pass


# ===================================================================================== #
#  STAGE 3 -- full battery (3 bands x n seeds; verdict = mechanism matrix)
# ===================================================================================== #
def run(log, nseeds):
    import time
    seeds = list(range(nseeds))
    log(f"=== RELAY GATE-3 :: STAGE-3 BATTERY (seeds {seeds}, 3 bands, K={K_PRIMARY}) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    recs, flat = {}, {}
    outp = os.path.join(RESDIR, "gate3_mechanism.json")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..", "relay_gate3_mechanism_spec.md"))}
    colrep = verify_no_collision()
    t0 = time.perf_counter()
    for i in seeds:
        for b in BAND_ORDER:
            ts = time.perf_counter()
            r = stage_a(b, i, geom)
            recs[(b, i)] = r
            flat[f"{b}|{i}"] = r
            log(f"  seed {i} {b:>5}: FULL {r['r2']['FULL']:+.4f} SQ {r['r2']['SQ']:+.4f} "
                f"LIN {r['r2']['LIN']:+.4f} ESP={r['esp']['ok_slow']} "
                f"({time.perf_counter()-ts:.0f}s; {time.perf_counter()-t0:.0f}s elapsed)")
        g0._dump_json(outp, {"gate": "relay-gate3", "stage": "3-battery", "seeds_done": seeds[:i + 1],
                             "framing": FRAMING, "K": K_PRIMARY, "span": STAGE_SPAN,
                             "seed_scheme": colrep, "hashes": hashes, "env": g1._env_full(),
                             "anchor_provenance": ANCHOR_SRC, "recs": flat})
    verdict = decide(recs, seeds)
    wall = time.perf_counter() - t0
    payload = {"gate": "relay-gate3", "stage": "3-battery", "seeds": seeds, "framing": FRAMING,
               "K": K_PRIMARY, "span": STAGE_SPAN, "bands": BANDS, "modes": MODES,
               "seed_scheme": colrep, "hashes": hashes, "env": g1._env_full(),
               "anchor_provenance": ANCHOR_SRC, "wall_clock_s": wall,
               "verdict": verdict, "recs": flat}
    g0._dump_json(outp, payload)
    _write_md(os.path.join(RESDIR, "gate3_mechanism.md"), verdict, seeds, wall, hashes, colrep)
    log("\n=== BATTERY VERDICT ===")
    log(f"  {verdict['verdict']}")
    log("  STOP-and-report. Gate-4 (hop-length) is a separate decision.")
    return verdict


def _recs_from_flat(flat):
    out = {}
    for key, r in flat.items():
        b, i = key.split("|")
        out[(b, int(i))] = r
    return out


def reread(log):
    src = os.path.join(RESDIR, "gate3_mechanism.json")
    assert os.path.exists(src), f"missing battery record {src} -- run --run first"
    nm = json.load(open(src))
    recs = _recs_from_flat(nm["recs"])
    seeds = nm["seeds"]
    log("=== RELAY GATE-3 :: REREAD (re-frame verdict from unchanged recs; NO GPU) ===")
    verdict = decide(recs, seeds)
    reflat = {f"{b}|{i}": recs[(b, i)] for (b, i) in recs}            # independent re-serialization
    assert json.dumps(reflat, sort_keys=True) == json.dumps(nm["recs"], sort_keys=True), \
        "reread recs round-trip drifted from the battery record"
    log("  [integrity] recs round-trip (flat->recs->flat) byte-identical to the battery record: OK")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..", "relay_gate3_mechanism_spec.md"))}
    # run_hashes = the ORIGINAL battery-run sha, first-write-wins: on the first reread it is captured
    # from nm['hashes'] (still the battery's); on later rereads it is PRESERVED, never overwritten by an
    # intermediate re-render sha (chain-of-custody).
    run_hashes = nm.get("run_hashes") or nm.get("hashes")
    note = (f"Provenance: numbers from the battery run (harness sha "
            f"{(run_hashes or {}).get('code','?')}, {nm.get('wall_clock_s',0)/60:.0f} min GPU); "
            f"verdict RE-FRAMED by --reread (sha {hashes['code']}), recs unchanged.")
    g0._dump_json(src, {**nm, "verdict": verdict, "hashes": hashes, "run_hashes": run_hashes,
                        "reread": "verdict re-rendered from unchanged recs; no GPU, no number changed"})
    _write_md(os.path.join(RESDIR, "gate3_mechanism.md"), verdict, seeds, nm.get("wall_clock_s", 0.0),
              hashes, nm.get("seed_scheme", {"ok": True}), note=note)
    log(f"  {verdict['verdict']}")
    log(f"  [rewritten -> {os.path.relpath(src)} + gate3_mechanism.md]  (recs UNCHANGED; NOT committed)")
    return verdict


# ===================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--nseeds", type=int, default=8, help="spec target n>=5; default 8 (seeds 0..7)")
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)

    def log(msg):
        print(msg)

    if args.sandbox:
        raise SystemExit(0 if sandbox(log) else 1)
    if args.verdict_test:
        raise SystemExit(0 if verdict_test(log) else 1)
    if args.reread:
        reread(log)
        return
    if args.smoke:
        raise SystemExit(0 if smoke(log) else 1)
    if args.run:
        assert 1 <= args.nseeds <= SEED_MAX + 1, f"REF_TABLE covers seeds 0..{SEED_MAX}"
        run(log, args.nseeds)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
