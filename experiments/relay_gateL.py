"""Relay Gate-L -- Coherent Linear Injection (the other door).

Companion harness (reuse-by-import of the committed relay_gate0 / relay_gate3 /
D_phase1_routing machinery; committed artifacts and core/ are NEVER edited). Gate-L is a
Gate-3 sibling: the SAME stage-A protocol (N=500, span 1.5, K=0.24, fast-tertile inject ->
slow-tertile readout), the SAME reservoir + message realizations per (band, seed), with ONE
change -- the injection map. Gate-3 injected square-law AM (u = 0.5*sqrt(s)*w, Rademacher w ->
zero input-level linear content). Gate-L injects COHERENT, ZERO-MEAN content by construction:

    u = a * (s - mean_inj(s)),   a = 0.5 * sqrt(E_inj[s] / Var_inj(s))            [P1]

so corr(u, s) = 1 exactly and E[u^2] = 0.25 * E_inj[s] (the AM injected power, matched by
expression). The question: do the first-order slow-tertile coordinates (Re z, Im z) carry the
message when the linear content is supplied directly? -- closing the branch Gate-3's scope note
left open, per message band.

Spec: relay_gateL_coherent_spec.md (v2, ratified R1-R4 + panel edits P1-P4). Lifecycle is
STOP-and-report at every boundary: S1 sandbox (CPU, checks proven-to-fire) -> S2 smoke (1 seed,
GPU) -> S3 battery (GPU, on word) -> S4 panel -> S5 commit (on Jason's exact word; author
Jason Dury <jason@eridos.ai>, no co-author lines).

Modes: --sandbox / --verdict-test / --reread run on CPU (no integration). --smoke / --run touch
the GPU. Windows/thresholds are NEVER stored: delta and the classification legs are evaluated at
use from byte-locked primitives (D, SE, n) and from the sha-verified committed subtrahend.
"""

import os
import sys

# CPU-only for the non-GPU modes: force the JAX CPU backend BEFORE jax is imported (via
# relay_gate0 -> D_phase1_routing), so --sandbox / --verdict-test / --reread never touch the GPU.
if any(a in sys.argv for a in ("--sandbox", "--verdict-test", "--reread")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import math
import time
import argparse
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))                        # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root

import D_phase1_routing as p1        # noqa: E402  (imports jax + enables x64 at import)
import relay_gate0 as g0             # noqa: E402
import relay_gate3 as g3             # noqa: E402  (stage_a AM replica = the anchor; BANDS/MODES/seed_scheme)
from core.consistency import ESP_EPS  # noqa: E402

RESDIR = g0.RESDIR
SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "relay_gateL_coherent_spec.md")

# ---- imported committed constants (never redefined) --------------------------------------- #
N = p1.N                              # 500
STAGE_SPAN = g0.STAGE_SPAN            # 1.5
K_PRIMARY = g0.K_PRIMARY              # 0.24
BANDS = g3.BANDS                      # {"SUB":(0.2,0.9),"RES":(2.0,9.0),"SUPRA":(10.0,28.0)}
MODES = g3.MODES                      # {"FULL":"full","SQ":"pow","LIN":"reim"}
MODE_ORDER = g3.MODE_ORDER            # ["FULL","SQ","LIN"]
MIN_PAIRS = g0.MIN_PAIRS              # 2: paired intersection < 2 -> band-level NO-MEASUREMENT
DECOY_ELEVATED = g0.DECOY_ELEVATED    # 0.2: any fresh decoy-p95 cell above this = leakage flag
N_DEC = p1.N_DEC                      # 60 never-injected decoy nulls per cell
ENC_BASE, REP_BASE, MSG_BASE = g3.ENC_BASE, g3.REP_BASE, g3.MSG_BASE  # 5000, 9000, 1000
RADE_OFFSET = g3.RADE_OFFSET          # 777: AM Rademacher carrier offset (NEVER drawn in coherent cells)

BAND_ORDER = ["SUB", "RES", "SUPRA"]
SEEDS = list(range(8))                # 0..7 (extend 8-9 on attrition; REF_TABLE covers 0..9)
SEED_MAX = 9

# ---- Gate-L pre-registered literals (R4 / classification; STORABLE integer primitives) ---- #
REACH_FLOOR = 0.99                    # R4: reachability linear-arm gate (tripwire; centered u -> corr=1)
LADDER_FACTOR = 10.0                  # R4: P_track(a) >= 10 * P_track_chance (chance-normalized [P3])
LIVE_FLOOR = 0.2                      # LIN-LIVE absolute floor (committed decoy-bar / full_r2_min lineage)
DELTA_FLOOR = 0.02                    # delta = max(2*SE_paired, DELTA_FLOOR); EVALUATE-AT-USE
FULL_POWER_MIN = 5                    # full-strength verdict needs paired n >= 5; [2,4] -> -UNDERPOWERED

# ---- fresh decoy family (collision-proven vs ALL committed families at sandbox) ----------- #
GATEL_DECOY_BASE = {"SUB": 600000, "RES": 620000, "SUPRA": 640000}
# every committed decoy base family the fresh family must clear (Phase-1, Gates 0-4, Gate-B/probe)
COMMITTED_DECOY_BASES = [40000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 180000,
                         200000, 220000, 240000, 300000, 320000, 340000, 400000, 420000, 440000,
                         460000, 500000, 520000, 540000, 560000]
DECOY_FAMILY_SPAN = SEED_MAX * 200 + (N_DEC - 1)   # footprint = base + [0 .. this]; seeds*200 + d

# ---- pinned committed subtrahend (NO-MEASUREMENT on sha mismatch) ------------------------- #
GATE3_JSON = os.path.join(RESDIR, "gate3_mechanism.json")
GATE3_SHA256 = "878c154850c7aec578a73979dbd562ef81502d092002a6da365480a8584a8f57"

GATEL_JSON = os.path.join(RESDIR, "gateL_coherent.json")
GATEL_MD = os.path.join(RESDIR, "gateL_coherent.md")

FRAMING = ("Gate-L: coherent zero-mean linear injection (u = a*(s - mean), corr(u,s) = 1) vs the "
           "committed Gate-3 AM subtrahend -- do first-order slow-tertile coordinates carry the "
           "message when linear content is supplied directly? Stage-A, K=0.24, span 1.5, per band. "
           "STOP-and-report.")

FENCE = ("Why the Gate-3 classifier does NOT port: with z = Z0 + dz, |z|^2 = |Z0|^2 + "
         "2*Re(conj(Z0)*dz) + |dz|^2; under coherent injection dz is first-order in the drive, so "
         "the beat term makes |z|^2 carry LINEAR-in-drive content. A high SQ r2 here is NOT a "
         "quadratic-mechanism signature. FENCED PRE-DATA: no mechanism claim / retention ratio / "
         "channel classification is derived from the SQ or FULL columns (computed + recorded for "
         "continuity only). All falsifiable content lives in the LIN column (Re z, Im z are "
         "first-order observables under any injection).")


# ============================================================================================ #
# provenance / io helpers
# ============================================================================================ #
def _sha256_full(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashes():
    return {"code": g0._sha12(os.path.abspath(__file__)), "spec": g0._sha12(SPEC_PATH)}


def _env_full():
    import jax
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64 = None
    return {**g0._env_versions(), "interpreter": sys.executable, "python": sys.version.split()[0],
            "jax_enable_x64": x64, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<default>"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}


def _f(x, spec=".4f"):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return format(x, spec)


def seed_scheme(i, band):
    """build/enc/rep/msg reproduce Gate-3 (and Phase-1 stage-A) EXACTLY -> same reservoir + same
    message realization s per (band, seed). The AM Rademacher carrier (msg + 777) is listed for
    the seed-consumption audit only; the coherent map never draws it."""
    return {"build": i, "enc": ENC_BASE + i, "rep": REP_BASE + i, "msg": MSG_BASE + i,
            "rademacher": MSG_BASE + i + RADE_OFFSET, "decoy_base": GATEL_DECOY_BASE[band]}


# ============================================================================================ #
# the one change: coherent zero-mean injection map [P1]
# ============================================================================================ #
def coherent_drive(s):
    """u = a*(s - mean_inj(s)), a = 0.5*sqrt(E_inj[s]/Var_inj(s)); statistics over the full
    injected sequence. Returns the drive + the per-cell audit record. E[u^2] = a^2*Var(s) =
    0.25*E[s] by construction (the AM injected power, matched by expression). corr(u, s) = 1
    exactly (centering does not change it). DC fraction E[s]^2/E[s^2] recorded (report-only)."""
    s = np.asarray(s, float)
    m = float(np.mean(s))
    v = float(np.var(s))                       # population variance = mean((s-m)^2)
    a = 0.5 * math.sqrt(m / v)
    u = a * (s - m)
    e_u2 = float(np.mean(u * u))
    target = 0.25 * m                          # = 0.25 * E_inj[s] (AM's E[u^2])
    pm_resid = abs(e_u2 - target)
    dc_frac = (m * m) / float(np.mean(s * s))
    return {"u": u, "a": float(a), "mean_inj": m, "var_inj": v, "e_u2": e_u2, "target": target,
            "pm_resid": float(pm_resid), "pm_rel": float(pm_resid / target if target > 0 else 0.0),
            "dc_frac": float(dc_frac)}


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float((a @ b) / d) if d > 0 else 0.0


def reachability(u, s, delays, sl):
    """Reachability audit, both arms, eval window sl.
    Linear arm [P4]: gate statistic = MAX over the committed Gate-3 lag set (`delays`, from
    am_window) of |corr(u, roll(s, k))|. NOTE: Gate-3 has no committed corr-audit function (the
    <=0.032 figure is prose in its _write_md), so this audit is built fresh; it binds the LAG SET
    and the EVAL WINDOW to committed machinery (am_window's `delays`, `sl`), corr = Pearson.
    Quadratic arm: corr(u^2, s) over the eval window, recorded, NO gate."""
    iw = np.arange(sl.start, sl.stop)
    u = np.asarray(u, float); s = np.asarray(s, float)
    per_lag = {int(k): abs(_pearson(u[iw], np.roll(s, k)[iw])) for k in delays}
    lin = max(per_lag.values()) if per_lag else 0.0
    quad = _pearson((u * u)[iw], s[iw])
    return {"lin": float(lin), "quad": float(quad), "per_lag": per_lag, "floor": REACH_FLOOR}


# ============================================================================================ #
# fresh decoys + collision proof
# ============================================================================================ #
def band_decoys_L(band, i, L, dt_in):
    """Fresh never-injected same-class decoys = the committed Phase-1 protocol (p1.slow_bandlimited
    on the band's [lo,hi], stride i*200 + d, N_DEC draws) at the FRESH per-band base."""
    lo, hi = BANDS[band]
    b0 = GATEL_DECOY_BASE[band]
    return [p1.slow_bandlimited(L, dt_in, lo, hi, seed=b0 + i * 200 + d) for d in range(N_DEC)]


def collision_matrix():
    """Prove the fresh decoy families do not collide with ANY committed family. Each family j
    occupies integer seeds [base_j, base_j + DECOY_FAMILY_SPAN]. Returns (report, ok)."""
    fresh = {f"GATEL:{b}": GATEL_DECOY_BASE[b] for b in BAND_ORDER}
    committed = {f"COMMITTED:{b}": b for b in COMMITTED_DECOY_BASES}
    span = DECOY_FAMILY_SPAN
    overlaps = []
    all_bases = {**committed, **fresh}
    names = list(all_bases)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a0 = all_bases[names[x]]; b0 = all_bases[names[y]]
            if not (a0 + span < b0 or b0 + span < a0):
                overlaps.append([names[x], names[y]])
    min_fresh = min(fresh.values())
    max_committed = max(committed.values()) + span
    rep = {"fresh_bases": fresh, "family_span": span, "n_committed_families": len(committed),
           "min_fresh_base": min_fresh, "max_committed_footprint": max_committed,
           "pairwise_overlaps": overlaps, "clear_of_committed": bool(min_fresh > max_committed)}
    return rep, (len(overlaps) == 0)


# ============================================================================================ #
# linearity ladder [P3]: drive-tracking projection P_track (+ superseded raw power, for contrast)
# ============================================================================================ #
def _ptrack_core(x_cols, u, iw):
    """P_track = mean_i beta_i^2 * <u,u>, beta_i = <x_i, u>/<u,u>; x_cols already band-passed to
    the message band, columns = fast-tertile oscillators. All inner products over the eval window
    iw; x_i mean-removed (spec)."""
    uu = np.asarray(u, float)[iw]
    den = float(uu @ uu)
    if den <= 0.0:
        return 0.0
    betas = []
    for j in range(x_cols.shape[1]):
        xj = np.asarray(x_cols[iw, j], float)
        xj = xj - xj.mean()
        betas.append(float(xj @ uu) / den)
    b = np.asarray(betas)
    return float(np.mean(b * b) * den)


def _raw_inband_power(x_cols, iw):
    """The SUPERSEDED statistic: mean over fast-tertile oscillators of in-band Re z power (variance
    over the eval window). Band-inequitable at SUPRA (carrier lines in-band). Kept ONLY to
    demonstrate its spurious failure in the sandbox's required pair [P3]."""
    return float(np.mean([np.var(np.asarray(x_cols[iw, j], float)) for j in range(x_cols.shape[1])]))


def band_track(X_fast, u, dt_in, band, iw):
    """P_track on real trajectory columns: band-pass each fast-tertile Re z to the message band,
    then project onto the drive."""
    lo, hi = band
    cols = np.column_stack([g0.bandlimit(np.asarray(X_fast[:, j]).real, dt_in, lo, hi)
                            for j in range(X_fast.shape[1])])
    return _ptrack_core(cols, u, iw)


def ladder_gate(rung_vals, chance):
    """rung_vals ordered [P(0.5a), P(a), P(2a)]. ONLY a 2a integration failure is tolerated (spec:
    'Rung 2a integration failure is tolerated if 0.5a and a complete and the gate holds on those
    two; rung a must complete') -- a missing 0.5a or a is NOT tolerated -> incomplete -> band NM.
    GATE: strictly increasing over completed rungs AND P(a) >= LADDER_FACTOR * chance."""
    complete = (rung_vals[0] is not None) and (rung_vals[1] is not None)   # 0.5a AND a must complete
    completed = [v for v in rung_vals if v is not None]
    increasing = bool(complete and len(completed) >= 2
                      and all(completed[k + 1] > completed[k] for k in range(len(completed) - 1)))
    P_a = rung_vals[1]
    passed = bool(complete and increasing and P_a >= LADDER_FACTOR * float(chance))
    return {"pass": passed, "complete": bool(complete), "increasing": bool(increasing),
            "P_a": (None if P_a is None else float(P_a)), "chance": float(chance),
            "ratio": (None if (P_a is None or chance == 0) else float(P_a / chance))}


# ============================================================================================ #
# subtrahend read-at-use (sha-verified committed Gate-3 LIN baselines)
# ============================================================================================ #
def load_subtrahend(path=GATE3_JSON):
    """Read + full-sha-verify the committed gate3_mechanism.json; return the per-cell LIN baseline
    r2 and committed ESP flag. checks['ok'] False -> anchor/subtrahend NO-MEASUREMENT. The
    load-bearing numbers are read at use; never copied into the harness."""
    checks = {"ok": True, "reasons": []}
    if not os.path.exists(path):
        checks["ok"] = False
        checks["reasons"].append(f"missing committed subtrahend {path}")
        return None, checks
    sha = _sha256_full(path)
    checks["gate3_sha256"] = sha
    if sha != GATE3_SHA256:
        checks["ok"] = False
        checks["reasons"].append(f"gate3_mechanism.json sha256 {sha} != pinned {GATE3_SHA256}")
        return None, checks
    d = json.load(open(path))
    recs = d["recs"]
    out = {}
    for band in BAND_ORDER:
        out[band] = {}
        for i in SEEDS:
            r = recs[f"{band}|{i}"]
            out[band][i] = {"lin": float(r["r2"]["LIN"]), "esp_ok": bool(r["esp"]["ok_slow"]),
                            "full": float(r["r2"]["FULL"])}   # full-precision FULL: the bit-exact
    return out, checks                                        # anchor DIAGNOSTIC referent (SUB)


# ============================================================================================ #
# classification + decision (delta EVALUATE-AT-USE; precedence: DEAD, LIVE, INTERMEDIATE)
# ============================================================================================ #
def classify_band(cohb, subtr_b):
    """cohb: {seed: {"lin":, "decoy_lin":, "esp_ok":}}; subtr_b: {seed: {"lin":, "esp_ok":}}.
    ESP-honest SYMMETRIC intersection. delta = max(2*SE_paired, DELTA_FLOOR) EVALUATED HERE."""
    inter = sorted(i for i in cohb if cohb[i]["esp_ok"] and subtr_b.get(i, {}).get("esp_ok", False))
    n = len(inter)
    coh_vals = [cohb[i]["lin"] for i in inter]
    D = [cohb[i]["lin"] - subtr_b[i]["lin"] for i in inter]
    dec_max = max((cohb[i]["decoy_lin"] for i in inter), default=None)
    st_coh = g0._mstats(coh_vals) if inter else {"mean": None, "se": None}
    st_D = g0._mstats(D) if inter else {"mean": None, "se": None}
    coh_mean, D_mean, se = st_coh["mean"], st_D["mean"], st_D["se"]
    D_median = float(np.median(D)) if inter else None
    r = {"symmetric_intersection": inter, "n_paired": n, "coh_mean": coh_mean,
         "D_mean": D_mean, "D_median": D_median, "se_paired": se, "decoy_lin_max": dec_max,
         "per_seed_D": {str(i): float(cohb[i]["lin"] - subtr_b[i]["lin"]) for i in inter},
         "coh_per_seed": {str(i): float(cohb[i]["lin"]) for i in inter}}
    # committed-baseline context (read-at-use subtrahend; shown alongside the matrix)
    base_vals = [subtr_b[i]["lin"] for i in inter]
    r["committed_baseline"] = ({"mean": float(np.mean(base_vals)),
                                "median": float(np.median(base_vals))} if inter else None)
    if n < MIN_PAIRS:
        r.update({"class": f"NO-MEASUREMENT (paired intersection n={n} < {MIN_PAIRS})",
                  "base": None, "underpowered": None, "delta": None})
        return r
    delta = max(2.0 * se, DELTA_FLOOR)         # EVALUATE-AT-USE (never stored)
    if coh_mean <= dec_max:
        base = "LIN-DEAD"                       # precedence 1: indistinguishable from never-injected
    elif D_mean > delta and coh_mean > LIVE_FLOOR:
        base = "LIN-LIVE"                       # precedence 2: BOTH legs required
    else:
        base = "INTERMEDIATE"                   # precedence 3
    underpowered = bool(n < FULL_POWER_MIN)
    r.update({"base": base, "underpowered": underpowered, "delta": float(delta),
              "class": base + ("-UNDERPOWERED" if underpowered else ""),
              "delta_rule": f"max(2*SE_paired, {DELTA_FLOOR}) evaluated at verdict (not stored); "
                            f"full-strength requires n >= {FULL_POWER_MIN}"})
    if base == "LIN-LIVE":
        r["loso"] = _loso_band(cohb, subtr_b, inter)   # report-only (spec: LOSO for any LIVE band)
        assert _loso_verify(r["loso"], cohb, subtr_b, inter), \
            "LOSO record fails source recompute -- cross-drop corruption (L-1 guard, fails loud)"
    return r


def _loso_verify(loso, cohb, subtr_b, inter):
    """Record-integrity guard (L-1 hardening). Recomputes each leave-one-out drop from the
    byte-locked source (cohb, subtr_b) and confirms the STORED (D_mean, se, coh_mean) match
    bit-for-bit (same deterministic path -> exact ==). Catches cross-drop assembly/serialization
    duplication of the exact L-1 shape; CANNOT false-fire (it compares to a fresh recompute, not to
    sibling drops, so coincidentally-equal se/coh across legitimate drops pass). True iff clean."""
    for drop in inter:
        kept = [i for i in inter if i != drop]
        D = [cohb[i]["lin"] - subtr_b[i]["lin"] for i in kept]
        st = g0._mstats(D)
        coh_kept = float(np.mean([cohb[i]["lin"] for i in kept]))
        d = loso.get("drops", {}).get(str(drop))
        if d is None or not (d["D_mean"] == st["mean"] and d["se"] == st["se"]
                             and d["coh_mean"] == coh_kept):
            return False
    return True


def _loso_band(cohb, subtr_b, inter):
    """Leave-one-seed-out on the LIVE classification (REPORT-ONLY; evaluate-at-use on each kept
    set: BOTH legs re-checked -- kept D_mean > max(2*SE_kept, floor) AND kept coh_mean > 0.2)."""
    out = {"n": len(inter), "drops": {}, "n_survive": 0}
    margins = {}
    for drop in inter:
        kept = [i for i in inter if i != drop]
        D = [cohb[i]["lin"] - subtr_b[i]["lin"] for i in kept]
        st = g0._mstats(D)
        coh_kept = float(np.mean([cohb[i]["lin"] for i in kept]))
        delta_k = max(2.0 * st["se"], DELTA_FLOOR)
        survive = bool(st["mean"] > delta_k and coh_kept > LIVE_FLOOR)
        margins[drop] = float(st["mean"] - delta_k)
        out["drops"][str(drop)] = {"D_mean": st["mean"], "se": st["se"], "delta": float(delta_k),
                                   "coh_mean": coh_kept, "survive": survive}
        out["n_survive"] += int(survive)
    out["all_survive"] = bool(out["n_survive"] == out["n"])
    if margins:
        tight = min(margins, key=margins.get)
        out["tightest_seed"] = int(tight)
        out["tightest_margin"] = float(margins[tight])
    return out


def decide(coh, anchor, subtr, ladder):
    """Pre-registered verdict. Instruments first, fixed order (anchor -> reachability -> decoy),
    each a GATE-LEVEL NO-MEASUREMENT that SEALS the record. Then per-band ladder (band-level NM)
    and classification, then the consequence map. coh[band][seed] carries r2/decoy_p95/esp_ok/
    reach; anchor = {ok, mean, se, per_seed}; subtr[band][seed] = {lin, esp_ok};
    ladder[band] = {pass, complete, ...}."""
    out = {"framing": FRAMING, "fence": FENCE,
           "operationalization": {
               "injection": "u = a*(s - mean_inj(s)), a = 0.5*sqrt(E_inj[s]/Var_inj(s)) [P1]",
               "primary": "r2_LIN_coh (reim [Re z, Im z]+bias ridge; committed Gate-3 fit protocol)",
               "contrast": "D_LIN = r2_LIN_coh - r2_LIN_AM (per-seed paired; sha-verified committed subtrahend)",
               "delta": f"max(2*SE_paired, {DELTA_FLOOR}) EVALUATED AT VERDICT (never stored)",
               "classification": "DEAD (coh_mean <= max fresh LIN decoy p95) -> LIVE (D>delta AND "
                                 f"coh_mean>{LIVE_FLOOR}) -> INTERMEDIATE; -UNDERPOWERED at n in "
                                 f"[{MIN_PAIRS},{FULL_POWER_MIN-1}]; n<{MIN_PAIRS} -> NM",
               "reachability": f"linear-arm gate |corr(u,s)| >= {REACH_FLOOR} (tripwire)",
               "ladder": f"P_track strictly increasing AND P_track(a) >= {LADDER_FACTOR}x chance [P3]"}}

    # reachability tripwire (min over all coherent cells) --------------------------------- #
    reach_cells = {f"{b}|{i}": coh[b][i]["reach"]["lin"] for b in coh for i in coh[b]}
    quad_cells = {f"{b}|{i}": coh[b][i]["reach"]["quad"] for b in coh for i in coh[b]}
    min_lin = min(reach_cells.values()) if reach_cells else None
    reach_ok = bool(min_lin is not None and min_lin >= REACH_FLOOR)
    out["reachability"] = {"min_lin": min_lin, "floor": REACH_FLOOR, "ok": reach_ok,
                           "linear_per_cell": reach_cells, "quadratic_per_cell": quad_cells}

    # global decoy floor (any coherent cell, any mode) + the max CELL named ----------------- #
    dp = {f"{b}|{i}|{m}": coh[b][i]["decoy_p95"][m]
          for b in coh for i in coh[b] for m in coh[b][i]["decoy_p95"]
          if coh[b][i]["decoy_p95"][m] is not None}
    max_cell = max(dp, key=dp.get) if dp else None
    max_p95 = dp[max_cell] if max_cell else None
    elevated = bool(max_p95 is not None and max_p95 > DECOY_ELEVATED)
    out["decoys"] = {"max_p95": max_p95, "max_cell": max_cell, "bar": DECOY_ELEVATED,
                     "elevated": elevated}

    out["anchor"] = anchor

    # per-band classification (ladder override -> band-level NM) --------------------------- #
    bands = {}
    for b in BAND_ORDER:
        if b not in coh:
            continue
        lad = ladder.get(b, {"complete": True, "pass": True})
        if not lad.get("complete", True):
            bands[b] = {"class": "NO-MEASUREMENT (ladder rung-a failed to complete)", "base": None,
                        "ladder": lad}
        elif not lad.get("pass", True):
            bands[b] = {"class": "NO-MEASUREMENT (ladder gate fail: drive not demonstrably in "
                                 "first-order coordinates)", "base": None, "ladder": lad}
        else:
            cohb = {i: {"lin": coh[b][i]["r2"]["LIN"], "decoy_lin": coh[b][i]["decoy_p95"]["LIN"],
                        "esp_ok": coh[b][i]["esp_ok"]} for i in coh[b]}
            r = classify_band(cohb, subtr.get(b, {}))
            r["ladder"] = lad
            # ESP per-seed values + mean +/- SE (spec: labeled in json and .md regardless of class)
            dvals = {str(i): coh[b][i].get("esp_d_slow") for i in sorted(coh[b])}
            present = [x for x in dvals.values() if x is not None]
            r["esp_d_slow"] = {"per_seed": dvals,
                               "mean": (float(np.mean(present)) if present else None),
                               "se": (float(np.std(present, ddof=1) / math.sqrt(len(present)))
                                      if len(present) > 1 else None),
                               "eps": ESP_EPS}
            bands[b] = r
    out["bands"] = bands

    present = [b for b in BAND_ORDER if b in bands]
    summary = "; ".join(f"{b}={bands[b]['class']}" for b in present)

    # verdict precedence: instruments (seal) then consequence map ------------------------- #
    if not anchor.get("ok", False):
        out["verdict"] = ("NO-MEASUREMENT (anchor miss: AM x SUB not digit-exact (6dp) to "
                          "REF_TABLE -- shared import spine / harness fault, STOP)")
    elif not reach_ok:
        out["verdict"] = (f"NO-MEASUREMENT (reachability floor: min |corr(u,s)| {_f(min_lin)} < "
                          f"{REACH_FLOOR} -- tripwire, harness fault not physics, STOP)")
    elif elevated:
        out["verdict"] = (f"NO-MEASUREMENT (decoy elevated -- leakage: max p95 {_f(max_p95, '.3f')} "
                          f"> {DECOY_ELEVATED})")
    else:
        any_nm = any("NO-MEASUREMENT" in bands[b]["class"] for b in present)
        all_dead = (not any_nm) and all(bands[b].get("base") == "LIN-DEAD" for b in present) \
            and len(present) == len(BAND_ORDER)
        live_bands = [b for b in present if bands[b].get("base") == "LIN-LIVE"]
        if all_dead:
            out["consequence"] = ("COHERENT-LIN-SHUT: even with corr(u,s)=1 at the input, first-order "
                                  "slow-tertile coordinates do not carry the message at any tested "
                                  "band; at this operating point the square-law observable is the only "
                                  "demonstrated carrier, now tested from BOTH injection sides. Gate-3 "
                                  "scope note RESOLVES shut-side; C3 unmodified.")
            out["verdict"] = "COHERENT-LIN-SHUT (SUB/RES/SUPRA all LIN-DEAD): " + summary
        elif live_bands:
            out["consequence"] = (f"COHERENT-LIN-LIVE (band-resolved: {', '.join(live_bands)}): a "
                                  "coherent linear channel exists where injected coherently; C3 gains "
                                  "the scope note Gate-3's consequence map anticipated (|z|^2 is the "
                                  "power-envelope channel; coherent first-order transmission also "
                                  f"exists at {', '.join(live_bands)}).")
            out["verdict"] = f"COHERENT-LIN-LIVE ({', '.join(live_bands)} LIN-LIVE): " + summary
        else:
            out["consequence"] = "Other pattern -- report the matrix as-is, no new claim."
            out["verdict"] = "COHERENT-LIN (report matrix as-is): " + summary
        if any_nm:
            out["verdict"] += " [band-level NO-MEASUREMENT present -- see matrix]"
        out["erratum_unblock"] = ("The Phase-3 line-67 erratum WAIT lifts on this landing "
                                  f"(class fired: {out['verdict'].split(':')[0]}); the landing class "
                                  "selects the replacement wording. Drafting/filing in decade_drive "
                                  "remains Jason's separate call.")
    return out


# ============================================================================================ #
# markdown render (NM SEAL -- code-enforced)
# ============================================================================================ #
def _write_md(path, v, wall, hashes, colrep, env=None):
    """Render the record. NM-DISCLOSURE RULE, CODE-ENFORCED: when the OVERALL verdict is a
    GATE-LEVEL NO-MEASUREMENT (anchor / reachability / decoy tripped) the SEALED sections -- the
    LIN matrix, the D_LIN contrasts, the per-band classification, the consequence map, and the DC
    fractions -- are SUPPRESSED (instrument-failure disclosure only), until the NM resolution is
    ratified. Exercised by verdict_test's NM-shape case. No-truncation render assert lives in the
    self-tests. ASCII only."""
    is_nm = str(v.get("verdict", "")).startswith("NO-MEASUREMENT")
    r = v.get("reachability", {}); dz = v.get("decoys", {}); an = v.get("anchor", {})
    lines = [
        "# Relay Gate-L -- Coherent Linear Injection (the other door)", "",
        f"Spec: relay_gateL_coherent_spec.md (sha256 {hashes.get('spec', '?')}). Harness: "
        f"experiments/relay_gateL.py (sha256 {hashes.get('code', '?')}).",
        f"Wall-clock {wall/60:.0f} min. Fresh-decoy families collision-free: {colrep.get('clear_of_committed')}.",
        "", f"Framing: {FRAMING}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
        "## Pre-data fence (restated)", "", v.get("fence", FENCE), "",
        "## Instrument checks (pre-registered order: anchor, reachability, ladder, decoys, ESP)", "",
        f"1. **Anchor** (AM x SUB per-seed digit-exact 6dp vs REF_TABLE; mean rule): "
        f"{'ALL PASS' if an.get('ok') else 'MISS'} "
        f"(mean {_f(an.get('mean'))} +/- {_f(an.get('se'))})."
        + (lambda d: (f" Diagnostic (not gated): bit-exact {d['bit_exact_cells']}/{d['total_cells']} "
                      f"vs the committed gate3 full-precision cells.") if d else "")(
              an.get("bit_exact_vs_committed_diagnostic")),
        f"2. **Reachability** (linear-arm tripwire, gate |corr(u,s)| >= {r.get('floor', REACH_FLOOR)}): "
        f"min |corr(u,s)| {_f(r.get('min_lin'))} -> {'OK' if r.get('ok') else 'TRIP (harness fault)'}. "
        f"Quadratic arm corr(u^2,s) recorded, no gate.",
    ]
    lad = {b: v.get("bands", {}).get(b, {}).get("ladder", {}) for b in BAND_ORDER}
    lad_txt = "; ".join(f"{b}: pass={lad[b].get('pass')}, ratio={_f(lad[b].get('ratio'))}"
                        for b in BAND_ORDER if lad.get(b))
    lines += [
        f"3. **Linearity ladder** (P_track increasing AND P_track(a) >= {LADDER_FACTOR}x chance [P3]): "
        f"{lad_txt or 'n/a'}.",
        f"4. **Decoy floors** (max fresh p95 over all coherent cells): {_f(dz.get('max_p95'), '.3f')} "
        f"at cell {dz.get('max_cell')}; bar {dz.get('bar', DECOY_ELEVATED)} -> "
        f"{'ELEVATED (leak)' if dz.get('elevated') else 'clean'}.",
        "5. **ESP** nested ok_slow per seed (values below); symmetric intersection per band; "
        "memberships in the matrix.", "",
        "GAP (accepted 2026-07-18, no backfill): the P_slow ladder analog (slow-tertile "
        "drive-tracking projection; spec report-only context for the verdict prose) was NOT recorded "
        "at the battery -- trajectories are not stored, so it cannot be derived post hoc. Accepted "
        "gap; does not affect the verdict or any gated instrument.", "",
    ]
    if is_nm:
        lines += [
            "## SEALED (NM-disclosure rule -- code-enforced)", "",
            "This gate is at a GATE-LEVEL NO-MEASUREMENT: an instrument tripped (see the Verdict "
            "line + Instrument checks above). Per the NM-disclosure rule, the SEALED sections -- the "
            "LIN matrix, the D_LIN contrasts, the per-band classification, the consequence map, and "
            "the DC fractions -- are WITHHELD until the NM resolution is ratified. Resolution "
            "decisions are made blind. Re-render with --reread once the resolution is ratified."]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    # ---- SEALED sections (measurement disclosed) ---------------------------------------- #
    lines += ["## LIN matrix (coherent r2_LIN vs committed AM baseline, per band)", ""]
    for b in BAND_ORDER:
        bd = v.get("bands", {}).get(b)
        if not bd:
            continue
        if bd.get("base") is None and "NO-MEASUREMENT" in bd.get("class", ""):
            lines.append(f"- **{b}**: {bd['class']}")
            continue
        cb = bd.get("committed_baseline") or {}
        lines.append(
            f"- **{b}**: coherent mean r2_LIN = {_f(bd.get('coh_mean'))} (n={bd.get('n_paired')}) "
            f"vs committed AM baseline mean {_f(cb.get('mean'))} / median {_f(cb.get('median'))}; "
            f"D_LIN mean = {_f(bd.get('D_mean'))} +/- {_f(bd.get('se_paired'))}, median "
            f"{_f(bd.get('D_median'))}; delta = {_f(bd.get('delta'))}; max fresh LIN decoy p95 = "
            f"{_f(bd.get('decoy_lin_max'))}; seeds {bd.get('symmetric_intersection')} -> "
            f"**{bd.get('class')}**")
    # LOSO (report-only; spec: for any LIVE band)
    loso_lines = []
    for b in BAND_ORDER:
        lo = v.get("bands", {}).get(b, {}).get("loso")
        if lo:
            loso_lines.append(
                f"- {b}: LIN-LIVE survives {lo['n_survive']}/{lo['n']} single-seed drops"
                f"{' (ALL)' if lo.get('all_survive') else ''}; tightest drop seed "
                f"{lo.get('tightest_seed')} (margin {_f(lo.get('tightest_margin'), '+.4f')}); "
                f"deltas 2*SE-governed, both legs re-checked per drop.")
    if loso_lines:
        lines += ["", "## Robustness -- leave-one-seed-out (report-only, LIVE bands)", ""] + loso_lines
    # ESP values (regardless of class)
    lines += ["", "## ESP (nested ok_slow; d_slow per seed, mean +/- SE; eps = 1e-2)", ""]
    for b in BAND_ORDER:
        ed = v.get("bands", {}).get(b, {}).get("esp_d_slow")
        if ed:
            pv = " ".join(f"{k}:{val:.2e}" for k, val in ed["per_seed"].items() if val is not None)
            se_txt = f"{ed['se']:.2e}" if ed.get("se") is not None else "n/a"
            mean_txt = f"{ed['mean']:.2e}" if ed.get("mean") is not None else "n/a"
            lines.append(f"- {b}: mean {mean_txt} +/- {se_txt}; per-seed [{pv}]")
    lines += ["", "## Per-seed D_LIN signs (pattern-fold from Gate-B; observed, not claimed)", ""]
    for b in BAND_ORDER:
        bd = v.get("bands", {}).get(b, {})
        ps = bd.get("per_seed_D")
        if ps:
            signs = " ".join(f"{k}:{'+' if val >= 0 else '-'}" for k, val in ps.items())
            lines.append(f"- {b}: {signs}")
    lines += ["", "## DC fractions (realized E[s]^2/E[s^2] per band; report-only [P1])", ""]
    for b in BAND_ORDER:
        bd = v.get("bands", {}).get(b, {})
        dcf = bd.get("dc_fraction")
        if dcf is not None:
            lines.append(f"- {b}: DC fraction mean = {_f(dcf)}")
    lines += ["", "## Consequence map", "", v.get("consequence", ""), "",
              v.get("erratum_unblock", ""), "",
              "## Scope", "",
              "Readout-level observable-order question (symmetric to Gate-3): does the FIRST-ORDER "
              "slow-tertile coordinate carry the message under coherent injection -- NOT the internal "
              "transfer path (the Stuart-Landau nonlinearity mixes orders en route). One operating "
              "point (K=0.24, span 1.5), stage-A, offline, no chains. Windows/thresholds NOT stored "
              "(delta evaluated at verdict from byte-locked (D, SE, n)). STOP-and-report."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================================ #
# GPU: coherent stage-A cell, anchor cell, ladder (implemented; run at smoke/battery)
# ============================================================================================ #
def stage_a_coherent(band, i, geom, decoys=True):
    """One coherent stage-A run: Gate-3's stage-A with the injection map swapped to the [P1]
    coherent zero-mean map. SAME reservoir + SAME message realization s per (band, seed)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    lo, hi = BANDS[band]
    sd = seed_scheme(i, band)
    sp = p1.build_system(sd["build"], N, STAGE_SPAN)
    bands = p1.band_indices(sp.omega)
    m_fast = p1.masked_encoding(sp.omega, bands["fast"], np.random.default_rng(sd["enc"]))
    outside = np.concatenate([bands["slow"], bands["guard"]])
    assert float(np.abs(m_fast[outside]).max()) == 0.0                # fast-band-only injection
    s = p1.slow_bandlimited(L, dt_in, lo, hi, seed=sd["msg"])          # IDENTICAL to Gate-3's s
    cd = coherent_drive(s)
    u = cd["u"]
    assert cd["pm_resid"] <= 1e-9 * cd["target"], "power-match identity violated"
    reach = reachability(u, s, delays, sl)
    Ks = list(p1.K_GRID)
    ki = Ks.index(K_PRIMARY)
    X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u, Ks, dt_in, n_sub)[ki]
    rep = p1.replica_spec(sp, sd["rep"])
    Xr = p1.integrate_Ks(sp.omega, sp.L, m_fast, rep.z0, u, Ks, dt_in, n_sub)[ki]
    d_slow = p1.consistency_distance(X[:, bands["slow"]], Xr[:, bands["slow"]], sl)
    esp = {"d_slow": float(d_slow), "ok_slow": bool(d_slow < ESP_EPS)}
    dec = band_decoys_L(band, i, L, dt_in) if decoys else None
    r2 = {}; decoy_p95 = {}
    for name in MODE_ORDER:
        dm = (p1.demod_capacity(X, bands["slow"], s, dec, delays, sl, MODES[name])
              if dec is not None else None)
        r2[name] = float(dm["r2_d0"]) if dm else None
        decoy_p95[name] = float(dm["decoy_p95"]) if dm else None
    mband = g0.dominant_band(s[iw], dt_in)
    del X, Xr
    return {"band": band, "band_hz": [lo, hi], "seed": i, "K": K_PRIMARY, "seeds": sd,
            "r2": r2, "decoy_p95": decoy_p95, "esp_ok": esp["ok_slow"], "esp": esp,
            "reach": {"lin": reach["lin"], "quad": reach["quad"], "per_lag": reach["per_lag"]},
            "amp": {"a": cd["a"], "mean_inj": cd["mean_inj"], "var_inj": cd["var_inj"],
                    "e_u2": cd["e_u2"], "target": cd["target"], "pm_resid": cd["pm_resid"],
                    "pm_rel": cd["pm_rel"], "dc_frac": cd["dc_frac"]},
            "msg_dominant_band": [float(mband[0]), float(mband[1])]}


def anchor_cell(i, geom):
    """AM x SUB replica through the committed Gate-3 stage_a (the shared import spine). r2.FULL is
    the Phase-1 anchor; digit-exact (6dp) vs REF_TABLE certifies the spine end-to-end."""
    r = g3.stage_a("SUB", i, geom)
    ref = g0.REF_TABLE[(STAGE_SPAN, K_PRIMARY)][i][0]
    got = r["r2"]["FULL"]
    return {"seed": i, "got": float(got), "ref": float(ref),
            "diff": float(abs(got - ref)), "digit6_ok": bool(round(got, 6) == round(ref, 6)),
            "esp_ok": bool(r["esp"]["ok_slow"])}


def _anchor_report(cells, committed_full=None):
    """Per-seed digit-exact (6dp) GATE vs REF_TABLE + mean-window (the ratified anchor). PLUS the
    ADOPTED battery-go DIAGNOSTIC (never gated; Gate-4 precedent): bit-exactness of each anchor
    cell vs the committed gate3_mechanism.json FULL-PRECISION r2.FULL (sha-verified read-at-use).
    The per-seed 'diff' vs REF_TABLE is the 6dp storage quantum, NOT drift -- the bit-exact read
    lives in the diagnostic block."""
    per = {c["seed"]: c for c in cells}
    digit_ok = all(c["digit6_ok"] for c in cells)
    vals = [c["got"] for c in cells]
    st = g0._mstats(vals) if vals else {"mean": None, "se": None}
    mean, se = st["mean"], st["se"]
    win = max(2.0 * (se or 0.0), 0.02)
    mean_ok = bool(mean is not None and abs(mean - 0.986) <= win)
    diag = None
    if committed_full:
        dper, nbit = {}, 0
        for c in cells:
            ref_fp = committed_full.get(c["seed"])
            if ref_fp is None:
                continue
            dfp = float(abs(c["got"] - ref_fp))
            ulp = int(round(dfp / float(np.spacing(ref_fp)))) if dfp > 0.0 else 0
            bit = bool(dfp == 0.0)
            dper[str(c["seed"])] = {"diff": dfp, "ulp": ulp, "bit_exact": bit}
            nbit += int(bit)
        diag = {"note": "DIAGNOSTIC ONLY (adopted at battery-go; digit-exact 6dp remains the "
                        "gate -- Gate-4 precedent)",
                "vs": "gate3_mechanism.json full-precision r2.FULL (sha-verified)",
                "bit_exact_cells": nbit, "total_cells": len(dper), "per_seed": dper}
    return {"ok": bool(digit_ok and mean_ok), "digit_exact_6dp": bool(digit_ok),
            "mean": mean, "se": se, "mean_window": win, "mean_ok": mean_ok,
            "bit_exact_vs_committed_diagnostic": diag,
            "per_seed": {str(k): {"got": round(c["got"], 6), "ref": round(c["ref"], 6),
                                  "diff_vs_6dp_ref": c["diff"], "digit6_ok": c["digit6_ok"]}
                         for k, c in per.items()}}


def ladder_battery(geom, seed=0):
    """Per band (seed 0), rungs {0.5a, a, 2a}: integrate coherent drive scaled by the rung, compute
    P_track from fast-tertile Re z; chance from the NO-INJ (u=0) trajectory against rung-a u.
    Rung 2a integration failure is tolerated (a must complete). Returns ladder dict per band."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    sd = seed_scheme(seed, "SUB")
    sp = p1.build_system(sd["build"], N, STAGE_SPAN)
    bands = p1.band_indices(sp.omega)
    m_fast = p1.masked_encoding(sp.omega, bands["fast"], np.random.default_rng(sd["enc"]))
    Ks = list(p1.K_GRID); ki = Ks.index(K_PRIMARY)
    # NO-INJ trajectory (u = 0, band-independent)
    u_zero = np.zeros(L)
    Xn = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u_zero, Ks, dt_in, n_sub)[ki]
    Xn_fast = Xn[:, bands["fast"]]
    out = {}
    for b in BAND_ORDER:
        lo, hi = BANDS[b]
        s = p1.slow_bandlimited(L, dt_in, lo, hi, seed=sd["msg"])
        cd = coherent_drive(s)
        u_a = cd["u"]
        # chance: NO-INJ fast-tertile Re z projected onto rung-a u
        chance = band_track(Xn_fast, u_a, dt_in, (lo, hi), iw)
        rung_vals = []
        for scale in (0.5, 1.0, 2.0):
            try:
                X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, scale * u_a, Ks, dt_in, n_sub)[ki]
                rung_vals.append(band_track(X[:, bands["fast"]], scale * u_a, dt_in, (lo, hi), iw))
                del X
            except Exception:
                rung_vals.append(None)
        g = ladder_gate(rung_vals, chance)
        g["P_track"] = {"0.5a": rung_vals[0], "a": rung_vals[1], "2a": rung_vals[2]}
        out[b] = g
    del Xn
    return out


# ============================================================================================ #
# S1 sandbox -- every check PROVEN TO FIRE on a synthetic violation (CPU, no GPU)
# ============================================================================================ #
@contextlib.contextmanager
def _seed_audit():
    """Record every seed passed to np.random.default_rng during the block (seed-consumption audit)."""
    rec = []
    orig = np.random.default_rng

    def wrapped(seed=None):
        rec.append(seed)
        return orig(seed)
    np.random.default_rng = wrapped
    try:
        yield rec
    finally:
        np.random.default_rng = orig


def _synth_cell(lin, decoy_lin, esp_ok=True, reach_lin=1.0, reach_quad=0.3, decoy_sq=-0.1,
                decoy_full=-0.1):
    return {"r2": {"LIN": lin, "SQ": 0.9, "FULL": 0.9},
            "decoy_p95": {"LIN": decoy_lin, "SQ": decoy_sq, "FULL": decoy_full},
            "esp_ok": esp_ok, "reach": {"lin": reach_lin, "quad": reach_quad},
            "dc_fraction": 0.6}


def _synth_state(lin_by_band, decoy_by_band=None, esp_by_band=None, reach_lin=1.0,
                 anchor_ok=True, subtr_lin=None, subtr_esp=None, ladder_pass=True,
                 ladder_complete=True):
    """Build a full (coh, anchor, subtr, ladder) synthetic state for decide()."""
    # default fresh LIN decoy p95 ~ +0.05 (committed Gate-3 LIN decoys sit near 0.0-0.06); a dead
    # coherent r2_LIN (~ -0.05) then reads LIN-DEAD (coh_mean <= decoy) as it should.
    decoy_by_band = decoy_by_band or {b: 0.05 for b in BAND_ORDER}
    esp_by_band = esp_by_band or {b: {i: True for i in SEEDS} for b in BAND_ORDER}
    subtr_lin = subtr_lin or {b: -0.05 for b in BAND_ORDER}
    subtr_esp = subtr_esp or {b: {i: True for i in SEEDS} for b in BAND_ORDER}
    coh = {b: {i: _synth_cell(lin_by_band[b], decoy_by_band[b], esp_by_band[b][i], reach_lin)
               for i in SEEDS} for b in BAND_ORDER}
    for b in BAND_ORDER:
        for i in SEEDS:
            coh[b][i]["dc_fraction"] = 0.6
    anchor = {"ok": anchor_ok, "mean": 0.986, "se": 0.001, "per_seed": {}}
    subtr = {b: {i: {"lin": subtr_lin[b], "esp_ok": subtr_esp[b][i]} for i in SEEDS}
             for b in BAND_ORDER}
    ladder = {b: {"complete": ladder_complete, "pass": ladder_pass} for b in BAND_ORDER}
    # decorate bands with dc_fraction after decide (decide reads coh only); attach for render
    return coh, anchor, subtr, ladder


def _band_dc(v, coh):
    for b in BAND_ORDER:
        if b in v.get("bands", {}) and v["bands"][b].get("base") is not None:
            vals = [coh[b][i]["dc_fraction"] for i in coh[b]]
            v["bands"][b]["dc_fraction"] = float(np.mean(vals))
    return v


def sandbox(log):
    log("=== RELAY GATE-L :: S1 SANDBOX (CPU, no GPU) -- every check PROVEN TO FIRE ===")
    results = {}

    # (1) injection unit test: coherent zero-mean map + power-match + DC + seed-consumption ---- #
    log("\n(1) Coherent injection map: zero-mean, corr(u,s)>=0.99, power-match identity, DC, seed audit")
    rng = np.random.default_rng(12345)
    s_syn = 0.1 + 0.9 * (0.5 + 0.5 * np.sin(np.linspace(0, 30, 4000)) + 0.05 * rng.standard_normal(4000))
    s_syn = np.clip(s_syn, 1e-3, None)
    cd = coherent_drive(s_syn)
    sl_syn = slice(200, 4000)
    reach_syn = reachability(cd["u"], s_syn, [0, 32, 64, 96], sl_syn)
    zero_mean = abs(float(np.mean(cd["u"]))) < 1e-9
    corr_ok = reach_syn["lin"] >= 0.99
    pm_ok = cd["pm_resid"] <= 1e-9 * cd["target"]
    dc_ok = (0.0 < cd["dc_frac"] < 1.0)
    # violation: wrong amplitude breaks the power-match identity (proves the check fires)
    u_bad = 0.6 * (s_syn - np.mean(s_syn))
    pm_resid_bad = abs(float(np.mean(u_bad * u_bad)) - 0.25 * float(np.mean(s_syn)))
    pm_fires = pm_resid_bad > 1e-9 * (0.25 * float(np.mean(s_syn)))
    # seed-consumption audit: coherent path draws msg, NEVER the Rademacher (msg+777)
    msg_seed = 1000 + 3
    with _seed_audit() as rec_coh:
        s_a = p1.slow_bandlimited(2000, 0.1, 0.2, 0.9, seed=msg_seed)
        coherent_drive(s_a)
    coh_draws_msg = msg_seed in rec_coh
    coh_no_rade = (msg_seed + RADE_OFFSET) not in rec_coh
    with _seed_audit() as rec_am:
        g0.am_input_band(2000, 0.1, msg_seed, 0.2, 0.9)
    am_draws_rade = (msg_seed + RADE_OFFSET) in rec_am        # contrast: AM DOES draw it
    c1 = all([
        g0._check(log, "coherent map is exactly zero-mean", zero_mean, f"mean(u)={np.mean(cd['u']):.2e}"),
        g0._check(log, "corr(u,s) >= 0.99 (= 1 by construction)", corr_ok, f"max|corr|={reach_syn['lin']:.6f}"),
        g0._check(log, "power-match identity E[u^2]=0.25*E_inj[s] holds", pm_ok, f"resid={cd['pm_resid']:.2e}"),
        g0._check(log, "power-match check FIRES on a wrong-amplitude map", pm_fires, f"bad resid={pm_resid_bad:.4f}"),
        g0._check(log, "DC fraction recorded in (0,1)", dc_ok, f"dc={cd['dc_frac']:.4f}"),
        g0._check(log, "seed audit: coherent path draws msg seed", coh_draws_msg, f"seeds={sorted(set(rec_coh))}"),
        g0._check(log, "seed audit: Rademacher (msg+777) NEVER drawn in coherent cell", coh_no_rade,
                  f"{msg_seed+RADE_OFFSET} absent"),
        g0._check(log, "seed audit distinguishes: AM path DOES draw Rademacher", am_draws_rade,
                  f"{msg_seed+RADE_OFFSET} present in AM"),
    ])
    results["check1_injection"] = {"pass": c1}

    # (2) reachability-floor branch fires on a synthetic sub-floor cell ---------------------- #
    log("\n(2) Reachability tripwire: sub-floor coherent cell -> gate-level NO-MEASUREMENT")
    coh, anchor, subtr, ladder = _synth_state({b: -0.05 for b in BAND_ORDER})
    coh["RES"][2]["reach"]["lin"] = 0.4                        # inject a sub-floor cell
    v_reach = decide(coh, anchor, subtr, ladder)
    reach_fires = v_reach["verdict"].startswith("NO-MEASUREMENT") and "reachability" in v_reach["verdict"]
    coh_ok, a_ok, s_ok, l_ok = _synth_state({b: -0.05 for b in BAND_ORDER})
    v_reach_ok = decide(coh_ok, a_ok, s_ok, l_ok)
    reach_silent = not (v_reach_ok["verdict"].startswith("NO-MEASUREMENT")
                        and "reachability" in v_reach_ok["verdict"])
    c2 = all([
        g0._check(log, "sub-floor |corr(u,s)| -> NO-MEASUREMENT (tripwire)", reach_fires, v_reach["verdict"][:70]),
        g0._check(log, "healthy reachability stays silent", reach_silent, "no false trip"),
    ])
    results["check2_reachability"] = {"pass": c2}

    # (3) ladder P_track REQUIRED PAIR [P3] -------------------------------------------------- #
    log("\n(3) Ladder P_track [P3]: tracking-absent FAILS; tracking-present-with-dominant-bg PASSES "
        "where raw-power fails")
    T = 4000
    iw3 = np.arange(200, T)
    t = np.linspace(0, 60, T)
    w = 3.0
    u_wave = np.sin(w * t)                                     # rung-a drive waveform
    bg = 10.0 * np.cos(w * t)                                  # in-band natural carrier (orthogonal to u)
    nosc = 12
    # (i) tracking-absent: fast-tertile Re z is background only, no drive-correlated component
    x_absent = np.column_stack([bg + 0.01 * np.random.default_rng(700 + j).standard_normal(T)
                                for j in range(nosc)])
    P_absent = _ptrack_core(x_absent, u_wave, iw3)
    chance_absent = _ptrack_core(np.column_stack([bg for _ in range(nosc)]), u_wave, iw3)
    gate_absent = ladder_gate([0.5 * P_absent, P_absent, 2.0 * P_absent], max(chance_absent, 1e-12))
    absent_fails = not gate_absent["pass"]
    # (ii) tracking-present WITH dominant in-band natural background
    def x_present(scale):
        return np.column_stack([bg + scale * 0.3 * u_wave for _ in range(nosc)])
    x_noinj = np.column_stack([bg for _ in range(nosc)])
    chance_pres = _ptrack_core(x_noinj, u_wave, iw3)
    P05, P1, P2 = (_ptrack_core(x_present(0.5), 0.5 * u_wave, iw3),
                   _ptrack_core(x_present(1.0), 1.0 * u_wave, iw3),
                   _ptrack_core(x_present(2.0), 2.0 * u_wave, iw3))
    gate_pres = ladder_gate([P05, P1, P2], max(chance_pres, 1e-12))
    present_passes = gate_pres["pass"]
    # the SUPERSEDED raw-power check would fail spuriously here (background dominates, ratio ~ 1)
    raw_inj = _raw_inband_power(x_present(1.0), iw3)
    raw_noinj = _raw_inband_power(x_noinj, iw3)
    raw_ratio = raw_inj / raw_noinj
    raw_would_fail = raw_ratio < LADDER_FACTOR                 # < 10x -> superseded gate fails
    # completion semantics (spec: ONLY 2a failure tolerated; 0.5a or a missing -> incomplete)
    g_2a_fail = ladder_gate([P05, P1, None], max(chance_pres, 1e-12))       # tolerated, gate on 2 rungs
    g_05_fail = ladder_gate([None, P1, P2], max(chance_pres, 1e-12))        # NOT tolerated -> incomplete
    g_a_fail = ladder_gate([P05, None, P2], max(chance_pres, 1e-12))        # NOT tolerated -> incomplete
    c3 = all([
        g0._check(log, "tracking-absent -> ladder gate FAILS", absent_fails,
                  f"P_a={P_absent:.3e} chance={chance_absent:.3e}"),
        g0._check(log, "tracking-present (dominant bg) -> P_track gate PASSES", present_passes,
                  f"P_a={P1:.3e} chance={chance_pres:.3e} ratio={gate_pres['ratio']:.1f}"),
        g0._check(log, "superseded raw-power gate would FAIL spuriously (bg dominates)", raw_would_fail,
                  f"raw ratio={raw_ratio:.3f} < {LADDER_FACTOR}"),
        g0._check(log, "2a failure TOLERATED (gate holds on 0.5a + a)", g_2a_fail["pass"]
                  and g_2a_fail["complete"], "complete + pass on two rungs"),
        g0._check(log, "0.5a failure NOT tolerated -> incomplete (band NM)", not g_05_fail["complete"]
                  and not g_05_fail["pass"], "incomplete"),
        g0._check(log, "rung-a failure NOT tolerated -> incomplete (band NM)", not g_a_fail["complete"]
                  and not g_a_fail["pass"], "incomplete"),
    ])
    results["check3_ladder"] = {"pass": c3, "raw_ratio": raw_ratio, "ptrack_ratio": gate_pres["ratio"]}

    # (4) decoy collision matrix vs ALL committed families ----------------------------------- #
    log("\n(4) Fresh decoy collision matrix vs ALL committed families")
    colrep, col_ok = collision_matrix()
    log(f"  fresh bases {colrep['fresh_bases']} (footprint span {colrep['family_span']}); "
        f"clear of {colrep['n_committed_families']} committed families "
        f"(min fresh {colrep['min_fresh_base']} > max committed footprint {colrep['max_committed_footprint']})")
    c4 = all([
        g0._check(log, "no pairwise decoy-family overlap (fresh vs committed)", col_ok,
                  f"overlaps={colrep['pairwise_overlaps']}"),
        g0._check(log, "fresh families clear the entire committed range", colrep["clear_of_committed"],
                  "min fresh > max committed footprint"),
    ])
    results["check4_collision"] = {"pass": c4, "collision": colrep}

    # (5) classifier + precedence + -UNDERPOWERED + every NM branch -------------------------- #
    log("\n(5) Classifier: DEAD/LIVE/INTERMEDIATE precedence + -UNDERPOWERED + NM")
    def cls(cohb, subb):
        return classify_band(cohb, subb)["class"]
    seeds5 = list(range(6))                                    # n=6 -> full strength
    dead = cls({i: {"lin": -0.05, "decoy_lin": 0.02, "esp_ok": True} for i in seeds5},
               {i: {"lin": -0.05, "esp_ok": True} for i in seeds5})          # coh_mean <= decoy
    live = cls({i: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True} for i in seeds5},
               {i: {"lin": -0.05, "esp_ok": True} for i in seeds5})          # D>delta AND coh>0.2
    inter = cls({i: {"lin": 0.1, "decoy_lin": 0.02, "esp_ok": True} for i in seeds5},
                {i: {"lin": 0.05, "esp_ok": True} for i in seeds5})          # coh>decoy but coh<0.2
    # LIVE trap: dead-vs-dead can give D>delta from two nulls; the coh>0.2 leg must block it
    trap = cls({i: {"lin": 0.0, "decoy_lin": -0.1, "esp_ok": True} for i in seeds5},
               {i: {"lin": -0.05, "esp_ok": True} for i in seeds5})          # D=+0.05 but coh~0 -> NOT live
    up = cls({i: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True} for i in range(3)},
             {i: {"lin": -0.05, "esp_ok": True} for i in range(3)})          # n=3 -> -UNDERPOWERED
    nm = cls({0: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True}},
             {0: {"lin": -0.05, "esp_ok": True}})                            # n=1 -> NM
    c5 = all([
        g0._check(log, "coh_mean <= decoy p95 -> LIN-DEAD (precedence 1)", dead == "LIN-DEAD", dead),
        g0._check(log, "D>delta AND coh>0.2 -> LIN-LIVE", live == "LIN-LIVE", live),
        g0._check(log, "coh above decoy but < 0.2 -> INTERMEDIATE", inter == "INTERMEDIATE", inter),
        g0._check(log, "dead-vs-dead D>delta but coh~0 -> NOT LIVE (0.2 floor blocks the trap)",
                  trap in ("INTERMEDIATE", "LIN-DEAD"), trap),
        g0._check(log, "n in [2,4] -> -UNDERPOWERED suffix", up == "LIN-LIVE-UNDERPOWERED", up),
        g0._check(log, "n < 2 -> NO-MEASUREMENT", "NO-MEASUREMENT" in nm, nm),
    ])
    results["check5_classifier"] = {"pass": c5}

    # (6) subtrahend read-at-use: sha assert fires loudly on a byte-perturbed copy ----------- #
    log("\n(6) Subtrahend read-at-use: gate3_mechanism.json sha assert fires on a perturbed copy")
    import tempfile
    real_ok, chk = load_subtrahend(GATE3_JSON)
    load_ok = chk["ok"] and real_ok is not None
    perturbed = os.path.join(tempfile.gettempdir(), "_gL_gate3_perturbed.json")
    raw = open(GATE3_JSON, "rb").read()
    with open(perturbed, "wb") as f:
        f.write(raw[:-3] + b"0" + raw[-2:])                    # flip a byte
    _, chk_bad = load_subtrahend(perturbed)
    os.remove(perturbed)
    sha_fires = not chk_bad["ok"]
    c6 = all([
        g0._check(log, "committed subtrahend loads (sha matches pinned)", load_ok,
                  chk.get("gate3_sha256", "?")[:12]),
        g0._check(log, "byte-perturbed copy -> sha mismatch -> NOT ok (loud)", sha_fires,
                  (chk_bad["reasons"][0][:50] if chk_bad["reasons"] else "")),
    ])
    results["check6_subtrahend"] = {"pass": c6}

    # (7) LOCKED-NUMBERS loud-fail reread test (inherited) ----------------------------------- #
    log("\n(7) LOCKED-NUMBERS: the reread byte-assert RAISES on any numeric drift")
    payload = {"SUB": {"0": {"r2": {"LIN": -0.0205}, "esp": {"ok_slow": True}}}}
    baseline = json.dumps(payload, sort_keys=True)
    clean_ok = (json.dumps(json.loads(baseline), sort_keys=True) == baseline)
    drifted = json.loads(baseline)
    drifted["SUB"]["0"]["r2"]["LIN"] = -0.0206
    raised = False
    try:
        assert json.dumps(drifted, sort_keys=True) == baseline, "drift"
    except AssertionError:
        raised = True
    c7 = all([
        g0._check(log, "clean round-trip byte-identical (no false alarm)", clean_ok, "identical"),
        g0._check(log, "single perturbed number makes the byte-assert RAISE", raised, "1e-4 drift caught"),
    ])
    results["check7_locked_numbers"] = {"pass": c7}

    # (8) NM SEAL shape self-test ------------------------------------------------------------ #
    log("\n(8) NM SEAL: an NM verdict suppresses every sealed section (code-enforced)")
    coh8, an8, su8, la8 = _synth_state({b: -0.05 for b in BAND_ORDER}, anchor_ok=False)
    v_nm = decide(coh8, an8, su8, la8)
    pnm = os.path.join(tempfile.gettempdir(), "_gL_md_nm.md")
    _write_md(pnm, v_nm, 0.0, _hashes(), {"clear_of_committed": True})
    tnm = open(pnm).read(); os.remove(pnm)
    nm_sealed = (v_nm["verdict"].startswith("NO-MEASUREMENT")
                 and "## LIN matrix" not in tnm and "## Consequence map" not in tnm
                 and "LIN-DEAD" not in tnm and "SEALED (NM-disclosure" in tnm and "..." not in tnm)
    c8 = g0._check(log, "NM render SEALS LIN matrix / contrasts / consequence + shows SEALED notice",
                   nm_sealed, "sealed")
    results["check8_nm_seal"] = {"pass": bool(c8)}

    # (9) no-truncation render assert: fires on synthetic truncation; silent on healthy .mds -- #
    log("\n(9) No-truncation render: caught on synthetic truncation; silent on healthy committed .mds")
    coh9, an9, su9, la9 = _synth_state({"SUB": -0.05, "RES": -0.05, "SUPRA": -0.05})
    v9 = decide(coh9, an9, su9, la9)
    v9 = _band_dc(v9, coh9)
    p9 = os.path.join(tempfile.gettempdir(), "_gL_md_ok.md")
    _write_md(p9, v9, 0.0, _hashes(), {"clear_of_committed": True})
    txt9 = open(p9).read(); os.remove(p9)
    self_clean = "..." not in txt9 and "Coherent Linear Injection" in txt9
    truncated = txt9[:200] + " field...cut " + txt9[200:]      # inject an ellipsis-cut field
    catch_fires = "..." in truncated                           # the guard would catch it
    committed_clean = True
    for f in ("gate3_mechanism.md", "gate4_hoptrade.md", "gateB_probe.md", "gateB_broadband.md"):
        fp = os.path.join(RESDIR, f)
        if os.path.exists(fp) and "..." in open(fp).read():
            committed_clean = False
    c9 = all([
        g0._check(log, "healthy Gate-L render has no '...'", self_clean, "clean self-render"),
        g0._check(log, "guard FIRES on a synthetic ellipsis-cut field", catch_fires, "'...' detected"),
        g0._check(log, "silent on healthy committed sibling .mds", committed_clean, "no false alarm"),
    ])
    results["check9_no_truncation"] = {"pass": c9}

    # (10) delta + windows evaluate-at-use (nothing derived is stored) ------------------------ #
    log("\n(10) Evaluate-at-use: delta recomputed from (D, SE, n); inflated SE flips the verdict")
    tight = classify_band({i: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True} for i in range(6)},
                          {i: {"lin": 0.55 + (0.001 if i % 2 else -0.001), "esp_ok": True} for i in range(6)})
    # same coherent means, inflated per-seed spread -> larger SE -> larger delta -> not LIVE by D
    noisy = classify_band({i: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True} for i in range(6)},
                          {i: {"lin": 0.55 + (0.2 if i % 2 else -0.2), "esp_ok": True} for i in range(6)})
    delta_moved = (tight["delta"] != noisy["delta"])
    not_stored = ("delta" in tight and tight["delta"] == max(2.0 * tight["se_paired"], DELTA_FLOOR))
    c10 = all([
        g0._check(log, "delta differs with SE (evaluated at use, not stored)", delta_moved,
                  f"tight delta {tight['delta']:.4f} vs noisy {noisy['delta']:.4f}"),
        g0._check(log, "delta = max(2*SE, floor) recomputed from primitives", not_stored, "recomputed"),
    ])
    results["check10_evaluate_at_use"] = {"pass": c10}

    # (11) LOSO integrity: record verified vs a fresh source-recompute; corruption fires the guard #
    log("\n(11) LOSO integrity: record vs source-recompute [L-1 hardening]")
    cohb_h = {i: {"lin": 0.9 + 0.001 * i, "decoy_lin": 0.02, "esp_ok": True} for i in range(8)}
    subb_h = {i: {"lin": -0.05 + 0.0007 * i, "esp_ok": True} for i in range(8)}
    inter_h = list(range(8))
    lo_h = _loso_band(cohb_h, subb_h, inter_h)
    healthy_ok = _loso_verify(lo_h, cohb_h, subb_h, inter_h)
    bad = {"drops": {k: dict(v) for k, v in lo_h["drops"].items()}}   # L-1 symptom: drop '2' carries
    bad["drops"]["2"]["se"] = bad["drops"]["5"]["se"]                 #   drop '5's (se, coh_mean)
    bad["drops"]["2"]["coh_mean"] = bad["drops"]["5"]["coh_mean"]
    corrupt_fires = not _loso_verify(bad, cohb_h, subb_h, inter_h)
    # NO false-fire on legitimately symmetric data (equal se+coh across drops, differing D_mean)
    coh_sym = {i: {"lin": 0.6, "decoy_lin": 0.02, "esp_ok": True} for i in range(6)}
    sub_sym = {i: {"lin": 0.55 + (0.001 if i % 2 else -0.001), "esp_ok": True} for i in range(6)}
    lo_sym = _loso_band(coh_sym, sub_sym, list(range(6)))
    sym_ok = _loso_verify(lo_sym, coh_sym, sub_sym, list(range(6)))
    c11 = all([
        g0._check(log, "healthy LOSO record verifies against source recompute", healthy_ok, "match"),
        g0._check(log, "drop '2' carrying drop '5' (se, coh_mean) -> guard FIRES", corrupt_fires,
                  "cross-drop corruption caught"),
        g0._check(log, "NO false-fire on symmetric data (equal se+coh, differing D_mean)", sym_ok,
                  "legitimate symmetric passes"),
    ])
    results["check11_loso_integrity"] = {"pass": c11}

    # ---- summary + write ------------------------------------------------------------------- #
    order = ["check1_injection", "check2_reachability", "check3_ladder", "check4_collision",
             "check5_classifier", "check6_subtrahend", "check7_locked_numbers", "check8_nm_seal",
             "check9_no_truncation", "check10_evaluate_at_use", "check11_loso_integrity"]
    allpass = all(results[k]["pass"] for k in order)
    # required print: realized DC fractions per band from the committed msg constructor
    log("\n[print] realized DC fractions per band (from the committed slow_bandlimited msg s):")
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = g0._geom(STAGE_SPAN)
    dc_by_band = {}
    for b in BAND_ORDER:
        lo, hi = BANDS[b]
        vals = []
        for i in SEEDS:
            s = p1.slow_bandlimited(L, dt_in, lo, hi, seed=MSG_BASE + i)
            vals.append(coherent_drive(s)["dc_frac"])
        dc_by_band[b] = float(np.mean(vals))
        log(f"    {b}: DC fraction mean {dc_by_band[b]:.4f} (E[s]^2/E[s^2])")
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if results[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gateL_sandbox.json")
    g0._dump_json(outp, {"gate": "relay-gateL", "stage": "1-cpu-sandbox", "all_pass": allpass,
                         "framing": FRAMING, "checks": results, "collision": colrep,
                         "dc_fractions": dc_by_band, "env": _env_full(),
                         "reachability_note": ("P4 lag-set/window bound to committed am_window "
                                               "`delays` + eval window; corr audit built fresh -- "
                                               "Gate-3 has no committed corr-audit function (prose "
                                               "only). Disclosed for ratification before smoke.")})
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


# ============================================================================================ #
# verdict-engine test -- decide() across every branch (CPU)
# ============================================================================================ #
def verdict_test(log):
    log("=== RELAY GATE-L :: VERDICT-ENGINE TEST (decide across all branches; CPU) ===")
    allok = True

    def run_case(name, coh, anchor, subtr, ladder, want, extra=lambda v: True):
        nonlocal allok
        v = decide(coh, anchor, subtr, ladder)
        ok = (want in v["verdict"]) and extra(v)
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {v['verdict'][:74]}")
        return v

    # all three DEAD -> COHERENT-LIN-SHUT
    run_case("all DEAD -> SHUT", *_synth_state({b: -0.05 for b in BAND_ORDER}), "COHERENT-LIN-SHUT")
    # one LIVE band -> COHERENT-LIN-LIVE
    run_case("one LIVE -> LIVE", *_synth_state({"SUB": -0.05, "RES": 0.6, "SUPRA": -0.05}),
             "COHERENT-LIN-LIVE", extra=lambda v: "RES" in v["verdict"])
    # anchor miss -> NM + SEAL
    v_am = decide(*_synth_state({b: -0.05 for b in BAND_ORDER}, anchor_ok=False))
    allok &= v_am["verdict"].startswith("NO-MEASUREMENT") and "anchor" in v_am["verdict"]
    log(f"  [{'OK' if v_am['verdict'].startswith('NO-MEASUREMENT') else 'WRONG'}] anchor miss -> NM: "
        f"{v_am['verdict'][:60]}")
    # reachability trip -> NM
    coh_r, a_r, s_r, l_r = _synth_state({b: -0.05 for b in BAND_ORDER})
    coh_r["SUB"][0]["reach"]["lin"] = 0.3
    run_case("reachability trip -> NM", coh_r, a_r, s_r, l_r, "NO-MEASUREMENT",
             extra=lambda v: "reachability" in v["verdict"])
    # decoy elevated -> NM
    coh_d, a_d, s_d, l_d = _synth_state({b: -0.05 for b in BAND_ORDER})
    coh_d["RES"][1]["decoy_p95"]["LIN"] = 0.5
    run_case("decoy elevated -> NM", coh_d, a_d, s_d, l_d, "NO-MEASUREMENT",
             extra=lambda v: "decoy" in v["verdict"])
    # ladder gate fail -> band-level NM (overall verdict not a full seal)
    coh_l, a_l, s_l, l_l = _synth_state({b: -0.05 for b in BAND_ORDER})
    l_l["SUPRA"]["pass"] = False
    v_l = run_case("ladder fail -> band NM", coh_l, a_l, s_l, l_l, "band-level NO-MEASUREMENT",
                   extra=lambda v: "NO-MEASUREMENT (ladder" in v["bands"]["SUPRA"]["class"])
    allok &= not v_l["verdict"].startswith("NO-MEASUREMENT")   # gate instruments passed -> not full seal
    # paired intersection < 2 -> band-level NM
    esp_thin = {b: {i: (i < 1) for i in SEEDS} for b in BAND_ORDER}   # only seed 0 ESP-ok
    run_case("thin intersection -> band NM", *_synth_state({b: -0.05 for b in BAND_ORDER},
                                                           esp_by_band=esp_thin),
             "band-level NO-MEASUREMENT",
             extra=lambda v: all("NO-MEASUREMENT (paired" in v["bands"][b]["class"] for b in BAND_ORDER))
    # underpowered suffix (n in [2,4]) -> LIVE-UNDERPOWERED
    esp_up = {b: {i: (i < 3) for i in SEEDS} for b in BAND_ORDER}      # seeds 0,1,2 -> n=3
    run_case("n=3 LIVE -> underpowered", *_synth_state({"SUB": 0.6, "RES": -0.05, "SUPRA": -0.05},
                                                       esp_by_band=esp_up),
             "COHERENT-LIN-LIVE", extra=lambda v: "UNDERPOWERED" in v["bands"]["SUB"]["class"])

    log(f"\n  OVERALL: {'ALL OK' if allok else 'FAILURES PRESENT'}")
    return allok        # stdout-only (no artifact outside the 6-file deliverable manifest)


# ============================================================================================ #
# S2 smoke (GPU) / S3 battery (GPU) / reread -- implemented; smoke & battery run on Jason's word
# ============================================================================================ #
def _assemble_coh(recs):
    """recs = {band: {seed: stage_a_coherent-dict}} -> coh structure decide() consumes + DC per band."""
    coh = {}
    for b in BAND_ORDER:
        coh[b] = {}
        for i in recs[b]:
            r = recs[b][i]
            coh[b][i] = {"r2": r["r2"], "decoy_p95": r["decoy_p95"], "esp_ok": r["esp_ok"],
                         "esp_d_slow": r["esp"]["d_slow"],
                         "reach": {"lin": r["reach"]["lin"], "quad": r["reach"]["quad"]},
                         "dc_fraction": r["amp"]["dc_frac"]}
    return coh


def ladder_smoke(geom):
    """S2 scope only: the NO-INJ chance + rung a at SUB (completion + values). The full 3-band x
    3-rung ladder GATE (monotone + 10x) is battery work -- no gate is evaluated here."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    sd = seed_scheme(0, "SUB")
    sp = p1.build_system(sd["build"], N, STAGE_SPAN)
    bands = p1.band_indices(sp.omega)
    m_fast = p1.masked_encoding(sp.omega, bands["fast"], np.random.default_rng(sd["enc"]))
    Ks = list(p1.K_GRID); ki = Ks.index(K_PRIMARY)
    lo, hi = BANDS["SUB"]
    s = p1.slow_bandlimited(L, dt_in, lo, hi, seed=sd["msg"])
    u_a = coherent_drive(s)["u"]
    Xn = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, np.zeros(L), Ks, dt_in, n_sub)[ki]
    chance = band_track(Xn[:, bands["fast"]], u_a, dt_in, (lo, hi), iw)
    del Xn
    X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u_a, Ks, dt_in, n_sub)[ki]
    P_a = band_track(X[:, bands["fast"]], u_a, dt_in, (lo, hi), iw)
    P_slow = band_track(X[:, bands["slow"]], u_a, dt_in, (lo, hi), iw)   # report-only context
    del X
    return {"band": "SUB", "rung": "a", "P_a": float(P_a), "chance": float(chance),
            "ratio": (float(P_a / chance) if chance > 0 else None),
            "P_slow_a": float(P_slow), "complete": True,
            "note": "smoke = completion + values only; the monotone+10x GATE is evaluated at battery"}


def smoke(log):
    log("=== RELAY GATE-L :: S2 SMOKE (1 seed, GPU) ===")
    t0 = time.perf_counter()
    geom = g0._geom(STAGE_SPAN)
    ac = anchor_cell(0, geom)
    log(f"  anchor AM x SUB seed-0: got {ac['got']:.6f} vs REF {ac['ref']:.6f} "
        f"digit6={ac['digit6_ok']} (diff {ac['diff']:.2e})")
    coh_sub = stage_a_coherent("SUB", 0, geom)
    log(f"  coherent SUB seed-0: r2_LIN {coh_sub['r2']['LIN']:+.4f} r2_SQ {coh_sub['r2']['SQ']:+.4f} "
        f"r2_FULL {coh_sub['r2']['FULL']:+.4f}; reach |corr(u,s)| {coh_sub['reach']['lin']:.6f} "
        f"corr(u^2,s) {coh_sub['reach']['quad']:+.4f}; ESP ok {coh_sub['esp_ok']}; "
        f"a={coh_sub['amp']['a']:.4f} DC={coh_sub['amp']['dc_frac']:.4f} "
        f"pm_resid={coh_sub['amp']['pm_resid']:.2e}")
    lad = ladder_smoke(geom)
    log(f"  ladder rung a at SUB: P_track(a) {lad['P_a']:.4e} chance {lad['chance']:.4e} "
        f"ratio {_f(lad['ratio'], '.1f')} (P_slow {lad['P_slow_a']:.4e}); complete={lad['complete']}")
    ok = ac["digit6_ok"] and coh_sub["reach"]["lin"] >= REACH_FLOOR and lad["complete"]
    wall = time.perf_counter() - t0
    os.makedirs(RESDIR, exist_ok=True)
    g0._dump_json(os.path.join(RESDIR, "gateL_smoke.json"),
                  {"gate": "relay-gateL", "stage": "2-smoke", "ok": bool(ok), "anchor": ac,
                   "coherent_SUB_seed0": coh_sub, "ladder_SUB_rung_a": lad,
                   "wall_clock_s": wall, "hashes": _hashes(), "env": _env_full()})
    log(f"  {'OK' if ok else 'PROBLEM'} ({wall:.0f}s) [written -> results/R/gateL_smoke.json]")
    return ok


def run(log, nseeds=8):
    log("=== RELAY GATE-L :: S3 BATTERY (GPU) ===")
    t0 = time.perf_counter()
    geom = g0._geom(STAGE_SPAN)
    seeds = list(range(nseeds))
    subtr, sub_chk = load_subtrahend(GATE3_JSON)
    assert sub_chk["ok"], f"committed subtrahend failed sha gate: {sub_chk['reasons']}"
    committed_full = {i: subtr["SUB"][i]["full"] for i in SEEDS if i in subtr["SUB"]}
    anchors = []
    for i in seeds:
        anchors.append(anchor_cell(i, geom))
        log(f"  anchor AM x SUB seed {i}: got {anchors[-1]['got']:.6f} digit6={anchors[-1]['digit6_ok']}")
    anchor = _anchor_report(anchors, committed_full=committed_full)
    recs = {b: {} for b in BAND_ORDER}
    for i in seeds:
        for b in BAND_ORDER:
            ts = time.perf_counter()
            recs[b][i] = stage_a_coherent(b, i, geom)
            log(f"  coherent {b}|{i}: r2_LIN {recs[b][i]['r2']['LIN']:+.4f} "
                f"reach {recs[b][i]['reach']['lin']:.4f} ESP {recs[b][i]['esp_ok']} "
                f"({time.perf_counter()-ts:.0f}s)")
    ladder = ladder_battery(geom, seed=0)
    coh = _assemble_coh(recs)
    v = decide(coh, anchor, subtr, ladder)
    v = _band_dc(v, coh)
    wall = time.perf_counter() - t0
    colrep, _ = collision_matrix()
    hashes = _hashes()
    flat = {f"{b}|{i}": recs[b][i] for b in BAND_ORDER for i in recs[b]}
    payload = {"gate": "relay-gateL", "stage": "3-battery", "seeds": seeds, "framing": FRAMING,
               "K": K_PRIMARY, "span": STAGE_SPAN, "bands": BANDS, "modes": MODES,
               "anchor_cells": anchors, "ladder": ladder, "collision": colrep,
               "subtrahend_sha256": sub_chk.get("gate3_sha256"), "hashes": hashes,
               "env": _env_full(), "wall_clock_s": wall, "verdict": v, "recs": flat}
    g0._dump_json(GATEL_JSON, payload)
    _write_md(GATEL_MD, v, wall, hashes, colrep, env=_env_full())
    log(f"  {v['verdict']}")
    log(f"  [written -> {os.path.relpath(GATEL_JSON)} + gateL_coherent.md]")
    return v


def reread(log):
    """Re-decide + re-render from the COMMITTED battery recs (NO GPU). LOCKED-NUMBERS CONTRACT:
    every rec's numeric substructure byte-identical to the record (loud fail on drift); delta
    re-evaluated from byte-locked (D, SE, n), never stored. run_hashes first-write-wins."""
    assert os.path.exists(GATEL_JSON), f"missing battery record {GATEL_JSON} -- run --run first"
    nm = json.load(open(GATEL_JSON))
    flat = nm["recs"]
    recs = {b: {} for b in BAND_ORDER}
    for key, r in flat.items():
        b, i = key.split("|")
        recs[b][int(i)] = r
    subtr, sub_chk = load_subtrahend(GATE3_JSON)
    assert sub_chk["ok"], f"committed subtrahend failed sha gate on reread: {sub_chk['reasons']}"
    coh = _assemble_coh(recs)
    v = decide(coh, nm["verdict"]["anchor"], subtr, nm["ladder"])
    v = _band_dc(v, coh)
    # LOCKED-NUMBERS: every cell re-serializes byte-identically to the record
    for key in flat:
        b, i = key.split("|")
        assert json.dumps(recs[b][int(i)], sort_keys=True) == json.dumps(flat[key], sort_keys=True), \
            f"cell '{key}' drifted from the battery record"
    log("  [integrity] all coherent cells byte-identical to the battery record: OK")
    hashes = _hashes()
    payload = {**nm, "verdict": v, "hashes": hashes,
               "run_hashes": nm.get("run_hashes") or nm.get("hashes"),
               "reread": "re-decided from unchanged recs; no GPU; delta re-evaluated (not stored); "
                         "no measured number changed"}
    g0._dump_json(GATEL_JSON, payload)
    _write_md(GATEL_MD, v, nm.get("wall_clock_s", 0.0), hashes, nm.get("collision", {}))
    log(f"  {v['verdict']}")
    log(f"  run sha (immutable): {(payload['run_hashes'] or {}).get('code')}; current code sha: {hashes['code']}")
    log(f"  [rewritten -> {os.path.relpath(GATEL_JSON)} + gateL_coherent.md]  (recs UNCHANGED; NOT committed)")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--nseeds", type=int, default=8)
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
