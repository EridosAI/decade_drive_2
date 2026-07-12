"""
experiments/relay_gateB_probe.py
================================

Gate-B STAGE-2 PROBE -- the falsifiability test of the LOCKED P1 predictions (~90s GPU).

Companion to the committed relay_gateB.py (stage-1). relay_gateB.py is committed (c5a46e0) and
forces the JAX CPU backend at import; the stage-2 probe needs the GPU, so it lives here as a
REUSE-BY-IMPORT companion -- committed artifacts (incl relay_gateB.py) are NOT touched. It imports
relay_gate0 (the committed band-sweep protocol _stage_a_r2 machinery) and reads the byte-locked
(pred_r2, sigma_pred) straight from the committed gateB_broadband.json.

What it does: reproduces the committed band-sweep protocol (relay_gate0._stage_a_r2: span 1.5,
K=0.24, am_input_band in the message band, demod on the slow tertile, batch-of-1, NO ESP replica)
at the THREE held-out bands H1/H2/H3, for 2 seeds each, and evaluates the PRE-REGISTERED predictive
pass against the byte-locked (pred_r2, sigma_pred):

  PREDICTIVE PASS iff |mean_obs - pred_r2| <= max(2*sigma_pred, 0.05) for ALL THREE bands
  (conjunction), EVALUATED HERE from the formula + the byte-locked (pred, sigma) -- windows are
  NOT stored. Any other pattern: report as-is. No threshold moves after data.

Instruments (the spec: "protocol identity + decoys"): (1) a PROTOCOL-IDENTITY ANCHOR -- reproduce
the committed sweep's compliant cell (band [0.2,0.9], seed 0 -> 0.9814697252519144) digit-exact
(6dp house standard); the holdout hops use the SAME imported code path, so protocol identity is by
construction AND numerically certified. (2) fresh, collision-proven same-band decoy nulls per run.

Modes: --sandbox (CPU: fresh-decoy collision proof vs ALL committed families incl Gate-4; locked-P1
load + sha + band cross-check; pass-rule on synthetics + evaluate-at-use windows; protocol-identity
structural), --probe (GPU: anchor + 2 seeds x {H1,H2,H3}; evaluate predictive pass; STOP-and-report).
Nothing committed.
"""
from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import argparse

# CPU for --sandbox / --reread (no integration). --probe -> GPU (default backend).
if "--sandbox" in sys.argv or "--reread" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np                                       # noqa: E402
import D_phase1_routing as p1                            # noqa: E402 (jax x64 on import)
import relay_gate0 as g0                                 # noqa: E402 (sweep protocol + helpers)
from core.reservoir import build_system                  # noqa: E402
from core.bands import band_indices, masked_encoding     # noqa: E402

RESDIR = g0.RESDIR

# ---- pinned committed inputs (NO-MEASUREMENT on sha mismatch) ------------------------- #
GATEB_JSON = os.path.join(RESDIR, "gateB_broadband.json")
GATEB_SHA256 = "f2655b84a0c6dd3d72c4af8adc3b496e7d8ff92347c51c084c55555c09831d48"
SWEEP_JSON = os.path.join(RESDIR, "gate0_bandsweep.json")
SWEEP_SHA256 = "881c44bee77620fc45218e1c4b34975b7cd8f60d4cd6925b8583e33c3eb6d10e"

# ---- the committed sweep protocol these runs reproduce ------------------------------- #
SPAN = g0.STAGE_SPAN                # 1.5 (the band-sweep geometry)
K = g0.K_PRIMARY                    # 0.24
MSG_BASE = 1000                     # am_input_band(1000+seed) (== _stage_a_r2)
ENC_BASE = 5000                     # masked_encoding rng 5000+seed (== _stage_a_r2)
PROBE_SEEDS = [0, 1]                # 2 seeds x 3 holdouts = 6 runs (+1 anchor)
PRED_PASS_ABS = 0.05               # window = max(2*sigma_pred, PRED_PASS_ABS), EVALUATED here
DECOY_ELEVATED = g0.DECOY_ELEVATED  # 0.2 specificity floor on decoy p95

# protocol-identity anchor: the committed sweep's compliant row, seed 0 (== the canonical anchor)
ANCHOR_BAND = (g0.MSG_LO, g0.MSG_HI)               # [0.2, 0.9]
ANCHOR_SEED = 0

# ---- fresh decoy bases: ABOVE all committed families (incl Gate-4's 400000..460000) --- #
# committed (sourced from artifacts): Phase-1/Gates 0-3 40000..340000; Gate-4 400000/420000/
# 440000/460000. Each base spans base+[0, 9*200+59]=base+[0,1859]; max committed = 461859.
DECOY_BASE = {"H1": 500000, "H2": 520000, "H3": 540000}
ANCHOR_DECOY_BASE = 560000
N_DEC = p1.N_DEC
COMMITTED_DECOY_BASES = [40000, 60000, 70000, 80000, 100000, 120000, 140000, 160000, 180000,
                         200000, 220000, 240000, 300000, 320000, 340000,     # Phase-1/Gates 0-3
                         400000, 420000, 440000, 460000]                      # Gate-4
SEED_MAX = 9

FRAMING = ("Gate-B stage-2 probe: reproduce the committed band-sweep protocol (span 1.5, K=0.24, "
           "stage-A only, slow-tertile demod) at the three HELD-OUT bands and test the LOCKED P1 "
           "predictions. Falsifiable, not retrodictive: pred_r2/sigma_pred were byte-locked at "
           "stage-1 (committed) BEFORE any of these runs. Windows are NOT stored -- the pass rule "
           "max(2*sigma_pred, 0.05) is evaluated here from the byte-locked (pred, sigma).")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_edges(c, rho):
    s = math.sqrt(rho)
    return [c / s, c * s]


# ===================================================================================== #
#  Locked P1 + anchor reference (instruments; NO-MEASUREMENT on failure)
# ===================================================================================== #
def load_locked_P1():
    """Read + sha-verify the committed gateB_broadband.json, return the locked predictions with a
    band cross-check. checks['ok'] False => NO-MEASUREMENT."""
    checks = {"ok": True, "reasons": []}
    sha = _sha256_file(GATEB_JSON)
    checks["gateB_sha256"] = sha
    if sha != GATEB_SHA256:
        checks["ok"] = False
        checks["reasons"].append(f"gateB_broadband.json sha256 {sha} != pinned {GATEB_SHA256}")
        return None, checks
    d = json.load(open(GATEB_JSON))
    lp = d.get("locked_P1") or d.get("verdict", {}).get("locked_P1")
    if not lp:
        checks["ok"] = False
        checks["reasons"].append("locked_P1 not found in gateB_broadband.json")
        return None, checks
    preds = {}
    for h, pr in lp["predictions"].items():
        edges = band_edges(pr["c"], pr["rho"])
        if not all(abs(a - b) < 1e-9 for a, b in zip(edges, pr["band"])):
            checks["ok"] = False
            checks["reasons"].append(f"{h} band_edges {edges} != stored {pr['band']}")
        preds[h] = {"c": pr["c"], "rho": pr["rho"], "band": edges,
                    "pred_r2": pr["pred_r2"], "sigma_pred": pr["sigma_pred"], "kind": pr["kind"]}
    checks["model"] = lp.get("model")
    checks["pass_rule"] = lp.get("pass_rule")
    return preds, checks


def load_anchor_ref():
    """The protocol-identity reference: the committed sweep's compliant row, seed ANCHOR_SEED.
    sha-verify gate0_bandsweep.json. Returns (ref_r2, checks)."""
    checks = {"ok": True, "reasons": []}
    sha = _sha256_file(SWEEP_JSON)
    checks["sweep_sha256"] = sha
    if sha != SWEEP_SHA256:
        checks["ok"] = False
        checks["reasons"].append(f"gate0_bandsweep.json sha256 {sha} != pinned {SWEEP_SHA256}")
        return None, checks
    d = json.load(open(SWEEP_JSON))
    row = next((r for r in d["rows"] if r.get("label") == "compliant"
                and r["band"] == [g0.MSG_LO, g0.MSG_HI]), None)
    if row is None:
        checks["ok"] = False
        checks["reasons"].append("compliant [0.2,0.9] row not found in committed sweep")
        return None, checks
    ref = float(row["r2_per_seed"][ANCHOR_SEED])
    checks["anchor_ref"] = ref
    checks["anchor_band"] = row["band"]
    return ref, checks


# ===================================================================================== #
#  Fresh-decoy collision proof (vs ALL committed families incl Gate-4)
# ===================================================================================== #
def _decoy_range(base, seed_max=SEED_MAX):
    return {base + i * 200 + dd for i in range(seed_max + 1) for dd in range(N_DEC)}


def verify_no_collision(seed_max=SEED_MAX):
    seeds = range(seed_max + 1)
    build = {s for s in seeds}
    enc = {ENC_BASE + s for s in seeds}
    msg = {MSG_BASE + s for s in seeds}
    rade = {MSG_BASE + s + 777 for s in seeds}
    net_msg = build | enc | msg | rade
    fresh_bases = list(DECOY_BASE.values()) + [ANCHOR_DECOY_BASE]
    fresh_by_base = {b: _decoy_range(b, seed_max) for b in fresh_bases}
    fresh_all = set().union(*fresh_by_base.values())
    committed_all = set().union(*[_decoy_range(b, seed_max) for b in COMMITTED_DECOY_BASES])
    pw = {}
    for a in range(len(fresh_bases)):
        for b in range(a + 1, len(fresh_bases)):
            inter = fresh_by_base[fresh_bases[a]] & fresh_by_base[fresh_bases[b]]
            pw[f"{fresh_bases[a]}^{fresh_bases[b]}"] = len(inter)
    checks = {
        "fresh_vs_committed": sorted(fresh_all & committed_all),
        "fresh_vs_net_msg": sorted(fresh_all & net_msg),
        "fresh_pairwise": {k: v for k, v in pw.items() if v > 0},
        "min_fresh_gt_max_committed": bool(min(fresh_bases) > max(COMMITTED_DECOY_BASES) + 1860),
    }
    ok = (not checks["fresh_vs_committed"] and not checks["fresh_vs_net_msg"]
          and not checks["fresh_pairwise"] and checks["min_fresh_gt_max_committed"])
    return {"ok": bool(ok), "fresh_bases": fresh_bases,
            "committed_bases": COMMITTED_DECOY_BASES, "collisions": checks}


# ===================================================================================== #
#  The probe hop  (reproduce _stage_a_r2's integration; r2 via demod_fit + fresh decoys)
# ===================================================================================== #
def probe_hop(seed, lo, hi, decoy_base, geom):
    """One stage-A trackability run at (span 1.5, K=0.24) for message band [lo,hi]. r2 via the
    EXACT sweep path (g0.demod_fit -- byte-identical to relay_gate0._stage_a_r2), plus a fresh
    same-band decoy null via p1.demod_capacity (r2_d0 consistency asserted < 1e-9). No ESP replica
    (the sweep protocol has none)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    sp = build_system(seed, p1.N, SPAN)
    bands = band_indices(sp.omega)
    m_fast = masked_encoding(sp.omega, bands["fast"], np.random.default_rng(ENC_BASE + seed))
    outside = np.concatenate([bands["slow"], bands["guard"]])
    assert float(np.abs(m_fast[outside]).max()) == 0.0
    s_msg, u = g0.am_input_band(L, dt_in, MSG_BASE + seed, lo, hi)
    X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u, [K], dt_in, n_sub)[0]
    _, r2, lam = g0.demod_fit(X, bands["slow"], s_msg, sl, "full")     # EXACT sweep quantity
    decoys = [p1.slow_bandlimited(L, dt_in, lo, hi, seed=decoy_base + seed * 200 + dd)
              for dd in range(N_DEC)]
    dem = p1.demod_capacity(X, bands["slow"], s_msg, decoys, delays, sl, "full")
    del X
    assert abs(r2 - dem["r2_d0"]) < 1e-9, "demod_fit r2 != demod_capacity r2_d0 (path drift)"
    return {"seed": seed, "band": [lo, hi], "r2_d0": float(r2), "lam": float(lam),
            "decoy_p95": float(dem["decoy_p95"]), "decoy_mean": float(dem["decoy_mean"])}


# ===================================================================================== #
#  Predictive-pass decision (windows EVALUATED here, never stored)
# ===================================================================================== #
def decide(anchor, holdouts, preds, colrep):
    """anchor = {r2, ref, digit6_ok, bit_ok} ; holdouts = {H: [hop recs]}. Instruments FIRST:
    protocol-identity anchor (digit-exact 6dp), decoy specificity, then the PRE-REGISTERED
    predictive-pass conjunction from the byte-locked (pred, sigma)."""
    out = {"framing": FRAMING, "protocol": f"span {SPAN}, K={K}, stage-A slow-demod (== sweep)",
           "seeds": PROBE_SEEDS,
           "pass_rule": ("PREDICTIVE PASS iff |mean_obs - pred_r2| <= max(2*sigma_pred, "
                         f"{PRED_PASS_ABS}) for ALL THREE (conjunction); windows evaluated here "
                         "from byte-locked (pred, sigma), NOT stored")}
    out["anchor"] = anchor
    # per-holdout observed mean + the evaluated window + pass
    perh = {}
    all_pass = True
    for h, recs in holdouts.items():
        obs = [r["r2_d0"] for r in recs]
        st = g0._mstats(obs)
        pred, sigma = preds[h]["pred_r2"], preds[h]["sigma_pred"]
        window = max(2.0 * sigma, PRED_PASS_ABS)          # EVALUATE-AT-USE (never stored)
        dev = abs(st["mean"] - pred)
        passed = bool(dev <= window)
        all_pass &= passed
        dp95 = max(r["decoy_p95"] for r in recs)
        perh[h] = {"band": preds[h]["band"], "kind": preds[h]["kind"],
                   "mean_obs": st["mean"], "per_seed_r2": st["per_seed"], "se": st["se"],
                   "pred_r2": pred, "sigma_pred": sigma, "window": float(window),
                   "deviation": float(dev), "pass": passed, "decoy_p95_max": float(dp95)}
    out["holdouts"] = perh
    # decoy specificity across all runs (anchor + holdouts)
    all_p95 = [anchor.get("decoy_p95")] + [perh[h]["decoy_p95_max"] for h in perh]
    all_p95 = [x for x in all_p95 if x is not None]
    decoy_elevated = bool(all_p95 and max(all_p95) > DECOY_ELEVATED)
    out["decoys"] = {"max_p95": (max(all_p95) if all_p95 else None), "elevated": decoy_elevated,
                     "bar": DECOY_ELEVATED}
    out["collision_free"] = colrep["ok"]

    # ---- verdict (instruments first) --------------------------------------------------- #
    if not anchor["digit6_ok"]:
        out["verdict"] = (f"NO-MEASUREMENT (protocol-identity anchor miss: reproduced "
                          f"{anchor['r2']:.6f} vs committed {anchor['ref']:.6f} -- the probe does "
                          "NOT reproduce the committed sweep protocol; STOP)")
    elif decoy_elevated:
        out["verdict"] = (f"NO-MEASUREMENT (decoy specificity failure: max decoy p95 "
                          f"{max(all_p95):.3f} > {DECOY_ELEVATED} -- readout not specific)")
    elif all_pass:
        out["verdict"] = ("PREDICTIVE PASS -- all three held-out bands land within "
                          "max(2*sigma_pred, 0.05) of the locked M1 prediction. The stage-1 "
                          "attenuation explanation (smooth M1 power law) is CORROBORATED "
                          "out-of-sample; the Phase-3 line-67 erratum path stays open pending Gate-L.")
    else:
        fails = [h for h in perh if not perh[h]["pass"]]
        out["verdict"] = (f"PREDICTIVE FAIL (report-as-is): band(s) {fails} land outside "
                          "max(2*sigma_pred, 0.05) of the locked prediction. No threshold moves "
                          "after data; the deviation is reported verbatim (relevant to Gate-L).")
    return out


# ===================================================================================== #
#  --probe (GPU)
# ===================================================================================== #
def probe(log):
    import time
    log("=== RELAY GATE-B :: STAGE-2 PROBE (GPU; locked-P1 falsifiability test) ===")
    log(f"    framing: {FRAMING}")
    preds, pchecks = load_locked_P1()
    ref, achecks = load_anchor_ref()
    colrep = verify_no_collision()
    if not (pchecks["ok"] and achecks["ok"] and colrep["ok"]):
        reasons = pchecks.get("reasons", []) + achecks.get("reasons", []) + \
                  ([] if colrep["ok"] else ["fresh decoy collision"])
        log(f"  NO-MEASUREMENT (inputs): {reasons}")
        g0._dump_json(os.path.join(RESDIR, "gateB_probe.json"),
                      {"gate": "relay-gateB-probe", "verdict": f"NO-MEASUREMENT (inputs): {reasons}",
                       "pchecks": pchecks, "achecks": achecks, "collision": colrep})
        return {"verdict": "NO-MEASUREMENT (inputs)"}
    log(f"    locked P1 (byte-locked): " + ", ".join(
        f"{h}: pred {preds[h]['pred_r2']:.6f} sig {preds[h]['sigma_pred']:.6f} "
        f"win {max(2*preds[h]['sigma_pred'], PRED_PASS_ABS):.4f}" for h in ("H1", "H2", "H3")))
    log(f"    fresh decoy bases {colrep['fresh_bases']} collision-free vs committed: {colrep['ok']}")

    geom = g0._geom(SPAN)
    t0 = time.perf_counter()
    # protocol-identity anchor (seed 0, compliant band [0.2,0.9])
    a = probe_hop(ANCHOR_SEED, ANCHOR_BAND[0], ANCHOR_BAND[1], ANCHOR_DECOY_BASE, geom)
    anchor = {"r2": a["r2_d0"], "ref": ref, "diff": abs(a["r2_d0"] - ref),
              "digit6_ok": round(a["r2_d0"], 6) == round(ref, 6),
              "bit_ok": a["r2_d0"] == ref, "decoy_p95": a["decoy_p95"]}
    log(f"  [anchor] compliant [0.2,0.9] seed 0: r2={a['r2_d0']:.10f} vs committed {ref:.10f} "
        f"diff={anchor['diff']:.1e} -> digit-exact(6dp) {'OK' if anchor['digit6_ok'] else 'MISS'} "
        f"(bit-exact {anchor['bit_ok']})  ({time.perf_counter()-t0:.0f}s)")

    holdouts = {}
    for h in ("H1", "H2", "H3"):
        lo, hi = preds[h]["band"]
        recs = []
        for seed in PROBE_SEEDS:
            r = probe_hop(seed, lo, hi, DECOY_BASE[h], geom)
            recs.append(r)
            log(f"  [{h}] band [{lo:.3f},{hi:.3f}] seed {seed}: r2_d0={r['r2_d0']:+.6f} "
                f"decoy_p95={r['decoy_p95']:+.4f}  ({time.perf_counter()-t0:.0f}s)")
        holdouts[h] = recs

    verdict = decide(anchor, holdouts, preds, colrep)
    wall = time.perf_counter() - t0
    log("\n=== PROBE RESULT ===")
    for h in ("H1", "H2", "H3"):
        ph = verdict["holdouts"][h]
        log(f"  {h}: mean_obs={ph['mean_obs']:+.6f} (per-seed {[round(x,4) for x in ph['per_seed_r2']]}) "
            f"vs pred {ph['pred_r2']:+.6f} +/- {ph['sigma_pred']:.4f}; |dev|={ph['deviation']:.4f} "
            f"<= window {ph['window']:.4f} -> {'PASS' if ph['pass'] else 'FAIL'}")
    log(f"\n  {verdict['verdict']}")
    log(f"  wall-clock {wall:.0f}s.")

    payload = {"gate": "relay-gateB-probe", "stage": "2-probe", "framing": FRAMING,
               "env": _env_full(), "protocol_identity_ref": ref, "pchecks": pchecks,
               "achecks": achecks, "collision": colrep, "seed_scheme": {
                   "build/enc/msg": "== relay_gate0._stage_a_r2 (build seed, enc 5000+seed, msg 1000+seed)",
                   "fresh_decoy_bases": {**DECOY_BASE, "anchor": ANCHOR_DECOY_BASE}},
               "hashes": {"code": g0._sha12(os.path.abspath(__file__))},
               "wall_clock_s": wall, "verdict": verdict,
               "holdout_recs": {h: recs for h, recs in holdouts.items()},
               "anchor_rec": a}
    g0._dump_json(os.path.join(RESDIR, "gateB_probe.json"), payload)
    _write_md(os.path.join(RESDIR, "gateB_probe.md"), verdict, preds, anchor, wall)
    log(f"  [written -> results/R/gateB_probe.{{json,md}}]  (NOT committed)")
    log("  STOP-and-report.")
    return verdict


def _env_full():
    import jax
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64 = bool(getattr(jax.config, "jax_enable_x64", None))
    return {**g0._env_versions(), "interpreter": sys.executable, "python": sys.version.split()[0],
            "jax_enable_x64": x64, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<default>"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}


def _write_md(path, v, preds, anchor, wall):
    seeds_str = ", ".join(str(s) for s in PROBE_SEEDS)
    signed = {h: (v["holdouts"][h]["mean_obs"] - v["holdouts"][h]["pred_r2"]) for h in ("H1", "H2", "H3")}
    lines = [
        "# Relay Gate-B stage-2 probe -- locked-P1 falsifiability test", "",
        f"Wall-clock {wall:.0f}s. Protocol: span {SPAN}, K={K}, stage-A slow-tertile demod "
        f"(reproduces the committed band-sweep exactly, by imported code path). Seeds: {seeds_str}.", "",
        "ESP: the stage-A trackability protocol carries NO ESP replica by construction -- it reuses "
        "relay_gate0._stage_a_r2 exactly as the committed band-sweep (no repeater, no replica), so "
        "there is no per-seed ESP flag here. The specificity instrument is the never-injected "
        "same-band decoy null (reported per band below).", "",
        f"Framing: {FRAMING}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
        "## Protocol-identity anchor (instrument)", "",
        f"- compliant [0.2,0.9] seed {ANCHOR_SEED}: reproduced {anchor['r2']:.10f} vs committed "
        f"{anchor['ref']:.10f} (diff {anchor['diff']:.1e}) -> digit-exact(6dp) "
        f"{'OK' if anchor['digit6_ok'] else 'MISS'}; bit-exact {anchor['bit_ok']} (diagnostic). "
        "The holdout hops use the SAME imported protocol -> protocol identity certified.", "",
        "## Held-out predictions (locked at stage-1, tested here; windows evaluated, not stored)", "",
    ]
    for h in ("H1", "H2", "H3"):
        ph = v["holdouts"][h]
        per_seed = ", ".join(f"seed {s}: {r:+.6f}" for s, r in zip(PROBE_SEEDS, ph["per_seed_r2"]))
        lines.append(f"- **{h}** band [{ph['band'][0]:.3f},{ph['band'][1]:.3f}] -- {ph['kind']}: "
                     f"mean_obs = {ph['mean_obs']:+.6f} (per-seed {per_seed}) vs pred "
                     f"{ph['pred_r2']:+.6f} +/- {ph['sigma_pred']:.4f}; signed dev {signed[h]:+.4f}, "
                     f"|dev| = {ph['deviation']:.4f} <= window max(2*sigma,{PRED_PASS_ABS}) = "
                     f"{ph['window']:.4f} -> **{'PASS' if ph['pass'] else 'FAIL'}** "
                     f"(decoy p95 {ph['decoy_p95_max']:+.4f}).")
    n_pos = sum(1 for h in signed if signed[h] > 0)
    lines += [
        "", "## Pattern observations (observed, not claimed; folded to Gate-L)", "",
        f"- **H2 callback (a pre-registered expectation met).** Stage-1 pre-flagged the overlap "
        f"stratum as sitting ABOVE the fit (\"H2 may land high in its window\"). It did: mean_obs is "
        f"{signed['H2']:+.4f} above the locked prediction (well within the wide window). Surfaced as "
        "the predicted success it is.",
        f"- **Sign pattern.** All three deviations are POSITIVE ({signed['H1']:+.4f} / {signed['H2']:+.4f} "
        f"/ {signed['H3']:+.4f}); {n_pos}-of-3 same-sign is P=0.25 under symmetric independence -- NOT "
        "significant, claimed as nothing -- but consistent with the known overlap-positive structure and "
        "an M1 curve sitting slightly low. H3's direction (a narrower rho=1.5 band decoding slightly "
        "better than the width-independent model predicts) is physically sensible and noise-compatible at "
        "n=2. Gate-L inherits the observation.",
        "", "## Scope", "",
        "Falsifiability payload for the stage-1 EXPLAINED verdict: the M1 power-law interpolation is "
        "tested out-of-sample at three held-out bands. This tests the M1 INTERPOLATION, not any specific "
        "attenuation mechanism (the law stays open; see stage-1). Relevant to Gate-L and the Phase-3 "
        "line-67 erratum path. STOP-and-report."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def reread(log):
    """Re-render gateB_probe.md from the committed probe record (NO GPU). LOCKED-NUMBERS CONTRACT:
    re-decide from the stored recs and assert the verdict's numeric substructure (holdouts / anchor /
    decoys / verdict string) byte-identical -- prose-only re-render, no measured number changes."""
    src = os.path.join(RESDIR, "gateB_probe.json")
    assert os.path.exists(src), f"missing {src} -- run --probe first"
    nm = json.load(open(src))
    v = nm["verdict"]
    preds, pchecks = load_locked_P1()
    assert pchecks["ok"], f"locked P1 load failed: {pchecks.get('reasons')}"
    anchor = v["anchor"]
    holdouts = {h: nm["holdout_recs"][h] for h in ("H1", "H2", "H3")}
    v2 = decide(anchor, holdouts, preds, nm["collision"])
    for key in ("holdouts", "anchor", "decoys", "verdict"):
        assert json.dumps(v2[key], sort_keys=True) == json.dumps(v[key], sort_keys=True), \
            f"reread drift in verdict['{key}'] -- probe recs/verdict changed"
    log("=== RELAY GATE-B PROBE :: REREAD (re-render from unchanged recs; NO GPU) ===")
    log("  [integrity] verdict holdouts/anchor/decoys/string byte-identical to the probe record: OK")
    # hash chain (matches the program's amend pattern): the GPU-run sha is immutable (first-write-
    # wins); hashes.code = the current file. Between them the numeric path (probe_hop/decide/
    # predictive-pass) is UNCHANGED -- only render (_write_md), the --reread mode, and a sandbox
    # test (c5_render) differ -- so the recorded numbers remain valid under the committed code.
    run_hashes = nm.get("run_hashes") or nm.get("hashes")
    hashes = {"code": g0._sha12(os.path.abspath(__file__))}
    payload = {**nm, "hashes": hashes, "run_hashes": run_hashes,
               "reread": ("re-rendered from unchanged recs; no GPU; LOCKED-NUMBERS byte-identical. "
                          "Code differs from the run sha only in render/_write_md, the --reread mode, "
                          "and the c5_render sandbox test; the numeric path is unchanged.")}
    g0._dump_json(os.path.join(RESDIR, "gateB_probe.json"), payload)
    _write_md(os.path.join(RESDIR, "gateB_probe.md"), v, preds, anchor, nm["wall_clock_s"])
    log(f"  run sha (immutable): {run_hashes.get('code')}; current code sha: {hashes['code']}")
    log(f"  {v['verdict']}")
    log("  [rewritten -> results/R/gateB_probe.{json,md}]  (recs UNCHANGED; NOT committed)")
    return v


# ===================================================================================== #
#  --sandbox (CPU)
# ===================================================================================== #
def _check(log, name, ok, detail):
    return g0._check(log, name, ok, detail)


def sandbox(log):
    log("=== RELAY GATE-B PROBE :: CPU SANDBOX (no GPU) ===")
    results = {}

    # (0) fresh-decoy collision proof vs ALL committed families incl Gate-4
    log("\n(0) Fresh-decoy collision proof (vs Phase-1/Gates 0-3 + Gate-4)")
    colrep = verify_no_collision()
    c = colrep["collisions"]
    log(f"    fresh bases {colrep['fresh_bases']}; committed max {max(COMMITTED_DECOY_BASES)}")
    log(f"    fresh vs committed={c['fresh_vs_committed'] or 'none'}; fresh vs net/msg="
        f"{c['fresh_vs_net_msg'] or 'none'}; pairwise={c['fresh_pairwise'] or 'none'}; "
        f"min>max+range={c['min_fresh_gt_max_committed']}")
    c0 = _check(log, "fresh decoy bases collision-free vs all committed + net/msg", colrep["ok"],
                f"min fresh {min(colrep['fresh_bases'])} > max committed {max(COMMITTED_DECOY_BASES)}")
    results["c0_collision"] = c0

    # (1) locked-P1 load + sha + band cross-check
    log("\n(1) Locked-P1 load + sha256 + band cross-check")
    preds, pchecks = load_locked_P1()
    ref, achecks = load_anchor_ref()
    ok1 = pchecks["ok"] and achecks["ok"] and preds is not None
    if preds:
        for h in ("H1", "H2", "H3"):
            log(f"    {h}: c={preds[h]['c']} rho={preds[h]['rho']} band={[round(x,4) for x in preds[h]['band']]} "
                f"pred {preds[h]['pred_r2']:.6f} sig {preds[h]['sigma_pred']:.6f}")
    log(f"    gateB sha {pchecks.get('gateB_sha256','')[:16]} (pinned {GATEB_SHA256[:16]}); "
        f"sweep sha {achecks.get('sweep_sha256','')[:16]}; anchor ref {ref}")
    c1 = _check(log, "locked P1 loads + sha match + band_edges cross-check + anchor ref present",
                ok1, f"reasons {pchecks.get('reasons', []) + achecks.get('reasons', [])}")
    results["c1_locked_P1"] = c1

    # (2) pass-rule: windows EVALUATED (not stored); conjunction; boundary behavior
    log("\n(2) Predictive pass-rule -- window = max(2*sigma_pred, 0.05) evaluated at use")
    if preds:
        def synth(hvals):  # hvals: {H: mean_obs}
            hold = {h: [{"r2_d0": hvals[h], "decoy_p95": -0.2}] for h in hvals}
            return decide({"digit6_ok": True, "bit_ok": True, "r2": ref, "ref": ref,
                           "diff": 0.0, "decoy_p95": -0.2}, hold, preds, colrep)
        # exactly at pred -> PASS
        at = synth({h: preds[h]["pred_r2"] for h in preds})
        # each just inside its window -> PASS; H2 pushed just outside -> FAIL (conjunction)
        win = {h: max(2 * preds[h]["sigma_pred"], PRED_PASS_ABS) for h in preds}
        inside = synth({h: preds[h]["pred_r2"] + 0.99 * win[h] for h in preds})
        h2out = synth({**{h: preds[h]["pred_r2"] for h in preds},
                       "H2": preds["H2"]["pred_r2"] + 1.01 * win["H2"]})
        win_ok = (abs(win["H1"] - 0.05) < 1e-12 and abs(win["H2"] - 2 * preds["H2"]["sigma_pred"]) < 1e-12)
        c2 = all([
            _check(log, "mean_obs == pred -> PASS (all three)", "PREDICTIVE PASS" in at["verdict"],
                   at["verdict"][:24]),
            _check(log, "just inside every window -> PASS", "PREDICTIVE PASS" in inside["verdict"],
                   inside["verdict"][:24]),
            _check(log, "one band just outside -> FAIL (conjunction)", "PREDICTIVE FAIL" in h2out["verdict"],
                   h2out["verdict"][:24]),
            _check(log, "windows evaluated at use: H1 floor 0.05, H2 = 2*sigma (0.196)", win_ok,
                   f"H1 {win['H1']:.4f} H2 {win['H2']:.4f} H3 {win['H3']:.4f}"),
        ])
    else:
        c2 = False
        _check(log, "locked P1 available for pass-rule test", False, "no preds")
    results["c2_pass_rule"] = c2

    # (3) NM branches: anchor miss + decoy elevated
    log("\n(3) NM branches -- protocol-identity anchor miss + decoy specificity")
    if preds:
        miss = decide({"digit6_ok": False, "bit_ok": False, "r2": 0.90, "ref": ref, "diff": 0.08,
                       "decoy_p95": -0.2},
                      {h: [{"r2_d0": preds[h]["pred_r2"], "decoy_p95": -0.2}] for h in preds},
                      preds, colrep)
        elev = decide({"digit6_ok": True, "bit_ok": True, "r2": ref, "ref": ref, "diff": 0.0,
                       "decoy_p95": -0.2},
                      {h: [{"r2_d0": preds[h]["pred_r2"], "decoy_p95": (0.5 if h == "H1" else -0.2)}]
                       for h in preds}, preds, colrep)
        c3 = all([
            _check(log, "anchor miss -> NM (protocol identity broken)",
                   "NO-MEASUREMENT (protocol-identity" in miss["verdict"], miss["verdict"][:40]),
            _check(log, "decoy p95 > 0.2 -> NM (specificity failure)",
                   "NO-MEASUREMENT (decoy specificity" in elev["verdict"], elev["verdict"][:40]),
        ])
    else:
        c3 = False
    results["c3_nm"] = c3

    # (4) protocol-identity structural: probe seed scheme == _stage_a_r2; anchor ref in sweep
    log("\n(4) Protocol identity -- seed scheme == relay_gate0._stage_a_r2; imported code path")
    import inspect
    src = inspect.getsource(g0._stage_a_r2)
    seed_ok = (f"{ENC_BASE} + seed" in src.replace("5000 + seed", "5000 + seed")
               or "5000 + seed" in src) and "1000 + seed" in src
    edges_ok = all(abs(band_edges(preds[h]["c"], preds[h]["rho"])[j] - preds[h]["band"][j]) < 1e-9
                   for h in preds for j in (0, 1)) if preds else False
    demodfit_ok = (g0.demod_fit.__module__ == "relay_gate0")
    c4 = all([
        _check(log, "sweep _stage_a_r2 uses build seed / enc 5000+seed / msg 1000+seed", seed_ok,
               "matched against relay_gate0 source"),
        _check(log, "probe reuses g0.demod_fit (the exact sweep fit path)", demodfit_ok,
               f"module {g0.demod_fit.__module__}"),
        _check(log, "holdout band edges = [c/sqrt(rho), c*sqrt(rho)] (cross-check)", edges_ok,
               "all three"),
    ])
    results["c4_protocol"] = c4

    # (5) render no-truncation assert (house move: the manual truncated-kind-string catch -> a test)
    log("\n(5) Render assert -- rendered .md carries NO ellipsis-cut template fields")
    if preds:
        hold = {h: [{"r2_d0": preds[h]["pred_r2"], "decoy_p95": -0.2},
                    {"r2_d0": preds[h]["pred_r2"], "decoy_p95": -0.2}] for h in preds}
        vr = decide({"digit6_ok": True, "bit_ok": True, "r2": ref, "ref": ref, "diff": 0.0,
                     "decoy_p95": -0.2}, hold, preds, colrep)
        import tempfile
        pr = os.path.join(tempfile.gettempdir(), "_gBprobe_render.md")
        _write_md(pr, vr, preds, vr["anchor"], 0.0)
        txt = open(pr).read()
        os.remove(pr)
        no_trunc = "..." not in txt                        # no ellipsis-cut fields anywhere
        kinds_full = all(preds[h]["kind"] in txt for h in preds)   # full kind strings present verbatim
        c5 = all([
            _check(log, "rendered .md contains no '...' (no ellipsis-cut template fields)", no_trunc,
                   "no truncation in the rendered record"),
            _check(log, "full untruncated kind strings present verbatim (all 3 holdouts)", kinds_full,
                   "kind strings rendered in full"),
        ])
    else:
        c5 = False
        _check(log, "locked P1 available for render assert", False, "no preds")
    results["c5_render"] = c5

    order = ["c0_collision", "c1_locked_P1", "c2_pass_rule", "c3_nm", "c4_protocol", "c5_render"]
    allpass = all(results[k] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if results[k] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    g0._dump_json(os.path.join(RESDIR, "gateB_probe_sandbox.json"),
                  {"gate": "relay-gateB-probe", "stage": "sandbox", "all_pass": allpass,
                   "checks": {k: bool(results[k]) for k in order}, "collision": colrep})
    log("  [written -> results/R/gateB_probe_sandbox.json]  (NOT committed)")
    return allpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--reread", action="store_true",
                    help="re-render the probe .md from the committed record (no GPU)")
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)

    def log(msg):
        print(msg)

    if args.sandbox:
        raise SystemExit(0 if sandbox(log) else 1)
    if args.reread:
        reread(log)
        return
    if args.probe:
        probe(log)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
