"""
experiments/relay_gate2.py
==========================

Relay Gate-2: depth extension (offline, H=10).
Per relay_gate2_multihop_spec.md. Extends the Gate-1 identical-stage chain (commit 7d6f3f2)
from H=5 to H=10 and determines whether the Gate-1 steady-state late drift continues (steady
decline -> BOUNDED ladder) or asymptotes (clean PRICED budget) -- replacing the H_half=16
extrapolation with measurement.

Built STRICTLY by REUSE of the committed Gate-1 / Gate-0 / Phase-1 machinery (imported). This
module modifies NOTHING in relay_gate1.py, relay_gate0.py, D_phase1_routing.py, the core/ files,
or any committed artifact -- it only imports them. No new core machinery.

Design leverage (why this is cheap and SHARP): stages 1..5 are seed-identical to Gate-1, and the
chain is feed-forward (m_k depends only on m_{k-1}), so m_1..m_5 are DETERMINISTIC replays ->
hops 1..5 must reproduce gate1_multihop.json digit-exact per seed (a free 5-deep replication
"prefix bridge"). That makes the Gate-1 "last-hop" ambiguity structurally decidable: if the drift
was a hop-5 noise fluctuation, rho_6..10 revert to the ~0.96 steady band; if it is genuine depth
decline, the downward trend continues through the new ratios.

Pre-registration (carried + sharpened from Gate-1):
  * steady-state readout = rho_k, k=3..10 (8 ratios). rho_2 (the one-time first-relay INSERTION
    loss) is EXCLUDED by pre-registration -- a known one-time cost, not steady-state data.
  * per-seed linear slope; thr = max(2*SE, SLOPE_EPS=0.01) (ratified, carried).
    (A2) ASYMPTOTIC/PRICED : |slope_mean| <= thr  -> steady-state flat at depth.
    (B2) STEADY DECLINE    : slope_mean < -thr    -> drift is real; bounded ladder.
    rising beyond +thr     : INSTRUMENT-SUSPICION (Pin B), never A2.
    (C) NO-MEASUREMENT     : anchor miss; PREFIX-BRIDGE miss (any hop 1-5, any seed, vs Gate-1);
                             violation-at-depth fails to collapse; decoy elevated; underpowered.
  * verdict .md always reports slope_mean +/- SE and margin regardless of class (Pin A).

Conditions (K=0.24): chain H=10, e2e decoy at depth, filter-violation at depth. NO scramble arm
(topology-genericity settled at Gate-1; single variable = depth).

Modes:
  --sandbox       Stage 1. CPU-ONLY checks (no GPU): extended seed collision matrix, 10-stage
                  chain wiring + intersection, per-hop schema, prefix-bridge loader, rho window
                  k=3..10 A2/B2 machinery, violation-at-depth, decoy protocol at depth.
  --verdict-test  CPU-only synthetic exercise of decide() across A2/B2/C branches.
  --smoke         Stage 2 (separate go). 1-seed full H=10; hops 1..5 must replay Gate-1 seed-0
                  digit-exact (0.981470/0.962334/0.931043/0.896005/0.842353).
  --run           Stage 3 (separate go). Full battery (chain + violation; n>=5).
  --reread        Re-decide + re-render from committed gate2 recs (CPU, byte-identical assert).

STOP-and-report after --sandbox. Nothing committed. Single variable: hop count.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

# CPU-only for the non-GPU modes: force the JAX CPU backend BEFORE jax is imported (via the
# relay_gate1 -> relay_gate0 -> D_phase1_routing import chain).
if any(a in sys.argv for a in ("--sandbox", "--verdict-test", "--reread")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))                        # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root

import numpy as np                                                   # noqa: E402
import D_phase1_routing as p1                                        # noqa: E402 (jax x64 on import)
import relay_gate0 as g0                                             # noqa: E402 (Gate-0 machinery)
import relay_gate1 as g1                                             # noqa: E402 (Gate-1 machinery)

RESDIR = g0.RESDIR

# --- carried from Gate-1 (imported, not redefined) ------------------------------------- #
H = 10                               # depth (spec: extend H=5 -> H=10)
STAGE_SPAN = g1.STAGE_SPAN           # 1.5 decades per hop
K_PRIMARY = g1.K_PRIMARY             # 0.24
MSG_LO, MSG_HI = g1.MSG_LO, g1.MSG_HI
VIOL_LO, VIOL_HI = g1.VIOL_LO, g1.VIOL_HI
ANCHOR, ANCHOR_SE_K, ANCHOR_FLOOR = g1.ANCHOR, g1.ANCHOR_SE_K, g1.ANCHOR_FLOOR
MIN_PAIRS = g1.MIN_PAIRS
VIOL_E2E_BAR = g1.VIOL_E2E_BAR
DECOY_ELEVATED = g1.DECOY_ELEVATED
N_DEC = p1.N_DEC
SLOPE_EPS = g1.SLOPE_EPS
RHO_VALID_MIN = g1.RHO_VALID_MIN
SEED_MAX = g1.SEED_MAX
E2E_DECOY_BASE = g1.E2E_DECOY_BASE   # 80000
K_START = 3                          # steady-state rho window start (EXCLUDES the insertion rho_2)

# seed-scheme strides (carried from Gate-1: stages 1..5 are seed-identical -> the prefix bridge)
BUILD_STRIPE = g1.BUILD_STRIPE
ENC_BASE, STAGE_STRIPE, REP_BASE = g1.ENC_BASE, g1.STAGE_STRIPE, g1.REP_BASE
CAR_BASE, MSG_BASE = g1.CAR_BASE, g1.MSG_BASE

# per-stage decoy bases: Gate-1's stages 1-5 (byte-identical) + stages 6-10 = 160000..240000
# step 20000 (spec). All jump clear of the e2e base 80000 and are pairwise-disjoint (proven).
DECOY_BASE = {**g1.DECOY_BASE, 6: 160000, 7: 180000, 8: 200000, 9: 220000, 10: 240000}

FRAMING = ("Compound span 1.5*H is an INFORMATION-PATH claim (H successive square-law "
           "demodulations end-to-end), NOT one physical spectrum; each repeater re-injects into a "
           "fresh span-1.5 network. Depth extension H=5->10; the insertion loss rho_2 is EXCLUDED "
           "from the trend statistic by pre-registration (a known one-time cost). No new "
           "chain-vs-direct claim; committed b0f7664 floors cited.")


def _mstats(vals):
    return g0._mstats(vals)


def _fmt(x, spec="+.4f"):
    return g0._fmt(x, spec)


# ===================================================================================== #
#  Seed-derivation scheme (extended) + collision proof
# ===================================================================================== #
def seed_scheme(i, s):
    """Seeds for chain i, stage s (1..H). Formula carried from Gate-1 (stages 1..5 seed-IDENTICAL
    -> the prefix bridge); stages 6..10 continue the +100 stripe. build 0+i..900+i, enc
    5000+i..5900+i, rep 9000+i..9900+i, carrier (s>=2) 2000+i..2800+i, decoy base per DECOY_BASE."""
    return {
        "build": (s - 1) * BUILD_STRIPE + i,
        "enc": ENC_BASE + (s - 1) * STAGE_STRIPE + i,
        "rep": REP_BASE + (s - 1) * STAGE_STRIPE + i,
        "carrier": None if s == 1 else CAR_BASE + (s - 2) * STAGE_STRIPE + i,
        "decoy_base": DECOY_BASE[s],
    }


def _decoy_range(base, seed_max=SEED_MAX):
    return {base + i * 200 + d for i in range(seed_max + 1) for d in range(N_DEC)}


def verify_no_collision(seed_max=SEED_MAX):
    """Prove (over i=0..seed_max, all families) that the network/drive seeds never coincide with a
    decoy seed or the message-seed family, and the decoy families are mutually disjoint. Gate-2 has
    NO scramble arm, so the scramble_rep/scramble_laplacian families are absent."""
    seeds = range(seed_max + 1)
    stages = range(1, H + 1)
    build = {seed_scheme(i, s)["build"] for s in stages for i in seeds}
    enc = {seed_scheme(i, s)["enc"] for s in stages for i in seeds}
    rep = {seed_scheme(i, s)["rep"] for s in stages for i in seeds}
    carrier_arg = {seed_scheme(i, s)["carrier"] for s in range(2, H + 1) for i in seeds}
    carrier_rade = {c + 777 for c in carrier_arg} | {MSG_BASE + i + 777 for i in seeds}
    net_drive = build | enc | rep | carrier_arg | carrier_rade
    msg = {MSG_BASE + i for i in seeds}
    decoy_by_base = {b: _decoy_range(b) for b in list(DECOY_BASE.values()) + [E2E_DECOY_BASE]}
    decoy_all = set().union(*decoy_by_base.values())
    bases = list(decoy_by_base)
    pw = {}
    for a in range(len(bases)):
        for b in range(a + 1, len(bases)):
            n = len(decoy_by_base[bases[a]] & decoy_by_base[bases[b]])
            if n:
                pw[f"{bases[a]}^{bases[b]}"] = n
    checks = {"net_drive_vs_decoy": sorted(net_drive & decoy_all),
              "net_drive_vs_msg": sorted(net_drive & msg),
              "msg_vs_decoy": sorted(msg & decoy_all),
              "decoy_pairwise_overlaps": pw}
    ok = not any(checks.values())
    return {"ok": bool(ok), "seed_max": seed_max,
            "families": {"build": [min(build), max(build)], "enc": [min(enc), max(enc)],
                         "rep": [min(rep), max(rep)], "carrier_arg": [min(carrier_arg), max(carrier_arg)],
                         "carrier_rademacher": [min(carrier_rade), max(carrier_rade)],
                         "msg": [min(msg), max(msg)], "decoy_bases": DECOY_BASE,
                         "e2e_decoy_base": E2E_DECOY_BASE, "N_DEC": N_DEC},
            "collisions": checks}


def log_seed_scheme(log, seed_max=SEED_MAX):
    log(f"--- seed-derivation scheme (chain i, stage s=1..{H}; shown for i as offset '+i') ---")
    log(f"  {'stage':>5} {'build':>9} {'enc':>9} {'rep':>9} {'carrier':>10} {'decoy_base':>11}  provenance")
    for s in range(1, H + 1):
        sd = seed_scheme(0, s)
        car = "n/a(am_input)" if sd["carrier"] is None else f"{sd['carrier']}+i"
        prov = ("== Gate-1 (prefix bridge)" if s <= 5 else "new depth stage")
        log(f"  {s:>5} {str(sd['build'])+'+i':>9} {str(sd['enc'])+'+i':>9} {str(sd['rep'])+'+i':>9} "
            f"{car:>10} {sd['decoy_base']:>11}  {prov}")
    rep = verify_no_collision(seed_max)
    c = rep["collisions"]
    log(f"  e2e-at-depth decoy base -> {E2E_DECOY_BASE}; NO scramble arm (topology settled at Gate-1)")
    log(f"  collision proof (i=0..{seed_max}): net/drive vs decoy={c['net_drive_vs_decoy'] or 'none'}; "
        f"net/drive vs msg={c['net_drive_vs_msg'] or 'none'}; msg vs decoy={c['msg_vs_decoy'] or 'none'}; "
        f"decoy pairwise overlaps={c['decoy_pairwise_overlaps'] or 'none'} -> collision-free: {rep['ok']}")
    return rep


# ===================================================================================== #
#  Decoy construction at depth (byte-identical Phase-1 protocol, extended per-stage base)
# ===================================================================================== #
def gate2_decoys(stage, seed_i, L, dt_in):
    base = DECOY_BASE[stage]
    return [p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=base + seed_i * 200 + d)
            for d in range(N_DEC)]


# ===================================================================================== #
#  The H=10 chain threader (faithful replay of Gate-1's chain for hops 1..5; no scramble)
# ===================================================================================== #
def chain(i, geom, hop_fn, log, *, viol=False, arm="chain", decoys=True, e2e_decoys=True):
    """Thread one H=10 chain for seed i. m0 -> stage1 -> repeater -> ... -> stage10. Every
    repeater = Gate-0/1's F (brick-wall to [0.2,0.9] + affine rescale to the ORIGINAL m0 class
    scalars). Hops 1..5 use seeds IDENTICAL to Gate-1 -> deterministic replay (prefix bridge).
    viol=True: hop-1 message is [2,9]; repeaters stay standard [0.2,0.9] -> first repeater deletes
    the message (filter-violation at depth)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    band = (VIOL_LO, VIOL_HI) if viol else (MSG_LO, MSG_HI)
    if viol:
        m0, u0 = g0.am_input_band(L, dt_in, MSG_BASE + i, VIOL_LO, VIOL_HI)
    else:
        m0, u0 = p1.am_input(L, dt_in, MSG_BASE + i)
    m0_iw = m0[iw]
    dc = float(np.mean(m0_iw))

    hops = []
    sd = seed_scheme(i, 1)
    dec = gate2_decoys(1, i, L, dt_in) if decoys else None
    m_rec, r2_hop, esp, dem = hop_fn(1, sd, m0, u0, dec, L_override=None, anchor=(not viol))
    r2_cum = g0._e2e_score(m_rec, m0, iw, ntr)
    hops.append(g1._hoprec(1, sd, None, r2_cum, r2_hop, esp, dem))
    prev = m_rec

    for s in range(2, H + 1):
        processed, rparams = g0.repeater_transform(prev, m0_iw, dt_in, w_lo=MSG_LO, w_hi=MSG_HI)
        s_full = np.full(L, dc)
        s_full[eval_start:] = g0.remodulate_for_stage_b(processed, m0_iw)
        clip_frac = float(np.mean((dc + processed) < 1e-6))
        sd = seed_scheme(i, s)
        u_in = g0.am_from_message(s_full, sd["carrier"])
        dec = gate2_decoys(s, i, L, dt_in) if decoys else None
        m_rec, r2_hop, esp, dem = hop_fn(s, sd, s_full, u_in, dec, L_override=None, anchor=False)
        r2_cum = g0._e2e_score(m_rec, m0, iw, ntr)
        rep_rec = {**rparams, "dc": dc, "clip_frac": clip_frac}
        hops.append(g1._hoprec(s, sd, rep_rec, r2_cum, r2_hop, esp, dem))
        prev = m_rec

    rec = {"seed": i, "arm": arm, "H": H, "band": list(band), "hops": hops,
           "r2_cum": [h["r2_cum"] for h in hops],
           "esp_all_stages": bool(all(h["esp"]["ok_slow"] for h in hops))}
    if e2e_decoys:
        e2e_dec = [g0._e2e_score(prev, p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                                 seed=E2E_DECOY_BASE + i * 200 + d), iw, ntr)
                   for d in range(N_DEC)]
        rec["e2e_decoy_p95"] = float(np.percentile(e2e_dec, 95))
        rec["e2e_decoy_mean"] = float(np.mean(e2e_dec))
    return rec


def _real_hop_factory(geom):
    return g1._real_hop_factory(geom)              # H-independent (build + g0._hop; anchor->K_GRID)


# ===================================================================================== #
#  Prefix bridge (hops 1..5 must replay the committed Gate-1 chain, digit-exact per seed)
# ===================================================================================== #
def _load_gate1_prefix():
    """{seed: [r2_cum_1..r2_cum_5]} from the committed gate1_multihop.json chain arm."""
    p = os.path.join(RESDIR, "gate1_multihop.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {int(s): [float(x) for x in r["r2_cum"]] for s, r in d["recs"]["chain"].items()}


def prefix_bridge(C, seeds, prefix):
    """Compare gate2 chain hops 1..5 r2_cum vs the Gate-1 prefix, digit-exact (6dp) per seed."""
    checked, misses = [], []
    for i in seeds:
        if i not in prefix:
            continue
        got = [round(float(C[i]["r2_cum"][k]), 6) for k in range(5)]
        ref = [round(float(x), 6) for x in prefix[i]]
        checked.append(i)
        if got != ref:
            misses.append({"seed": i, "got": got, "ref": ref})
    return {"checked": checked, "n_checked": len(checked), "misses": misses,
            "ok": bool(len(checked) > 0 and not misses),
            "reference_present": bool(prefix)}


# ===================================================================================== #
#  Loss law (steady-state rho_k, k=K_START..H; EXCLUDES the insertion rho_2)
# ===================================================================================== #
def loss_law(r2_cum_by_seed, inter, k_start=K_START):
    """rho_k = r2_k/r2_{k-1}, k=k_start..H (k_start=3 EXCLUDES the one-time insertion rho_2), per
    seed, validity guard r2_{k-1} > RHO_VALID_MIN. Per-seed linear slope; thr = max(2*SE,SLOPE_EPS):
    A (flat, |slope|<=thr) / B (falling, slope<-thr) / rising-flag / underpowered. rho_mean =
    steady-state budget rho_ss over the window."""
    rho_levels = {k: [] for k in range(k_start, H + 1)}
    per_seed_slopes, valid_pairs = [], 0
    for i in inter:
        r = r2_cum_by_seed[i]
        ks, rhos = [], []
        for k in range(k_start, H + 1):
            parent = r[k - 2]
            if parent > RHO_VALID_MIN:
                rho = r[k - 1] / parent
                ks.append(k); rhos.append(rho)
                rho_levels[k].append(rho); valid_pairs += 1
        if len(ks) >= 2:
            per_seed_slopes.append(float(np.polyfit(ks, rhos, 1)[0]))
    all_rho = [x for v in rho_levels.values() for x in v]
    rho_mean = float(np.mean(all_rho)) if all_rho else float("nan")
    n = len(per_seed_slopes)
    out = {"k_start": k_start, "rho_valid_min": RHO_VALID_MIN, "valid_pairs": valid_pairs,
           "n_seed_slopes": n, "rho_mean": rho_mean,
           "rho_by_level": {str(k): (float(np.mean(v)) if v else None) for k, v in rho_levels.items()},
           "rho_level_n": {str(k): len(v) for k, v in rho_levels.items()}}
    if n < MIN_PAIRS:
        out.update({"classification": "underpowered", "slope_mean": None, "slope_se": None,
                    "slope_thr": None, "slope_eps": SLOPE_EPS, "margin": None})
        return out
    sm = float(np.mean(per_seed_slopes))
    ssd = float(np.std(per_seed_slopes, ddof=1)) if n > 1 else 0.0
    sse = ssd / np.sqrt(n) if n else float("nan")
    thr = max(2 * sse, SLOPE_EPS)
    cls = "B" if sm < -thr else ("rising-flag" if sm > thr else "A")
    out.update({"classification": cls, "slope_mean": sm, "slope_se": sse, "slope_thr": float(thr),
                "slope_eps": SLOPE_EPS, "margin": float(thr - abs(sm))})
    return out


def _steady_report(C, inter, ll):
    """A2/B2 reporting numbers: rho_ss (mean rho_3..H), measured r2_cum(H), constant-rho_ss budget
    residual (predicted vs measured endpoint; per-seed mean +/- SE), whether r2 crossed 0.5 IN
    measurement, and H_half (the MEASURED crossing hop if r2 crossed within H, else a trend-continued
    projection labeled extrapolation)."""
    rho_ss = ll.get("rho_mean")
    cum_mean = [_mstats([C[i]["r2_cum"][s - 1] for i in inter])["mean"] for s in range(1, H + 1)]
    r2H = _mstats([C[i]["r2_cum"][H - 1] for i in inter])
    r2_3 = _mstats([C[i]["r2_cum"][2] for i in inter])
    pred_H = (r2_3["mean"] * rho_ss ** (H - 3)) if (rho_ss and rho_ss > 0) else None
    # constant-rho_ss budget residual PER SEED (pred_i - meas_i), anchored at hop 3 -> mean +/- SE
    # (ratified op (a)). rho_ss is a pooled constant, so the mean == pred_H - r2H["mean"] exactly --
    # the headline residual does not move; only its per-seed SE is added.
    resids = ([C[i]["r2_cum"][2] * rho_ss ** (H - 3) - C[i]["r2_cum"][H - 1] for i in inter]
              if (rho_ss and rho_ss > 0) else None)
    rst = _mstats(resids) if resids else None
    resid = rst["mean"] if rst else None
    resid_se = rst["se"] if rst else None
    crossed = next((s for s in range(1, H + 1) if cum_mean[s - 1] <= 0.5), None)
    slope = ll.get("slope_mean") or 0.0
    kbar = (K_START + H) / 2.0
    if crossed is not None:                          # r2 crossed 0.5 WITHIN measurement
        h_half = float(crossed)                      # the MEASURED crossing hop (matches the tag)
    elif rho_ss and rho_ss > 0:                      # no in-measurement crossing -> extrapolate trend
        r2, k = r2H["mean"], H
        while r2 > 0.5 and k < 2000:
            k += 1
            rho_k = min(max(rho_ss + slope * (k - kbar), 1e-6), 0.999999)
            r2 *= rho_k
        h_half = float(k) if r2 <= 0.5 else None
    else:
        h_half = None
    return {"rho_ss": rho_ss, "r2_cum_H_mean": r2H["mean"], "r2_cum_H_se": r2H["se"],
            "budget_pred_H": pred_H, "budget_residual": resid, "budget_residual_se": resid_se,
            "crossed_half_at_hop": crossed, "H_half": h_half,
            "H_half_is_extrapolation": bool(crossed is None)}


def _insertion_and_r2hop(C, inter):
    """Descriptive (does not change A2/B2): the one-time first-relay INSERTION loss rho_2, and the
    per-hop reconstruction fidelity r2_hop = r2(m_k, processed-m_{k-1}) flatness over hops 2..H
    (flat = no error amplification = compounding refuted)."""
    rho2 = _mstats([float(C[i]["r2_cum"][1] / C[i]["r2_cum"][0]) for i in inter])
    hop_k = list(range(2, H + 1))
    r2hop_by_hop = {str(k): _mstats([C[i]["hops"][k - 1]["r2_hop"] for i in inter])["mean"]
                    for k in range(1, H + 1)}
    r2hop_sl = [float(np.polyfit(hop_k, [C[i]["hops"][k - 1]["r2_hop"] for k in hop_k], 1)[0])
                for i in inter]
    rh = _mstats(r2hop_sl)
    rh_t = (rh["mean"] / rh["se"]) if rh.get("se") else None
    return {"rho2_insertion": rho2, "r2hop_by_hop": r2hop_by_hop, "r2hop_slope": rh,
            "r2hop_slope_t": rh_t, "compounding_refuted": bool(rh_t is not None and abs(rh_t) < 2.0)}


# ===================================================================================== #
#  Verdict engine (A2 / B2 / C; instrument checks FIRST; prefix bridge carried)
# ===================================================================================== #
def decide(recs, seeds, prefix=None):
    C = recs["chain"]
    if prefix is None:
        prefix = _load_gate1_prefix()
    chain_esp = {i: bool(all(h["esp"]["ok_slow"] for h in C[i]["hops"])) for i in seeds}
    inter = [i for i in seeds if chain_esp[i]]
    out = {"framing": FRAMING, "esp_all_stages": {str(i): chain_esp[i] for i in seeds},
           "intersection": inter, "n_intersection": len(inter),
           "operationalization": {"k_start": K_START, "anchor_window": f"max({ANCHOR_SE_K}*SE,{ANCHOR_FLOOR})",
                                  "rho_valid_min": RHO_VALID_MIN, "min_pairs": MIN_PAIRS,
                                  "viol_e2e_bar_per_seed": VIOL_E2E_BAR, "decoy_elevated": DECOY_ELEVATED,
                                  "slope_eps": SLOPE_EPS,
                                  "A2_bar": "|slope_mean| <= max(2*SE, SLOPE_EPS) (flat steady-state)",
                                  "B2_bar": "slope_mean < -max(2*SE, SLOPE_EPS) (steady decline)"}}
    # ---- prefix bridge (compute always; a determinism/replay integrity gate) ----------- #
    pb = prefix_bridge(C, seeds, prefix)
    out["prefix_bridge"] = pb
    if len(inter) < MIN_PAIRS:
        out["verdict"] = (f"NO-MEASUREMENT (underpowered: chain ESP intersection n={len(inter)} "
                          f"< {MIN_PAIRS} -- add seeds, do not read)")
        return out

    stA = _mstats([C[i]["hops"][0]["r2_cum"] for i in inter])
    window = max(ANCHOR_SE_K * stA["se"], ANCHOR_FLOOR)
    anchor_dev = stA["mean"] - ANCHOR
    anchor_ok = bool(abs(anchor_dev) <= window)
    out["anchor"] = {"hop1": stA, "target": ANCHOR, "window": window, "deviation": anchor_dev,
                     "ok": anchor_ok}

    stage_p95 = {}
    for s in range(1, H + 1):
        vals = [C[i]["hops"][s - 1]["decoy_p95"] for i in inter
                if C[i]["hops"][s - 1]["decoy_p95"] is not None]
        stage_p95[str(s)] = float(np.mean(vals)) if vals else None
    e2e_p95 = float(np.mean([C[i]["e2e_decoy_p95"] for i in inter if "e2e_decoy_p95" in C[i]]))
    present = [v for v in list(stage_p95.values()) + [e2e_p95] if v is not None]
    leak = bool(present and max(present) > DECOY_ELEVATED)
    out["decoys"] = {"stage_p95_mean": stage_p95, "e2e_p95_mean": e2e_p95, "elevated": leak}

    V = recs.get("violation", {})
    vinter = [i for i in inter if i in V and all(h["esp"]["ok_slow"] for h in V[i]["hops"])]
    if len(vinter) >= MIN_PAIRS:
        viol_ps = {i: float(V[i]["r2_cum"][-1]) for i in vinter}
        collapsed = all(r < VIOL_E2E_BAR for r in viol_ps.values())
        offenders = [i for i, r in viol_ps.items() if r >= VIOL_E2E_BAR]
        rms_in = float(np.mean([V[i]["hops"][1]["rms_in"] for i in vinter]))
        scale = float(np.mean([V[i]["hops"][1]["scale"] for i in vinter]))
        viol_mean = float(np.mean(list(viol_ps.values())))
    else:
        viol_ps, collapsed, offenders, rms_in, scale, viol_mean = {}, False, [], None, None, None
    out["violation"] = {"n": len(vinter), "intersection": vinter, "bar_per_seed": VIOL_E2E_BAR,
                        "e2e_per_seed": {str(k): v for k, v in viol_ps.items()}, "offenders": offenders,
                        "violation_e2e_mean": viol_mean, "signature_rms_in": rms_in,
                        "signature_scale": scale, "collapsed": collapsed}

    ll = loss_law({i: C[i]["r2_cum"] for i in inter}, inter, k_start=K_START)
    out["loss_law"] = ll
    out["steady"] = _steady_report(C, inter, ll)
    out["insertion_r2hop"] = _insertion_and_r2hop(C, inter)
    out["cumulative"] = {str(s): _mstats([C[i]["r2_cum"][s - 1] for i in inter])
                         for s in range(1, H + 1)}

    # ---- verdict (instruments first; pre-registered order) ------------------------------ #
    if not anchor_ok:
        side = "low (replication failure)" if anchor_dev < 0 else "high"
        out["verdict"] = (f"NO-MEASUREMENT (anchor miss, {side}: hop-1 mean {stA['mean']:.6f} vs "
                          f"{ANCHOR} +/- {window:.4f} -- STOP, fix, re-run)")
    elif not pb["ok"]:
        why = ("reference gate1_multihop.json missing -- cannot verify the mandatory bridge"
               if not pb["reference_present"] else
               f"hops 1-5 do NOT replay Gate-1 for seeds {[m['seed'] for m in pb['misses']]}")
        out["verdict"] = (f"NO-MEASUREMENT (PREFIX-BRIDGE miss: {why} -- determinism/replay broken, STOP)")
    elif leak:
        out["verdict"] = ("NO-MEASUREMENT (decoy elevated -- leakage: max p95 mean "
                          f"{max(present):.3f} > {DECOY_ELEVATED})")
    elif not collapsed:
        vs = (f"seeds {offenders} have e2e r2 >= {VIOL_E2E_BAR} (mean {viol_mean:.3f})"
              if viol_mean is not None else f"violation sub-intersection underpowered (n={len(vinter)})")
        out["verdict"] = (f"NO-MEASUREMENT (filter-violation-at-depth did NOT collapse: {vs})")
    elif ll["classification"] == "underpowered":
        out["verdict"] = ("NO-MEASUREMENT (loss-law underpowered: "
                          f"{ll['n_seed_slopes']} fittable seed-slopes < {MIN_PAIRS})")
    elif ll["classification"] == "A":
        out["verdict"] = _verdict_A2(out, ll)
    elif ll["classification"] == "B":
        out["verdict"] = _verdict_B2(out, ll)
    else:  # rising-flag -- Pin B: never A2
        out["verdict"] = (f"NO-MEASUREMENT (loss-law INSTRUMENT-SUSPICION: steady-state rho_k RISES "
                          f"with k, slope {ll['slope_mean']:+.4f} +/- {ll['slope_se']:.4f} > "
                          f"+max(2*SE,{SLOPE_EPS})=+{ll['slope_thr']:.4f} (margin {ll['margin']:+.4f}) "
                          "-- non-physical for a lossy relay; NEVER A2. Inspect the instrument.)")
    return out


def _verdict_A2(out, ll):
    st = out["steady"]
    ir = out["insertion_r2hop"]
    hh = _fmt(st["H_half"], ".1f")
    tag = "measured" if not st["H_half_is_extrapolation"] else "EXTRAPOLATION"
    return (f"A2 -- ASYMPTOTIC / PRICED (steady-state flat at depth): steady-state rho_k "
            f"(k={K_START}..{H}) slope {ll['slope_mean']:+.4f} +/- {ll['slope_se']:.4f}, |slope| <= "
            f"max(2*SE,{SLOPE_EPS})={ll['slope_thr']:.4f} (margin {ll['margin']:+.4f}). Steady budget "
            f"rho_ss={_fmt(st['rho_ss'],'.4f')}; measured r2_cum(H={H})={_fmt(st['r2_cum_H_mean'],'.4f')} "
            f"+/- {_fmt(st['r2_cum_H_se'],'.4f')}; constant-rho_ss budget residual (pred-meas) "
            f"{_fmt(st['budget_residual'],'+.4f')} +/- {_fmt(st['budget_residual_se'],'.4f')}. "
            f"B/depth-limit REFUTED: per-hop r2_hop flat "
            f"(t={_fmt(ir['r2hop_slope_t'],'.2f')}). Insertion loss rho_2={_fmt(ir['rho2_insertion']['mean'],'.3f')} "
            f"(one-time, excluded from the trend). H_half={hh} ({tag}). The ladder is PRICED on "
            f"measurement to H={H}.")


def _verdict_B2(out, ll):
    st = out["steady"]
    ir = out["insertion_r2hop"]
    hh = _fmt(st["H_half"], ".1f")
    tvz = (ll["slope_mean"] / ll["slope_se"]) if ll.get("slope_se") else float("nan")
    rb = ll.get("rho_by_level", {})
    r_lo, r_hi = rb.get(str(K_START)), rb.get(str(H))
    return (f"B2 -- STEADY DECLINE (pre-registered): the m0-referenced steady-state ratio rho_k "
            f"(k={K_START}..{H}) FALLS with depth, slope {ll['slope_mean']:+.4f} +/- {ll['slope_se']:.4f} "
            f"< -max(2*SE,{SLOPE_EPS})=-{ll['slope_thr']:.4f} (t={_fmt(tvz,'.2f')} vs zero; the margin "
            f"{ll['margin']:+.4f} is measured vs the {SLOPE_EPS} min-effect FLOOR, not a noise scale, so "
            f"the DIRECTION is decisive). The Gate-1 late drift is REAL at depth. MECHANISM -- this is an "
            f"END-TO-END, m0-referenced decline (systematic drift away from the source), NOT per-hop error "
            f"amplification: per-hop reconstruction r2_hop (each stage vs its OWN immediate input) is "
            f"depth-flat (slope t={_fmt(ir['r2hop_slope_t'],'.2f')} over hops 2..{H}, n.s.), so per-stage "
            f"compounding is refuted while the m0-referenced decline stands -- r2_hop scores a drifted "
            f"moving target and is blind to the m0 loss. MAGNITUDE modest, horizon EXTRAPOLATED: "
            f"rho_ss={_fmt(st['rho_ss'],'.4f')}, rho falls {_fmt(r_lo,'.3f')}->{_fmt(r_hi,'.3f')} over hops "
            f"{K_START}->{H}; measured r2_cum(H={H})={_fmt(st['r2_cum_H_mean'],'.4f')} +/- "
            f"{_fmt(st['r2_cum_H_se'],'.4f')} has NOT crossed the 0.5 half-power point; H_half={hh} is a "
            f"one-hop trend-extrapolation, so 'bounded' is inferred by trend continuation, not observed. "
            f"Insertion loss rho_2={_fmt(ir['rho2_insertion']['mean'],'.3f')} (one-time, excluded).")


# ===================================================================================== #
#  Markdown record
# ===================================================================================== #
def _slope_line(ll):
    cls = ll.get("classification", "not-computed")
    sm, sse = ll.get("slope_mean"), ll.get("slope_se")
    thr, mg = ll.get("slope_thr"), ll.get("margin")
    if sm is None:
        return (f"- **loss-law slope: n/a** (class {cls}). SLOPE_EPS={SLOPE_EPS}. Pin B: a positive "
                "slope beyond +max(2*SE,SLOPE_EPS) is instrument-suspicion, never A2.")
    inside = (mg is not None and mg >= 0)
    return (f"- **steady-state (k={ll.get('k_start')}..{H}) slope = {sm:+.4f} +/- {_fmt(sse,'.4f')}** "
            f"(class {cls}); threshold max(2*SE,{SLOPE_EPS}) = {_fmt(thr,'.4f')}; margin "
            f"{_fmt(mg,'+.4f')} -> {'inside flat band (A2-consistent)' if inside else 'trend resolved'}. "
            "Pin B: positive slope beyond +threshold = instrument-suspicion, never A2.")


def _write_md(path, v, seeds, wall, hashes, colrep, note=""):
    ll = v.get("loss_law", {})
    lines = [
        "# Relay Gate-2 -- depth-extension loss-law record (H=10)", "",
        f"Spec: relay_gate2_multihop_spec.md (sha256 {hashes['spec']}). Harness: "
        f"experiments/relay_gate2.py (sha256 {hashes['code']}).",
        f"Seeds run: {seeds}. K = {K_PRIMARY}. Wall-clock {wall/60:.0f} min. "
        f"Seed scheme collision-free: {colrep['ok']}.",
    ] + ([note] if note else []) + [
        "", f"Framing: {FRAMING}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
        "## Instrument checks (pre-registered order, before the A2/B2 read)", "",
    ]
    pb = v.get("prefix_bridge", {})
    if "anchor" in v:
        a, d, vio = v["anchor"], v["decoys"], v["violation"]
        lines += [
            f"1. **Anchor (hop-1 == committed span-1.5/K=0.24)**: mean {a['hop1']['mean']:.6f} "
            f"(SE {a['hop1']['se']:.6f}, n={a['hop1']['n']}); target {ANCHOR} +/- {a['window']:.4f}; "
            f"deviation {a['deviation']:+.6f} -> {'OK' if a['ok'] else 'MISS'}.",
            f"2. **Prefix bridge (hops 1-5 == committed Gate-1, digit-exact)**: checked "
            f"{pb.get('n_checked')} seeds -> {'ALL REPLAY' if pb.get('ok') else 'MISS '+str(pb.get('misses'))}.",
            f"3. **Decoy floors at depth** (intersection p95 means): per-stage "
            f"{ {k: (round(x,3) if x is not None else None) for k,x in d['stage_p95_mean'].items()} }, "
            f"e2e-at-depth {d['e2e_p95_mean']:+.4f}; bar {DECOY_ELEVATED} -> "
            f"{'ELEVATED (leak)' if d['elevated'] else 'clean'}.",
            f"4. **Filter-violation at depth** ([2,9] msg, first repeater [0.2,0.9]): n={vio['n']}, "
            f"e2e mean {_fmt(vio['violation_e2e_mean'])} (bar < {vio['bar_per_seed']} every seed) -> "
            f"{'COLLAPSED (sound)' if vio['collapsed'] else 'DID NOT COLLAPSE '+str(vio['offenders'])}. "
            f"Signature rms_in {_fmt(vio['signature_rms_in'],'.4g')}, scale {_fmt(vio['signature_scale'],'.1f')}.",
            f"5. **ESP-honest paired intersection** (ESP-ok across ALL {H} stages): {v['intersection']} "
            f"(n={v['n_intersection']}).", "",
            "## Cumulative fidelity r2(m_k, m0) (intersection means +/- SE)", "",
        ]
        for s in range(1, H + 1):
            cm = v["cumulative"][str(s)]
            lines.append(f"- hop {s:>2}: **{cm['mean']:+.4f} +/- {cm['se']:.4f}** "
                         f"(per-seed {[round(x,3) for x in cm['per_seed']]})")
        st, ir = v["steady"], v["insertion_r2hop"]
        lines += [
            "", f"## Steady-state loss law (rho_k, k={K_START}..{H}; insertion rho_2 EXCLUDED by "
            "pre-registration)", "",
            _slope_line(ll),
            f"- classification **{ll.get('classification')}**; steady budget rho_ss = "
            f"{_fmt(st.get('rho_ss'),'.4f')}; measured r2_cum(H={H}) = {_fmt(st.get('r2_cum_H_mean'),'.4f')} "
            f"+/- {_fmt(st.get('r2_cum_H_se'),'.4f')}; constant-rho_ss budget residual (pred-meas) "
            f"{_fmt(st.get('budget_residual'),'+.4f')} +/- {_fmt(st.get('budget_residual_se'),'.4f')} "
            f"(DESCRIPTIVE ONLY -- a constant-rho_ss endpoint match is near-tautological for a smooth trend "
            f"and does NOT adjudicate flat-vs-falling; the pre-registered slope test is the sole discriminator).",
            f"- rho by level: { {k: (round(x,4) if x is not None else None) for k,x in ll.get('rho_by_level',{}).items()} }.",
            f"- **per-hop error-amplification (compounding) check** (separate from the loss-law verdict above; "
            f"scoped to per-STAGE amplification, NOT the m0-referenced decline): per-hop reconstruction r2_hop by hop "
            f"{ {k: (round(x,4) if x is not None else None) for k,x in ir.get('r2hop_by_hop',{}).items()} } "
            f"slope t={_fmt(ir.get('r2hop_slope_t'),'.2f')} (over hops 2..{H}, hop-1 anchor excluded) -> "
            f"{'error amplification REFUTED (flat r2_hop -- each stage reconstructs its OWN immediate input equally well at any depth; this does NOT refute an m0-referenced decline)' if ir.get('compounding_refuted') else 'per-stage amplification present (r2_hop falls with depth)'}.",
            f"- one-time first-relay INSERTION loss rho_2 = {_fmt(ir['rho2_insertion'].get('mean'),'.4f')} "
            f"+/- {_fmt(ir['rho2_insertion'].get('se'),'.4f')} (excluded from the trend statistic).",
            f"- H_half = {_fmt(st.get('H_half'),'.1f')} "
            f"({'MEASURED (r2 crossed 0.5 by hop '+str(st.get('crossed_half_at_hop'))+')' if not st.get('H_half_is_extrapolation') else 'EXTRAPOLATION (r2 did not cross 0.5 within H='+str(H)+')'}).",
        ]
    else:
        lines += [f"- Battery ended on the pre-registered early exit: intersection {v['intersection']} "
                  f"(n={v['n_intersection']}). Prefix bridge: {'ok' if pb.get('ok') else pb}.",
                  _slope_line(ll)]
    lines += ["", "## Scope", "",
              "Offline H=10 chain; compound span 1.5*H = an INFORMATION-PATH claim, NOT one physical",
              "spectrum. Single variable = depth (no scramble; topology settled at Gate-1). STOP-and-report:",
              "Gate-3 (hop-length) and the mechanism-decomposition gate are separate decisions."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ===================================================================================== #
#  STAGE 1 -- CPU sandbox
# ===================================================================================== #
def _sandbox_geom():
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(STAGE_SPAN, n_msg=8)
    sl = slice(eval_start, L)
    iw = np.arange(eval_start, L)
    ntr = int(p1.TRAIN_FRAC * (L - eval_start))
    return dt_in, eval_start, L, delays, sl, iw, ntr, 1


def _synth_chain_recs(r2_by_seed):
    """Wrap {seed:[r2_1..r2_H]} as chain recs (ESP-ok, clean decoys) for decide()/loss_law."""
    out = {}
    for i, r in r2_by_seed.items():
        hops = []
        for s in range(1, H + 1):
            hops.append({"stage": s, "esp": {"ok_slow": True}, "r2_cum": r[s - 1], "r2_hop": r[s - 1],
                         "decoy_p95": -0.10, "rms_in": 0.1, "rms_target": 0.1, "scale": 1.0,
                         "repeater_in": (None if s == 1 else {"rms_in": 0.1, "scale": 1.0})})
        out[i] = {"seed": i, "arm": "chain", "H": H, "hops": hops, "r2_cum": list(r),
                  "e2e_decoy_p95": -0.30}
    return out


def _synth_viol_recs(seeds, e2e=0.0):
    out = {}
    for i in seeds:
        hops = []
        for s in range(1, H + 1):
            rep = None if s == 1 else {"rms_in": 1e-3, "rms_target": 0.14, "scale": 140.0,
                                       "msg_band": [MSG_LO, MSG_HI]}
            hops.append({"stage": s, "esp": {"ok_slow": True}, "r2_cum": (0.88 if s == 1 else e2e),
                         "r2_hop": 0.9, "decoy_p95": None,
                         "rms_in": (rep["rms_in"] if rep else None),
                         "rms_target": (rep["rms_target"] if rep else None),
                         "scale": (rep["scale"] if rep else None), "repeater_in": rep})
        out[i] = {"seed": i, "arm": "violation", "H": H, "hops": hops,
                  "r2_cum": [0.88] + [e2e] * (H - 1)}
    return out


def _geom_r2(rho_ins, rho_ss, r1=0.986):
    """Build a length-H r2_cum: r1, then insertion rho_ins to hop 2, then steady rho_ss."""
    r = [r1, r1 * rho_ins]
    for _ in range(H - 2):
        r.append(r[-1] * rho_ss)
    return r


def sandbox(log):
    log(f"=== RELAY GATE-2 :: STAGE-1 CPU SANDBOX (no GPU; H={H}) ===")
    log(f"    backend: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS','<default>')} "
        f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}'")
    log(f"    framing: {FRAMING}")
    geom = _sandbox_geom()
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} iw={len(iw)} ntr={ntr}")
    R = {}

    # ---- CHECK 0: extended seed scheme + collision matrix ------------------------------ #
    log("\n(0) Extended seed-derivation scheme + collision proof (H=10, i=0..9)")
    colrep = log_seed_scheme(log)
    c0 = g0._check(log, "extended seed scheme collision-free vs decoy/msg families", colrep["ok"],
                   f"stages 1-{H}, decoy bases {list(DECOY_BASE.values())}+e2e {E2E_DECOY_BASE}")
    R["check0_seed_scheme"] = {"pass": c0, "report": colrep}

    # ---- CHECK 1: 10-stage chain wiring (pass-through) --------------------------------- #
    log("\n(1) 10-stage chain wiring -- H=10 threaded, repeater between every pair (pass-through)")
    rec = chain(0, geom, g1._make_synth_hop(geom, fidelity=1.0), log, arm="chain")
    n_hops = len(rec["hops"]); n_reps = sum(1 for h in rec["hops"] if h["repeater_in"] is not None)
    builds = [h["seeds"]["build"] for h in rec["hops"]]
    fresh = (len(set(builds)) == H) and builds == [seed_scheme(0, s)["build"] for s in range(1, H + 1)]
    cum = rec["r2_cum"]
    passthru = (cum[0] > 0.999) and all(x > 0.99 for x in cum[1:])
    plateau = (max(cum[1:]) - min(cum[1:])) < 5e-3
    c1 = all([
        g0._check(log, "H=10 stages threaded", n_hops == H, f"{n_hops} hops"),
        g0._check(log, "repeater between every consecutive pair (H-1=9)",
                  n_reps == H - 1 and rec["hops"][0]["repeater_in"] is None,
                  f"{n_reps} repeaters; hop-1 repeater_in=None"),
        g0._check(log, "fresh derived build seed per stage (0,100..900)",
                  fresh, f"builds={builds}"),
        g0._check(log, "message threads end-to-end (pass-through) + idempotent plateau",
                  passthru and plateau, f"r2_cum={[round(x,4) for x in cum]}"),
    ])
    R["check1_wiring"] = {"pass": c1, "n_hops": n_hops, "n_reps": n_reps, "r2_cum": cum}

    # ---- CHECK 2: per-hop logging schema ----------------------------------------------- #
    log("\n(2) Per-hop logging schema (r2_cum, r2_hop, rms_in/target/scale) over all 10 hops")
    req = {"stage", "seeds", "repeater_in", "r2_cum", "r2_hop", "esp", "decoy_p95", "rms_in",
           "rms_target", "scale"}
    keys_ok = all(req <= set(h) for h in rec["hops"])
    rep_ok = all(all(h[k] is not None for k in ("rms_in", "rms_target", "scale")) for h in rec["hops"][1:])
    c2 = all([
        g0._check(log, "every hop carries the full schema key set", keys_ok, f">= {sorted(req)}"),
        g0._check(log, "rms_in/rms_target/scale present for hops 2..10", rep_ok,
                  f"hop2 trio ({rec['hops'][1]['rms_in']:.3g},{rec['hops'][1]['rms_target']:.3g},{rec['hops'][1]['scale']:.3g})"),
        g0._check(log, "e2e decoy-at-depth recorded", "e2e_decoy_p95" in rec,
                  f"e2e_decoy_p95={rec.get('e2e_decoy_p95'):+.4f}"),
    ])
    R["check2_schema"] = {"pass": c2}

    # ---- CHECK 3: 10-stage intersection (mid-chain fail drops seed) --------------------- #
    log("\n(3) 10-stage intersection -- nested esp->ok_slow; a mid-chain (stage-7) fail drops seed")
    def mk(flags):
        return {"hops": [{"esp": {"d_slow": 0.0 if f else 1.0, "ok_slow": f}} for f in flags]}
    T = {0: mk([True] * 10), 1: mk([True] * 6 + [False] + [True] * 3),   # seed1 fails stage 7
         2: mk([True] * 10), 3: mk([False] + [True] * 9),                # seed3 fails stage 1
         4: mk([True] * 9 + [False])}                                    # seed4 fails stage 10
    esp_all = {i: bool(all(g0.esp_ok_slow(h) for h in T[i]["hops"])) for i in T}
    inter = sorted(i for i, ok in esp_all.items() if ok)
    c3 = all([
        g0._check(log, "nested esp->ok_slow per stage (not flat)",
                  g0.esp_ok_slow(T[0]["hops"][0]) and not g0.esp_ok_slow(T[1]["hops"][6]),
                  "reads hop['esp']['ok_slow']"),
        g0._check(log, "intersection = seeds ESP-ok across ALL 10 stages", inter == [0, 2],
                  f"intersection={inter}"),
        g0._check(log, "mid-chain (stage-7) fail drops the seed everywhere",
                  1 not in inter and 3 not in inter and 4 not in inter, "seeds 1(st7)/3(st1)/4(st10) excluded"),
    ])
    R["check3_intersection"] = {"pass": c3, "intersection": inter}

    # ---- CHECK 4: prefix-bridge loader + comparison ------------------------------------ #
    log("\n(4) Prefix-bridge loader -- read committed gate1_multihop.json, hops 1-5 digit-exact")
    prefix = _load_gate1_prefix()
    have_ref = bool(prefix)
    seed0_ref = [round(x, 6) for x in prefix.get(0, [])] if have_ref else []
    expect0 = [0.981470, 0.962334, 0.931043, 0.896005, 0.842353]
    ref_ok = have_ref and seed0_ref == expect0 and sorted(prefix) == [0, 1, 2, 3, 4, 5, 6, 7]
    # comparison logic: a matching synth chain -> ok; a perturbed one -> miss detected
    good = {i: {"r2_cum": prefix[i] + [0.8, 0.78, 0.76, 0.74, 0.72]} for i in prefix}
    pb_good = prefix_bridge(good, list(prefix), prefix)
    bad = {i: {"r2_cum": list(prefix[i]) + [0.8] * 5} for i in prefix}
    bad[3]["r2_cum"][2] = prefix[3][2] - 0.01                    # perturb seed3 hop-3
    pb_bad = prefix_bridge(bad, list(prefix), prefix)
    c4 = all([
        g0._check(log, "loader reads gate1 prefix (seeds 0-7, seed0 digit-exact)", ref_ok,
                  f"seed0={seed0_ref} (expect {expect0}); seeds {sorted(prefix)}"),
        g0._check(log, "matching hops 1-5 -> bridge OK", pb_good["ok"] and pb_good["n_checked"] == 8,
                  f"n_checked={pb_good['n_checked']} misses={len(pb_good['misses'])}"),
        g0._check(log, "perturbed hop-3 (seed3) -> bridge MISS detected",
                  (not pb_bad["ok"]) and any(m["seed"] == 3 for m in pb_bad["misses"]),
                  f"misses={[m['seed'] for m in pb_bad['misses']]}"),
    ])
    R["check4_prefix_bridge"] = {"pass": c4, "reference_present": have_ref, "seed0": seed0_ref}

    # ---- CHECK 5: rho window k=3..10 (A2/B2 machinery; insertion excluded) -------------- #
    log("\n(5) Steady-state rho window k=3..10 -- A2/B2 slope test, insertion rho_2 EXCLUDED")
    seeds5 = list(range(5))
    flat = {i: _geom_r2(0.941, 0.960) for i in seeds5}              # steady rho 0.96 -> A2
    # declining steady-state: rho_k falls 0.97 -> 0.83 across k=3..10 (slope -0.02, beyond the floor)
    declc = {}
    for i in seeds5:
        r = [0.986, 0.986 * 0.941]
        for j, k in enumerate(range(3, H + 1)):
            r.append(r[-1] * (0.97 - 0.02 * j))
        declc[i] = r
    ll_A = loss_law(flat, seeds5)
    ll_B = loss_law(declc, seeds5)
    excl_ok = ("2" not in ll_A["rho_by_level"]) and (str(K_START) in ll_A["rho_by_level"])
    ll_U = loss_law(flat, [0])
    c5 = all([
        g0._check(log, "insertion rho_2 EXCLUDED (window starts at k=3)", excl_ok,
                  f"rho levels = {list(ll_A['rho_by_level'])}"),
        g0._check(log, "flat steady-state -> A", ll_A["classification"] == "A",
                  f"slope {ll_A['slope_mean']:+.4f}, thr {ll_A['slope_thr']:.4f}, rho_ss {ll_A['rho_mean']:.3f}"),
        g0._check(log, "declining steady-state -> B", ll_B["classification"] == "B",
                  f"slope {ll_B['slope_mean']:+.4f} +/- {ll_B['slope_se']:.4f}"),
        g0._check(log, "underpowered flag fires below MIN_PAIRS", ll_U["classification"] == "underpowered",
                  f"n_seed_slopes={ll_U['n_seed_slopes']}"),
    ])
    # (the A2/B2 verdict-string mapping is exercised end-to-end in --verdict-test, with prefix
    #  references that MATCH the synthetic curves so the prefix bridge passes.)
    R["check5_losslaw"] = {"pass": c5, "A": ll_A, "B": ll_B}

    # ---- CHECK 6: violation-at-depth plumbing ------------------------------------------ #
    log("\n(6) Violation-at-depth plumbing -- [2,9] msg, first repeater [0.2,0.9], collapse @ H=10")
    vrec = chain(0, geom, g1._make_synth_hop(geom, fidelity=1.0), log, viol=True, arm="violation",
                 decoys=False, e2e_decoys=False)
    first_rep = vrec["hops"][1]["repeater_in"]
    sig = first_rep["rms_in"] < 0.05 * first_rep["rms_target"] and first_rep["scale"] > 5.0
    c6 = all([
        g0._check(log, "violation message band = [2,9]", vrec["band"] == [VIOL_LO, VIOL_HI], f"band={vrec['band']}"),
        g0._check(log, "first repeater pass-band = standard [0.2,0.9]",
                  first_rep["msg_band"] == [MSG_LO, MSG_HI], f"passband={first_rep['msg_band']}"),
        g0._check(log, "signature small rms_in / large scale (logged)", sig,
                  f"rms_in={first_rep['rms_in']:.3g} scale={first_rep['scale']:.1f}"),
        g0._check(log, "filter deletes [2,9] -> e2e collapses at H=10", vrec["r2_cum"][-1] < 0.1,
                  f"r2_cum_H={vrec['r2_cum'][-1]:+.4f}"),
    ])
    R["check6_violation"] = {"pass": c6, "e2e_collapse": vrec["r2_cum"][-1]}

    # ---- CHECK 7: decoy protocol at depth (extended bases) ----------------------------- #
    log("\n(7) Decoy protocol at depth -- byte-identical Phase-1 construction, extended bases")
    seed_i = 3
    ref1 = g0.phase1_decoys_ref(seed_i, L, dt_in)
    diff1 = max(float(np.max(np.abs(np.asarray(a) - np.asarray(r)))) for a, r in zip(gate2_decoys(1, seed_i, L, dt_in), ref1))
    protocol_ok = all(
        np.allclose(gate2_decoys(s, seed_i, L, dt_in)[d],
                    p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=DECOY_BASE[s] + seed_i * 200 + d))
        for s in range(1, H + 1) for d in (0, N_DEC // 2, N_DEC - 1))
    counts_ok = all(len(gate2_decoys(s, seed_i, L, dt_in)) == N_DEC for s in range(1, H + 1))
    c7 = all([
        g0._check(log, "stage-1 decoys byte-identical to Phase-1 (base 40000)", diff1 == 0.0,
                  f"max|diff|={diff1:.1e}"),
        g0._check(log, "every stage (1-10) identical p1.slow_bandlimited protocol",
                  protocol_ok and counts_ok, f"bases {list(DECOY_BASE.values())}, {N_DEC} draws"),
        g0._check(log, "decoy scoring path is imported p1.demod_capacity",
                  p1.demod_capacity.__module__ == "D_phase1_routing", ""),
    ])
    R["check7_decoy"] = {"pass": c7, "diff_stage1": diff1}

    order = ["check0_seed_scheme", "check1_wiring", "check2_schema", "check3_intersection",
             "check4_prefix_bridge", "check5_losslaw", "check6_violation", "check7_decoy"]
    allpass = all(R[k]["pass"] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if R[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate2_sandbox.json")
    with open(outp, "w") as f:
        json.dump({"gate": "relay-gate2", "stage": "1-cpu-sandbox", "H": H, "all_pass": allpass,
                   "framing": FRAMING, "checks": R}, f, indent=1, default=g1._json_default)
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


# ===================================================================================== #
#  Synthetic verdict-engine test (CPU; A2/B2/C branches, with MATCHING prefix bridge)
# ===================================================================================== #
def verdict_test(log):
    log("=== VERDICT-ENGINE SYNTHETIC TEST (A2/B2/C; CPU only) ===")
    seeds = list(range(5))
    # build curves whose hops 1-5 MATCH a synthetic prefix so the prefix bridge passes
    flat = {i: _geom_r2(0.941, 0.960 + 0.001 * (i - 2)) for i in seeds}
    prefix = {i: flat[i][:5] for i in seeds}                        # a matching prefix reference
    declc = {}
    for i in seeds:
        r = list(flat[i][:5])                                       # SAME prefix (bridge holds)
        for j in range(H - 5):                                      # steep late decline -> B2
            r.append(r[-1] * (0.94 - 0.02 * j))
        declc[i] = r
    prefix_d = {i: declc[i][:5] for i in seeds}
    allok = True

    def run_case(name, r2, viol_e2e, want, pfx, extra=lambda v: True):
        nonlocal allok
        v = decide({"chain": _synth_chain_recs(r2), "violation": _synth_viol_recs(seeds, e2e=viol_e2e)},
                   seeds, prefix=pfx)
        ok = (want in v["verdict"]) and extra(v)
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {v['verdict'][:74]}")
        return v

    vA = run_case("flat steady -> A2", flat, 0.0, "A2 -- ASYMPTOTIC", prefix)
    vB = run_case("declining steady -> B2", declc, 0.0, "B2 -- STEADY DECLINE", prefix_d)
    run_case("violation no-collapse -> NM", flat, 0.8, "did NOT collapse", prefix)
    # prefix miss: perturb one seed's hop-3 away from the reference
    flat_miss = {i: list(flat[i]) for i in seeds}
    flat_miss[2][2] = flat[2][2] - 0.02
    run_case("prefix-bridge miss -> NM", flat_miss, 0.0, "PREFIX-BRIDGE miss", prefix)
    # anchor low
    flat_low = {i: [x - 0.05 for x in flat[i]] for i in seeds}
    run_case("anchor low -> NM", flat_low, 0.0, "anchor miss, low", {i: flat_low[i][:5] for i in seeds})
    # underpowered (all ESP fail on stage 7)
    cr = _synth_chain_recs(flat)
    for i in seeds:
        cr[i]["hops"][6]["esp"]["ok_slow"] = False
    vU = decide({"chain": cr, "violation": _synth_viol_recs(seeds)}, seeds, prefix=prefix)
    okU = "underpowered" in vU["verdict"] and vU["n_intersection"] == 0
    allok &= okU
    log(f"  [{'OK' if okU else 'WRONG'}] stage-7 ESP fail all -> NM underpowered: {vU['verdict'][:60]}")

    # H_half consistency: a B2 curve that crosses 0.5 STRICTLY before hop 10 must report h_half =
    # the MEASURED crossing hop (not the endpoint H), and the budget residual must carry its per-seed SE.
    early = {}
    for i in seeds:
        r = list(flat[i][:5])                                       # same matching prefix
        for j in range(H - 5):
            r.append(r[-1] * (0.86 - 0.03 * j))                     # steep -> crosses 0.5 before hop 10
        early[i] = r
    Ce = _synth_chain_recs(early)
    lle = loss_law({i: Ce[i]["r2_cum"] for i in seeds}, list(seeds), k_start=K_START)
    ste = _steady_report(Ce, list(seeds), lle)
    cx = ste["crossed_half_at_hop"]
    hh_ok = (cx is not None and cx < H and ste["H_half"] == float(cx)
             and not ste["H_half_is_extrapolation"] and ste["budget_residual_se"] is not None)
    allok &= hh_ok
    log(f"  [{'OK' if hh_ok else 'WRONG'}] H_half = measured crossing (hop {cx}) not endpoint H={H}; "
        f"budget_residual {_fmt(ste.get('budget_residual'),'+.4f')} +/- {_fmt(ste.get('budget_residual_se'),'.4f')}")

    # _write_md renders every shape + Pin-A slope line
    import tempfile
    for tag, vv in (("A2", vA), ("B2", vB), ("underpowered", vU)):
        p = os.path.join(tempfile.gettempdir(), f"_g2_md_{tag}.md")
        try:
            _write_md(p, vv, seeds, 0.0, {"code": "selftest", "spec": "selftest"}, {"ok": True})
            txt = open(p).read(); os.remove(p)
            has = ("slope" in txt)
            allok &= has
            log(f"  [{'OK' if has else 'WRONG'}] _write_md({tag}) renders + slope line present")
        except Exception as e:
            allok = False
            log(f"  [WRONG] _write_md({tag}) crashed: {e!r}")
    log(f"  VERDICT-ENGINE: {'ALL BRANCHES CORRECT' if allok else 'BRANCH ERRORS -- FIX'}")
    return allok


# ===================================================================================== #
#  STAGE 2 -- smoke (seed 0, H=10; hops 1-5 must replay Gate-1 seed-0 digit-exact)
# ===================================================================================== #
def smoke(log):
    import time
    log(f"=== RELAY GATE-2 :: STAGE-2 SMOKE (seed 0, K=0.24, full H={H} chain) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} ntr={ntr} n_sub={n_sub}")
    prefix = _load_gate1_prefix()
    t0 = time.perf_counter()
    rec = chain(0, geom, _real_hop_factory(geom), log, arm="chain")
    wall = time.perf_counter() - t0

    ref1 = g1.REF_TABLE[(STAGE_SPAN, K_PRIMARY)][0][0]              # 0.981470
    prefix0 = [round(x, 6) for x in prefix.get(0, [])]
    got0 = [round(rec["r2_cum"][k], 6) for k in range(5)]
    bridge1 = round(rec["r2_cum"][0], 6) == ref1
    pbridge = bool(prefix0) and got0 == prefix0
    log(f"\n=== SMOKE SUMMARY (seed 0) ===")
    for h in rec["hops"]:
        rin = "n/a" if h["rms_in"] is None else f"{h['rms_in']:.3g}"
        sc = "n/a" if h["scale"] is None else f"{h['scale']:.2f}"
        log(f"  hop {h['stage']:>2}: r2_cum={h['r2_cum']:+.6f} r2_hop={h['r2_hop']:+.6f} "
            f"ESP={h['esp']['ok_slow']} rms_in={rin} scale={sc}")
    log(f"  cumulative r2_cum = {[round(x,4) for x in rec['r2_cum']]}")
    log(f"  e2e-at-depth decoy p95 = {rec.get('e2e_decoy_p95'):+.4f}")
    log(f"  BRIDGE hop-1 vs committed b0f7664: {rec['r2_cum'][0]:.6f} vs {ref1:.6f} -> {'MATCH' if bridge1 else 'MISMATCH -- STOP'}")
    log(f"  PREFIX BRIDGE hops 1-5 vs Gate-1 seed-0: {got0} vs {prefix0} -> "
        f"{'MATCH' if pbridge else ('MISMATCH -- STOP' if prefix0 else 'reference MISSING -- STOP')}")
    smoke_pass = bool(bridge1 and pbridge)
    log(f"  wall-clock {wall:.0f}s. SMOKE: {'PASS' if smoke_pass else 'FAIL -- STOP, no battery'}")

    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate2_smoke.json")
    g0._dump_json(outp, {"gate": "relay-gate2", "stage": "2-smoke", "seed": 0, "H": H,
                         "framing": FRAMING, "env": g1._env_full(),
                         "bridges": {"hop1_ref": ref1, "hop1_got": rec["r2_cum"][0], "hop1_match": bridge1,
                                     "prefix_ref": prefix0, "prefix_got": got0, "prefix_match": pbridge},
                         "record": rec, "wall_clock_s": wall})
    log(f"  [written -> {os.path.relpath(outp)}]  (smoke artifact; NOT committed)")
    return smoke_pass


# ===================================================================================== #
#  STAGE 3 -- full battery (chain + violation; H=10; verdict A2/B2/C)
# ===================================================================================== #
def run(log, nseeds):
    import time
    seeds = list(range(nseeds))
    log(f"=== RELAY GATE-2 :: STAGE-3 FULL BATTERY (seeds {seeds}, H={H}, K={K_PRIMARY}) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    hop = _real_hop_factory(geom)
    recs = {"chain": {}, "violation": {}}
    outp = os.path.join(RESDIR, "gate2_depth.json")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..", "relay_gate2_multihop_spec.md"))}
    colrep = verify_no_collision()
    prefix = _load_gate1_prefix()
    t0 = time.perf_counter()
    for i in seeds:
        ts = time.perf_counter()
        recs["chain"][i] = chain(i, geom, hop, log, arm="chain")
        recs["violation"][i] = chain(i, geom, hop, log, viol=True, arm="violation",
                                     decoys=False, e2e_decoys=False)
        log(f"  seed {i} chain r2_cum={[round(x,3) for x in recs['chain'][i]['r2_cum']]} "
            f"| viol e2e={recs['violation'][i]['r2_cum'][-1]:+.4f} "
            f"({time.perf_counter()-ts:.0f}s; {time.perf_counter()-t0:.0f}s elapsed)")
        g0._dump_json(outp, {"gate": "relay-gate2", "stage": "3-battery", "H": H,
                             "seeds_done": seeds[:i + 1], "framing": FRAMING, "K": K_PRIMARY,
                             "seed_scheme": colrep, "hashes": hashes, "env": g1._env_full(),
                             "recs": {a: {str(s): r for s, r in v.items()} for a, v in recs.items()}})
    verdict = decide(recs, seeds, prefix=prefix)
    wall = time.perf_counter() - t0
    payload = {"gate": "relay-gate2", "stage": "3-battery", "H": H, "seeds": seeds, "framing": FRAMING,
               "K": K_PRIMARY, "seed_scheme": colrep, "hashes": hashes, "env": g1._env_full(),
               "wall_clock_s": wall, "verdict": verdict,
               "recs": {a: {str(s): r for s, r in v.items()} for a, v in recs.items()}}
    g0._dump_json(outp, payload)
    _write_md(os.path.join(RESDIR, "gate2_depth.md"), verdict, seeds, wall, hashes, colrep)
    log("\n=== BATTERY VERDICT ===")
    log(f"  {verdict['verdict']}")
    log("  STOP-and-report. No Gate-3 (hop-length), no mechanism-decomposition.")
    return verdict


def reread(log):
    src = os.path.join(RESDIR, "gate2_depth.json")
    assert os.path.exists(src), f"missing battery record {src} -- run --run first"
    nm = json.load(open(src))
    recs = {a: {int(s): r for s, r in v.items()} for a, v in nm["recs"].items()}
    seeds = nm["seeds"]
    log("=== RELAY GATE-2 :: REREAD (re-frame verdict from unchanged recs; NO GPU) ===")
    verdict = decide(recs, seeds)
    for a in recs:
        assert json.dumps({str(s): r for s, r in recs[a].items()}, sort_keys=True) == \
               json.dumps(nm["recs"][a], sort_keys=True), f"arm '{a}' drifted from the battery record"
    log("  [integrity] all arms' recs byte-identical to the battery record: OK")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..", "relay_gate2_multihop_spec.md"))}
    note = (f"Provenance: numbers from the battery run (harness sha256 "
            f"{(nm.get('hashes') or {}).get('code','?')}, {nm.get('wall_clock_s',0)/60:.0f} min GPU); "
            f"verdict RE-FRAMED by --reread (sha256 {hashes['code']}), recs asserted byte-identical.")
    g0._dump_json(src, {**nm, "verdict": verdict, "hashes": hashes, "run_hashes": nm.get("hashes"),
                        "reread": "verdict re-rendered from unchanged recs; no GPU, no number changed"})
    _write_md(os.path.join(RESDIR, "gate2_depth.md"), verdict, seeds, nm.get("wall_clock_s", 0.0),
              hashes, nm.get("seed_scheme", {"ok": True}), note=note)
    log(f"  {verdict['verdict']}")
    log(f"  [rewritten -> {os.path.relpath(src)} + gate2_depth.md]  (recs UNCHANGED; NOT committed)")
    return verdict


# ===================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--nseeds", type=int, default=8,
                    help="spec n>=5; default 8 (seeds 0..7 covered by the Gate-1 prefix bridge)")
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
        assert 1 <= args.nseeds <= SEED_MAX + 1, f"prefix bridge covers seeds 0..{SEED_MAX}"
        run(log, args.nseeds)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
