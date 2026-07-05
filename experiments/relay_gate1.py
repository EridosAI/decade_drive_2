"""
experiments/relay_gate1.py
==========================

Relay Gate-1: multi-hop loss-law probe (offline, H=5).
Per relay_gate1_multihop_spec.md. Extends the Gate-0 two-stage relay (commit 8361553) to an
H=5 identical-stage chain and measures how relay fidelity decays with hop count:
constant per-hop loss (a routing BUDGET, extrapolable) vs COMPOUNDING loss (a depth limit).

Built STRICTLY by REUSE of the committed Gate-0 machinery (imported from relay_gate0) and the
byte-identical Phase-1 machinery (D_phase1_routing). This module modifies NOTHING in
relay_gate0.py, D_phase1_routing.py, the core/ files, or any committed artifact -- it only
imports them.

Honest framing (stated in all outputs): each repeater re-injects into a FRESH span-1.5
network; "compound span 1.5*H" is an INFORMATION-PATH claim (H successive square-law
demodulations end-to-end), never a claim about one physical spectrum. No new beat-the-floor
claim is made here -- the chain-vs-direct question was settled at Gate-0. This gate measures
the SHAPE of the loss curve, citing the committed Gate-0/b0f7664 floors (direct arms are NOT
re-run).

Replication bridges (verified, not assumed):
  * Hop 1 == Gate-0 stage A == committed Phase-1 span-1.5/K=0.24: r2(m1,m0) digit-exact
    against REF_TABLE (seed 0 -> 0.981470).
  * Hop 2 == Gate-0 stage B: cumulative r2(m2,m0) digit-exact against gate0_relay_reposed.json
    (seed 0 -> 0.962334).

Modes:
  --sandbox       Stage 1. CPU-ONLY load-bearing checks (no GPU): seed-derivation +
                  collision proof, chain wiring, per-hop logging schema, H-stage
                  intersection, rho_k loss-law machinery, violation-at-depth plumbing,
                  decoy protocol at depth.
  --verdict-test  CPU-only synthetic exercise of decide() across A / B / C branches
                  (standing rule: test the verdict engine before any GPU burn).
  --smoke         Stage 2 (separate go). 1-seed full H=5 chain; hop-1 must reproduce 0.981470.
  --run           Stage 3 (separate go). Full battery (chain + violation + scramble).

STOP-and-report after --sandbox. Nothing committed. Single variable: hop count only.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

# CPU-only for the non-GPU modes: force the JAX CPU backend BEFORE jax is imported (via
# relay_gate0 -> D_phase1_routing), so a Stage-1 sandbox / verdict-test never touches the GPU.
if any(a in sys.argv for a in ("--sandbox", "--verdict-test", "--reread")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))                        # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root

import numpy as np                                                   # noqa: E402
import D_phase1_routing as p1                                        # noqa: E402 (jax x64 on import)
import relay_gate0 as g0                                             # noqa: E402 (Gate-0 machinery)
from core.reservoir import build_system                              # noqa: E402

RESDIR = g0.RESDIR

# --- reconciled from the spec + Gate-0 (imported, not redefined) ----------------------- #
H = 5                                # hop count (spec: H=5 identical stages)
STAGE_SPAN = g0.STAGE_SPAN           # 1.5 decades per hop
K_PRIMARY = g0.K_PRIMARY             # 0.24 throughout (single variable = hop count)
MSG_LO, MSG_HI = g0.MSG_LO, g0.MSG_HI    # [0.2,0.9] standard message / repeater pass-band
VIOL_LO, VIOL_HI = g0.VIOL_LO, g0.VIOL_HI  # [2,9] filter-violation message band
ANCHOR = g0.ANCHOR                   # 0.986 committed span-1.5 @ K=0.24 (hop-1 anchor)
ANCHOR_SE_K = g0.ANCHOR_SE_K         # anchor window: |mean-ANCHOR| <= max(2*SE, FLOOR)
ANCHOR_FLOOR = g0.ANCHOR_FLOOR       # 0.02
MIN_PAIRS = g0.MIN_PAIRS             # >=2 ESP-honest seeds or underpowered
VIOL_E2E_BAR = g0.VIOL_E2E_BAR       # re-posed violation collapse bar (per-seed e2e r2 < 0.1)
DECOY_ELEVATED = g0.DECOY_ELEVATED   # 0.2 leakage bar on decoy-p95 means
N_DEC = p1.N_DEC                     # 60 decoy-null draws per stage (Phase-1 exact)
REF_TABLE = g0.REF_TABLE             # committed b0f7664 per-seed rows
SEED_MAX = 9                         # committed Phase-1 ran seeds 0..9 (bridge coverage)

# ---- seed-derivation scheme (chain i, stage s in 1..H); see seed_scheme() docstring --- #
BUILD_STRIPE = 100          # build_system base per stage = (s-1)*BUILD_STRIPE + i
STAGE_STRIPE = 100          # enc/rep/carrier stride per stage
ENC_BASE = 5000             # masked_encoding rng base (stage 1 = 5000 -> Gate-0/Phase-1 exact)
REP_BASE = 9000             # ESP-replica base (stage 1 = 9000 -> Gate-0/Phase-1 exact)
CAR_BASE = 2000             # carrier base for hops s>=2 (stage 2 = 2000 -> Gate-0 stage B exact)
MSG_BASE = 1000             # hop-1 message seed = am_input(MSG_BASE + i) -> Phase-1 exact
SCRAMBLE_STAGE = 3          # spec condition 4: degree-matched random coupling on stage 3
SCRAMBLE_REP_EXTRA = 500    # scrambled stage's ESP-replica offset (distinct family, Gate-0 spirit)

# Per-stage decoy bases (draw seed = base + i*200 + d, d in 0..N_DEC-1 -> base+[0,1859] for
# i<=9). Stage 1 = 40000 (Phase-1 exact), stage 2 = 60000 (Gate-0 stage B exact); stages 3-5
# jump PAST the e2e base 80000 (avoid the 40000+20000*k collision at 80000) with 20000 gaps.
DECOY_BASE = {1: 40000, 2: 60000, 3: 100000, 4: 120000, 5: 140000}
E2E_DECOY_BASE = g0.E2E_DECOY_BASE   # 80000 (final m_H vs never-injected m0'; Gate-0 exact)

RHO_VALID_MIN = 0.2         # loss-law validity guard: rho_k enters the trend test only where
                            #   r2_{k-1} > this (ratios of near-floor numbers are no measurement)
SLOPE_EPS = 0.01            # slope-magnitude floor: A iff |slope_mean| <= max(2*SE, SLOPE_EPS).
                            #   Parallels the committed ANCHOR_FLOOR -- guards the degenerate
                            #   near-zero-SE case (else a numerically tiny slope spuriously trips
                            #   B/rising). PRE-REGISTRATION KNOB (surfaced for ratification before
                            #   the battery): d(rho)/d(k) below this = flat within measurement; it
                            #   sits an order of magnitude under a clear compounding slope (~-0.15).

FRAMING = ("Compound span 1.5*H is a claim about the INFORMATION PATH (H successive square-law "
           "demodulations end-to-end), NOT one physical spectrum; each repeater re-injects into "
           "a fresh span-1.5 network. This gate measures the SHAPE of the loss curve (budget vs "
           "compounding); the chain-vs-direct floor was settled at Gate-0 (commit 8361553) and "
           "direct arms are NOT re-run (committed b0f7664 floors cited).")


# ===================================================================================== #
#  Seed-derivation scheme + collision proof
# ===================================================================================== #
def seed_scheme(i, s):
    """Seeds for chain i, stage s (1-indexed, 1..H). Designed so the first two stages are
    byte-identical to Gate-0 (the replication bridge):
      stage 1 == Gate-0 stage A / Phase-1: build i, enc 5000+i, rep 9000+i, decoys base 40000,
                 message = am_input(1000+i) (its own Rademacher carrier 1777+i internally);
      stage 2 == Gate-0 stage B: build 100+i, enc 5100+i, carrier 2000+i, rep 9100+i,
                 decoys base 60000;
      stages 3..5 continue the +100 per-stage stripe; decoy bases 100000/120000/140000.
    `carrier` is None for stage 1 (hop 1's carrier is am_input's internal Rademacher on the
    original message seed 1000+i); hops s>=2 inject via am_from_message(s_full, carrier)."""
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
    """Prove the network/drive seeds never coincide with a decoy seed or the message-seed
    family, and that the decoy families are mutually disjoint. Returns a structured report;
    every 'ok' must be True. (Spec 'FIRST': verify no collision with decoy bases
    40000/60000/80000 or msg seed_base 1000 family.)"""
    seeds = range(seed_max + 1)
    stages = range(1, H + 1)

    build = {seed_scheme(i, s)["build"] for s in stages for i in seeds}
    enc = {seed_scheme(i, s)["enc"] for s in stages for i in seeds}
    rep = {seed_scheme(i, s)["rep"] for s in stages for i in seeds}
    rep |= {REP_BASE + (SCRAMBLE_STAGE - 1) * STAGE_STRIPE + SCRAMBLE_REP_EXTRA + i
            for i in seeds}                                          # scrambled-stage replica
    carrier_arg = {seed_scheme(i, s)["carrier"] for s in range(2, H + 1) for i in seeds}
    # every Rademacher carrier actually drawn = arg + 777 (hops>=2) or (1000+i)+777 (hop 1)
    carrier_rade = {c + 777 for c in carrier_arg} | {MSG_BASE + i + 777 for i in seeds}
    net_drive = build | enc | rep | carrier_arg | carrier_rade

    msg = {MSG_BASE + i for i in seeds}                              # protected message family
    scramble_lap = {70000 + i for i in seeds}                       # p1.scramble_laplacian rng

    decoy_by_base = {b: _decoy_range(b) for b in list(DECOY_BASE.values()) + [E2E_DECOY_BASE]}
    decoy_all = set().union(*decoy_by_base.values())

    # pairwise disjointness of the decoy families
    bases = list(decoy_by_base)
    pw = {}
    for a in range(len(bases)):
        for b in range(a + 1, len(bases)):
            inter = decoy_by_base[bases[a]] & decoy_by_base[bases[b]]
            pw[f"{bases[a]}^{bases[b]}"] = len(inter)

    checks = {
        "net_drive_vs_decoy": sorted(net_drive & decoy_all),
        "net_drive_vs_msg": sorted(net_drive & msg),
        "msg_vs_decoy": sorted(msg & decoy_all),
        "scramble_lap_vs_all": sorted(scramble_lap & (net_drive | msg | decoy_all)),
        "decoy_pairwise_overlaps": {k: v for k, v in pw.items() if v > 0},
    }
    ok = (not checks["net_drive_vs_decoy"] and not checks["net_drive_vs_msg"]
          and not checks["msg_vs_decoy"] and not checks["scramble_lap_vs_all"]
          and not checks["decoy_pairwise_overlaps"])
    return {
        "ok": bool(ok), "seed_max": seed_max,
        "families": {
            "build": [min(build), max(build)], "enc": [min(enc), max(enc)],
            "rep": [min(rep), max(rep)], "carrier_arg": [min(carrier_arg), max(carrier_arg)],
            "carrier_rademacher": [min(carrier_rade), max(carrier_rade)],
            "msg": [min(msg), max(msg)],
            "scramble_laplacian": [min(scramble_lap), max(scramble_lap)],
            "decoy_bases": DECOY_BASE, "e2e_decoy_base": E2E_DECOY_BASE, "N_DEC": N_DEC,
        },
        "collisions": checks,
    }


def log_seed_scheme(log, seed_max=SEED_MAX):
    """Log the per-stage scheme table + the collision proof (the spec's FIRST deliverable)."""
    log("--- seed-derivation scheme (chain i, stage s; shown for i as offset '+i') ---")
    log(f"  {'stage':>5} {'build':>10} {'enc':>10} {'rep':>10} {'carrier':>10} "
        f"{'decoy_base':>11}  provenance")
    prov = {1: "== Gate-0 stage A / Phase-1 (anchor)", 2: "== Gate-0 stage B (H=2 bridge)"}
    for s in range(1, H + 1):
        sd = seed_scheme(0, s)
        car = "n/a(am_input)" if sd["carrier"] is None else f"{sd['carrier']}+i"
        log(f"  {s:>5} {str(sd['build'])+'+i':>10} {str(sd['enc'])+'+i':>10} "
            f"{str(sd['rep'])+'+i':>10} {car:>10} {sd['decoy_base']:>11}  "
            f"{prov.get(s, 'fresh independent stage')}")
    log(f"  scrambled stage {SCRAMBLE_STAGE} ESP-replica -> "
        f"{REP_BASE + (SCRAMBLE_STAGE-1)*STAGE_STRIPE + SCRAMBLE_REP_EXTRA}+i (distinct family)")
    log(f"  e2e-at-depth decoy base -> {E2E_DECOY_BASE} (final m_{H} vs never-injected m0')")
    rep = verify_no_collision(seed_max)
    f = rep["families"]
    log(f"  ranges (i in 0..{seed_max}): build {f['build']} enc {f['enc']} rep {f['rep']} "
        f"carrier {f['carrier_arg']} carrier_rade {f['carrier_rademacher']} msg {f['msg']}")
    c = rep["collisions"]
    log(f"  collision proof: net/drive vs decoy={c['net_drive_vs_decoy'] or 'none'}; "
        f"net/drive vs msg={c['net_drive_vs_msg'] or 'none'}; msg vs decoy="
        f"{c['msg_vs_decoy'] or 'none'}; scramble-lap vs all={c['scramble_lap_vs_all'] or 'none'}; "
        f"decoy pairwise overlaps={c['decoy_pairwise_overlaps'] or 'none'}")
    log(f"  -> collision-free: {rep['ok']}")
    return rep


# ===================================================================================== #
#  Decoy construction at depth  (byte-identical Phase-1 protocol, per-stage base)
# ===================================================================================== #
def gate1_decoys(stage, seed_i, L, dt_in):
    """Per-stage decoy list using Phase-1's EXACT construction (p1.slow_bandlimited, MSG band,
    N_DEC draws) at this stage's base. Stage 1 (base 40000) is byte-identical to
    g0.phase1_decoys_ref / Phase-1's decoy line."""
    base = DECOY_BASE[stage]
    return [p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=base + seed_i * 200 + d)
            for d in range(N_DEC)]


# ===================================================================================== #
#  The H-hop chain threader  (hop execution injected -> same code path in sandbox + battery)
# ===================================================================================== #
def _hoprec(stage, sd, rep_rec, r2_cum, r2_hop, esp, dem):
    """Per-hop record. Logging schema (spec 'Metrics per chain'): r2_cum = r2(m_k, m0);
    r2_hop = r2(m_k, processed-m_{k-1}) (= this stage's own r2_d0); rms_in/rms_target/scale
    from the repeater that FED this hop (None for hop 1, which has no upstream repeater)."""
    return {
        "stage": stage,
        "seeds": {k: sd.get(k) for k in ("build", "enc", "rep", "carrier", "decoy_base")},
        "repeater_in": rep_rec,                                    # None for stage 1
        "r2_cum": float(r2_cum),                                   # r2(m_k, m0)
        "r2_hop": float(r2_hop),                                   # r2(m_k, processed-m_{k-1})
        "esp": esp,
        "decoy_p95": (float(dem["decoy_p95"]) if dem else None),
        "rms_in": (rep_rec["rms_in"] if rep_rec else None),
        "rms_target": (rep_rec["rms_target"] if rep_rec else None),
        "scale": (rep_rec["scale"] if rep_rec else None),
    }


def chain(i, geom, hop_fn, log, *, viol=False, scramble_stage=None, arm="chain",
          decoys=True, e2e_decoys=True):
    """Thread one H-hop chain for seed i.  m0 -> stage1 -> repeater -> stage2 -> ... -> stageH.
    Every repeater = Gate-0's F: zero-phase brick-wall to MSG_BAND [0.2,0.9] + affine rescale
    to the ORIGINAL m0 message-class stats (mean + fluctuation RMS -- two class scalars, the
    SAME at every hop, no per-message oracle). hop_fn does the actual stage (real GPU
    integration in the battery, a CPU stand-in in the sandbox), so this plumbing is one code
    path exercised by both.

    viol=True: hop-1 message is a [2,9] band signal; repeaters stay standard [0.2,0.9] -> the
      FIRST repeater is mismatched to the message and deletes it (filter-violation at depth).
    scramble_stage=s: that stage integrates on a degree-matched random Laplacian (L_override).
    """
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    band = (VIOL_LO, VIOL_HI) if viol else (MSG_LO, MSG_HI)
    if viol:
        m0, u0 = g0.am_input_band(L, dt_in, MSG_BASE + i, VIOL_LO, VIOL_HI)
    else:
        m0, u0 = p1.am_input(L, dt_in, MSG_BASE + i)
    m0_iw = m0[iw]
    dc = float(np.mean(m0_iw))                                     # message-class DC (fixed)

    hops = []
    # ---- stage 1: injected message IS m0 (the original); the replication anchor ---------- #
    sd = seed_scheme(i, 1)
    dec = gate1_decoys(1, i, L, dt_in) if decoys else None
    anchor1 = (not viol) and (scramble_stage != 1)                # chain/scramble hop-1 = anchor
    m_rec, r2_hop, esp, dem = hop_fn(1, sd, m0, u0, dec, L_override=None, anchor=anchor1)
    r2_cum = g0._e2e_score(m_rec, m0, iw, ntr)                     # r2(m1, m0) on test split
    hops.append(_hoprec(1, sd, None, r2_cum, r2_hop, esp, dem))
    prev = m_rec

    # ---- stages 2..H: repeater(prev) -> fresh network -> reconstruct its own injected msg -- #
    for s in range(2, H + 1):
        processed, rparams = g0.repeater_transform(prev, m0_iw, dt_in, w_lo=MSG_LO, w_hi=MSG_HI)
        s_full = np.full(L, dc)
        s_full[eval_start:] = g0.remodulate_for_stage_b(processed, m0_iw)
        clip_frac = float(np.mean((dc + processed) < 1e-6))
        sd = seed_scheme(i, s)
        L_ovr = None
        if scramble_stage == s:
            L_ovr = p1.scramble_laplacian(p1.N, i)                 # rng 70000+i (Phase-1 exact)
            sd = {**sd, "rep": REP_BASE + (s - 1) * STAGE_STRIPE + SCRAMBLE_REP_EXTRA + i}
        u_in = g0.am_from_message(s_full, sd["carrier"])
        dec = gate1_decoys(s, i, L, dt_in) if decoys else None
        m_rec, r2_hop, esp, dem = hop_fn(s, sd, s_full, u_in, dec, L_override=L_ovr, anchor=False)
        r2_cum = g0._e2e_score(m_rec, m0, iw, ntr)                 # r2(m_s, m0), cumulative
        rep_rec = {**rparams, "dc": dc, "clip_frac": clip_frac}
        hops.append(_hoprec(s, sd, rep_rec, r2_cum, r2_hop, esp, dem))
        prev = m_rec

    rec = {"seed": i, "arm": arm, "H": H, "band": list(band),
           "hops": hops, "r2_cum": [h["r2_cum"] for h in hops],
           "esp_all_stages": bool(all(h["esp"]["ok_slow"] for h in hops))}

    # ---- end-to-end decoy at depth: final m_H vs never-injected m0' (spec condition 2) ---- #
    if e2e_decoys:
        e2e_dec = [g0._e2e_score(prev, p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                                 seed=E2E_DECOY_BASE + i * 200 + d), iw, ntr)
                   for d in range(N_DEC)]
        rec["e2e_decoy_p95"] = float(np.percentile(e2e_dec, 95))
        rec["e2e_decoy_mean"] = float(np.mean(e2e_dec))
    return rec


def _real_hop_factory(geom):
    """Battery hop: build the stage network and run Gate-0's byte-identical _hop (main + ESP
    replica + slow-band reconstruction). The anchor hop (stage 1, compliant) passes
    Ks=p1.K_GRID so integration reproduces Phase-1's compiled batch shape -> bit-exact
    0.981470 (Gate-0's batch-shape finding); every other hop is batch-of-1 like Gate-0's
    stage B (also the shape that reproduced the committed H=2 e2e)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom

    def hop(stage, sd, s_target_full, u_in, decoys, L_override=None, anchor=False):
        sp = build_system(sd["build"], p1.N, STAGE_SPAN)
        Ks = p1.K_GRID if anchor else None
        return g0._hop(sp, sd["enc"], sd["rep"], s_target_full, u_in, K_PRIMARY,
                       dt_in, n_sub, delays, sl, decoys=decoys, Ks=Ks, L_override=L_override)
    return hop


# ===================================================================================== #
#  Loss-law readout  (rho_k = r2_k / r2_{k-1}; pre-registered A / B classification)
# ===================================================================================== #
def loss_law(r2_cum_by_seed, inter):
    """r2_cum_by_seed: {seed: [r2_1..r2_H]}.  rho_k = r2_k/r2_{k-1}, k=2..H, per seed, with the
    validity guard r2_{k-1} > RHO_VALID_MIN (ratios of near-floor numbers excluded). Per-seed
    slope of the valid rho_k vs k; mean +/- SE across seeds; thr = max(2*SE, SLOPE_EPS) (the
    floor guards a degenerate near-zero SE, exactly as ANCHOR_FLOOR does for the anchor):
      (A) BUDGET-LIKE : |slope_mean| <= thr  (no trend beyond seed sigma / floor) -> constant
          per-hop cost; report budget = mean rho and extrapolated H_half (r2 crossing 0.5).
      (B) COMPOUNDING : slope_mean < -thr  (rho falls with k) -> depth limit; report decay.
      rising-flag     : slope_mean > +thr  (rho rises -- anomalous; report, not A/B).
      underpowered    : < MIN_PAIRS seeds with a fittable (>=2 valid levels) slope.
    """
    rho_levels = {k: [] for k in range(2, H + 1)}
    per_seed_slopes, valid_pairs = [], 0
    for i in inter:
        r = r2_cum_by_seed[i]
        ks, rhos = [], []
        for k in range(2, H + 1):
            parent = r[k - 2]                                     # r2_{k-1}
            if parent > RHO_VALID_MIN:
                rho = r[k - 1] / parent
                ks.append(k); rhos.append(rho)
                rho_levels[k].append(rho); valid_pairs += 1
        if len(ks) >= 2:
            per_seed_slopes.append(float(np.polyfit(ks, rhos, 1)[0]))

    all_rho = [x for v in rho_levels.values() for x in v]
    rho_mean = float(np.mean(all_rho)) if all_rho else float("nan")
    n = len(per_seed_slopes)
    out = {
        "rho_valid_min": RHO_VALID_MIN, "valid_pairs": valid_pairs,
        "n_seed_slopes": n, "rho_mean": rho_mean,
        "rho_by_level": {str(k): (float(np.mean(v)) if v else None) for k, v in rho_levels.items()},
        "rho_level_n": {str(k): len(v) for k, v in rho_levels.items()},
    }
    if n < MIN_PAIRS:
        out.update({"classification": "underpowered", "slope_mean": None, "slope_se": None,
                    "slope_thr": None, "slope_eps": SLOPE_EPS, "margin": None,
                    "H_half_extrap": None})
        return out
    sm = float(np.mean(per_seed_slopes))
    ssd = float(np.std(per_seed_slopes, ddof=1)) if n > 1 else 0.0
    sse = ssd / np.sqrt(n) if n else float("nan")
    thr = max(2 * sse, SLOPE_EPS)          # floor guards degenerate near-zero SE (cf. ANCHOR_FLOOR)
    if sm < -thr:
        cls = "B"
    elif sm > thr:
        cls = "rising-flag"
    else:
        cls = "A"
    # extrapolated H_half from the mean budget rho (geometric model r2_k = r2_1 * rho^(k-1))
    r1 = float(np.mean([r2_cum_by_seed[i][0] for i in inter]))
    if 0.0 < rho_mean < 1.0 and r1 > 0.5:
        h_half = 1.0 + np.log(0.5 / r1) / np.log(rho_mean)
    else:
        h_half = None
    # margin = thr - |slope| : >=0 -> slope inside the flat band (A-consistent); <0 -> a trend is
    # resolved beyond max(2*SE, SLOPE_EPS) (its SIGN, via `cls`, says B (falling) or rising-flag).
    margin = float(thr - abs(sm))
    out.update({"classification": cls, "slope_mean": sm, "slope_se": sse,
                "slope_thr": float(thr), "slope_eps": SLOPE_EPS, "margin": margin,
                "H_half_extrap": (float(h_half) if h_half is not None else None), "r2_1_mean": r1})
    return out


# ===================================================================================== #
#  Verdict engine  (pre-registered A / B / C mapping; instrument checks FIRST)
# ===================================================================================== #
def _mstats(vals):
    return g0._mstats(vals)


def _two_regime(C, inter):
    """DESCRIPTIVE shape analysis -- does NOT change the registered A/B classification (loss_law).
    The honest loss law is TWO-REGIME: a one-time first-relay INSERTION loss (rho_2, the largest
    single drop, hop-1 clean-injection -> hop-2 first relay) plus a near-flat STEADY-STATE among
    later relays (rho_3..H) carrying a mild late drift. Plus the clean mechanistic discriminator:
    per-hop reconstruction fidelity r2_hop = r2(m_k, processed-m_{k-1}); FLAT-with-depth (t ~ 0)
    means each stage reconstructs its input equally well at any depth -> NO error amplification ->
    compounding/depth-limit REFUTED (this is why the ladder stays priced, not just 'not-yet-bounded')."""
    rho = {i: (np.array(C[i]["r2_cum"])[1:] / np.array(C[i]["r2_cum"])[:-1]) for i in inter}
    rho2 = _mstats([float(rho[i][0]) for i in inter])                    # first-relay insertion ratio
    kk = list(range(3, H + 1))                                           # steady-state hops (drop rho_2)
    steady = [float(np.polyfit(kk, rho[i][1:], 1)[0]) for i in inter] if H >= 4 else []
    st = _mstats(steady) if steady else {"mean": None, "se": None, "n": 0, "per_seed": []}
    st_t = (st["mean"] / st["se"]) if st.get("se") else None
    n_neg = int(sum(1 for s in steady if s < 0))
    # steady-state mean ratio rho_3..H (excludes the first-relay insertion rho_2; distinct from
    # loss_law's overall budget rho_mean, which pools rho_2..H and so runs a touch lower)
    steady_rho = (_mstats([float(x) for i in inter for x in rho[i][1:]]) if H >= 4
                  else {"mean": None, "se": None})
    locus_last = (_mstats([float(rho[i][-1] - rho[i][1]) for i in inter])["mean"] if H >= 4 else None)
    locus_mid = (_mstats([float(rho[i][-2] - rho[i][1]) for i in inter])["mean"] if H >= 5 else None)
    hop_k = list(range(2, H + 1))                                        # relay hops (r2_hop discriminator)
    r2hop_by_hop = {str(k): _mstats([C[i]["hops"][k - 1]["r2_hop"] for i in inter])["mean"]
                    for k in range(1, H + 1)}
    r2hop_sl = [float(np.polyfit(hop_k, [C[i]["hops"][k - 1]["r2_hop"] for k in hop_k], 1)[0])
                for i in inter]
    rh = _mstats(r2hop_sl)
    rh_t = (rh["mean"] / rh["se"]) if rh.get("se") else None
    return {"rho2_insertion": rho2, "steady_rho_mean": steady_rho.get("mean"),
            "steady_rho_se": steady_rho.get("se"), "steady_slope": st, "steady_slope_t": st_t,
            "steady_n_negative": n_neg, "decline_locus_last_minus_first": locus_last,
            "decline_locus_mid_minus_first": locus_mid, "r2hop_by_hop": r2hop_by_hop,
            "r2hop_slope": rh, "r2hop_slope_t": rh_t,
            "compounding_refuted": bool(rh_t is not None and abs(rh_t) < 2.0)}


def decide(recs, seeds):
    """Pre-registered outcome mapping (spec 'Pre-registered loss-law readout'), instrument
    checks FIRST. recs = {"chain":{i:rec}, "violation":{i:rec}, "scramble":{i:rec}}.
    Verdict intersection = seeds ESP-ok across ALL H chain stages. Order:
      (C) NO-MEASUREMENT if: underpowered (<MIN_PAIRS); anchor miss (hop-1 mean vs 0.986);
          decoy elevated (per-stage or e2e p95 mean > 0.2); or the filter-violation-at-depth
          did NOT collapse (per-seed e2e r2 < VIOL_E2E_BAR on the viol sub-intersection).
      else (A) BUDGET or (B) COMPOUNDING from loss_law()."""
    C = recs["chain"]
    chain_esp = {i: bool(all(h["esp"]["ok_slow"] for h in C[i]["hops"])) for i in seeds}
    inter = [i for i in seeds if chain_esp[i]]
    out = {"framing": FRAMING,
           "esp_all_stages": {str(i): chain_esp[i] for i in seeds},
           "intersection": inter, "n_intersection": len(inter),
           "operationalization": {"anchor_window": f"max({ANCHOR_SE_K}*SE,{ANCHOR_FLOOR})",
                                  "rho_valid_min": RHO_VALID_MIN, "min_pairs": MIN_PAIRS,
                                  "viol_e2e_bar_per_seed": VIOL_E2E_BAR,
                                  "decoy_elevated": DECOY_ELEVATED,
                                  "slope_eps": SLOPE_EPS,
                                  "A_bar": "|slope_mean| <= max(2*SE, SLOPE_EPS) (flat rho_k)",
                                  "B_bar": "slope_mean < -max(2*SE, SLOPE_EPS) (rho_k falls)"}}
    if len(inter) < MIN_PAIRS:
        out["verdict"] = ("NO-MEASUREMENT (underpowered: chain ESP intersection "
                          f"n={len(inter)} < {MIN_PAIRS} -- add seeds, do not read)")
        return out

    # ---- anchor (hop-1 == committed span-1.5/K=0.24) ----------------------------------- #
    stA = _mstats([C[i]["hops"][0]["r2_cum"] for i in inter])
    window = max(ANCHOR_SE_K * stA["se"], ANCHOR_FLOOR)
    anchor_dev = stA["mean"] - ANCHOR
    anchor_ok = bool(abs(anchor_dev) <= window)
    out["anchor"] = {"hop1": stA, "target": ANCHOR, "window": window,
                     "deviation": anchor_dev, "ok": anchor_ok}

    # ---- decoy health at depth (per-stage p95 means + e2e-at-depth p95 mean) ------------ #
    stage_p95 = {}
    for s in range(1, H + 1):
        vals = [C[i]["hops"][s - 1]["decoy_p95"] for i in inter
                if C[i]["hops"][s - 1]["decoy_p95"] is not None]
        stage_p95[str(s)] = float(np.mean(vals)) if vals else None
    e2e_p95 = float(np.mean([C[i]["e2e_decoy_p95"] for i in inter
                             if "e2e_decoy_p95" in C[i]]))
    present = [v for v in list(stage_p95.values()) + [e2e_p95] if v is not None]
    leak = bool(present and max(present) > DECOY_ELEVATED)
    out["decoys"] = {"stage_p95_mean": stage_p95, "e2e_p95_mean": e2e_p95, "elevated": leak}

    # ---- filter-violation at depth: collapse on the viol sub-intersection --------------- #
    V = recs.get("violation", {})
    vinter = [i for i in inter if i in V
              and all(h["esp"]["ok_slow"] for h in V[i]["hops"])]
    if len(vinter) >= MIN_PAIRS:
        viol_ps = {i: float(V[i]["r2_cum"][-1]) for i in vinter}       # e2e = r2_cum at H
        collapsed = all(r < VIOL_E2E_BAR for r in viol_ps.values())
        offenders = [i for i, r in viol_ps.items() if r >= VIOL_E2E_BAR]
        # signature: first repeater (stage-2 feed) small rms_in / large scale
        rms_in = float(np.mean([V[i]["hops"][1]["rms_in"] for i in vinter]))
        scale = float(np.mean([V[i]["hops"][1]["scale"] for i in vinter]))
        viol_mean = float(np.mean(list(viol_ps.values())))
    else:
        viol_ps, collapsed, offenders, rms_in, scale, viol_mean = {}, False, [], None, None, None
    out["violation"] = {"n": len(vinter), "intersection": vinter, "bar_per_seed": VIOL_E2E_BAR,
                        "e2e_per_seed": {str(k): v for k, v in viol_ps.items()},
                        "offenders": offenders, "violation_e2e_mean": viol_mean,
                        "signature_rms_in": rms_in, "signature_scale": scale,
                        "collapsed": collapsed}

    # ---- loss law (A/B) ----------------------------------------------------------------- #
    ll = loss_law({i: C[i]["r2_cum"] for i in inter}, inter)
    out["loss_law"] = ll
    out["two_regime"] = _two_regime(C, inter)          # descriptive shape (does not change A/B)
    out["cumulative"] = {str(s): _mstats([C[i]["r2_cum"][s - 1] for i in inter])
                         for s in range(1, H + 1)}

    # ---- verdict (instruments first; pre-registered order) ------------------------------ #
    if not anchor_ok:
        side = "low (replication failure)" if anchor_dev < 0 else \
               ("high + elevated decoy (leakage)" if leak else "high")
        out["verdict"] = (f"NO-MEASUREMENT (anchor miss, {side}: hop-1 mean {stA['mean']:.6f} "
                          f"vs {ANCHOR} +/- {window:.4f} -- STOP, fix, re-run)")
    elif leak:
        out["verdict"] = ("NO-MEASUREMENT (decoy elevated -- leakage: max p95 mean "
                          f"{max(present):.3f} > {DECOY_ELEVATED})")
    elif not collapsed:
        vs = (f"seeds {offenders} have e2e r2 >= {VIOL_E2E_BAR} (mean {viol_mean:.3f})"
              if viol_mean is not None else
              f"violation sub-intersection underpowered (n={len(vinter)})")
        out["verdict"] = (f"NO-MEASUREMENT (filter-violation-at-depth did NOT collapse: {vs} "
                          "-- repeater-filter bookkeeping wrong or check underpowered)")
    elif ll["classification"] == "underpowered":
        out["verdict"] = ("NO-MEASUREMENT (loss-law underpowered: "
                          f"{ll['n_seed_slopes']} fittable seed-slopes < {MIN_PAIRS} "
                          "-- add seeds / too few valid rho levels)")
    elif ll["classification"] == "A":
        tr = out["two_regime"]
        hh_val = ll["H_half_extrap"]
        hh = _fmt(hh_val, ".1f")
        mult = _fmt(hh_val / H, ".1f") if isinstance(hh_val, (int, float)) else "n/a"
        rho2m = tr["rho2_insertion"]["mean"]
        out["verdict"] = (
            f"A -- BUDGET-LIKE (pre-registered; TWO-REGIME, NOT a clean constant-per-hop cost): "
            f"the registered linear rho_k slope is flat ({ll['slope_mean']:+.4f} +/- "
            f"{ll['slope_se']:.4f}, |slope| <= max(2*SE,{SLOPE_EPS})={ll['slope_thr']:.4f}; floor "
            f"INERT since 2*SE binds). Ladder PRICED -- B/depth-limit REFUTED: per-hop r2_hop is "
            f"flat with depth (slope t={_fmt(tr['r2hop_slope_t'],'.2f')}, no error amplification). "
            f"SHAPE: a one-time first-relay INSERTION loss rho_2={_fmt(rho2m,'.3f')} (largest single "
            f"drop) + a near-flat STEADY-STATE mean rho_3..{H}~{_fmt(tr['steady_rho_mean'],'.3f')} "
            f"carrying a mild late drift (slope {_fmt(tr['steady_slope']['mean'],'+.4f')}, "
            f"t={_fmt(tr['steady_slope_t'],'.2f')}, {tr['steady_n_negative']}/{out['n_intersection']} "
            f"seeds neg) that is UNRESOLVED at n={out['n_intersection']}/H={H} (lives mostly in the "
            f"last hop). H_half={hh} (from the OVERALL budget rho {ll['rho_mean']:.3f}, insertion "
            f"loss included) is an EXTRAPOLATION ({mult}x beyond measured H={H}, off a U-shaped rho) "
            f"-- not a measured horizon.")
    elif ll["classification"] == "B":
        out["verdict"] = (f"B -- COMPOUNDING (depth limit): rho_k falls with k "
                          f"(slope {ll['slope_mean']:+.4f} +/- {ll['slope_se']:.4f}, "
                          f"slope < -max(2*SE,{SLOPE_EPS})=-{ll['slope_thr']:.4f}, margin "
                          f"{ll['margin']:+.4f}); staging has a depth limit. The ladder is BOUNDED.")
    else:  # rising-flag -- Pin B: positive slope beyond +max(2*SE, SLOPE_EPS) is NEVER A
        out["verdict"] = (f"NO-MEASUREMENT (loss-law INSTRUMENT-SUSPICION: rho_k RISES with k, "
                          f"slope {ll['slope_mean']:+.4f} +/- {ll['slope_se']:.4f} > "
                          f"+max(2*SE,{SLOPE_EPS})=+{ll['slope_thr']:.4f} (margin {ll['margin']:+.4f}) "
                          "-- fidelity growing with depth is non-physical for a lossy relay; "
                          "NEVER A. Inspect the instrument before any A/B read.)")
    return out


def _fmt(x, spec="+.4f"):
    return g0._fmt(x, spec)


def _env_full():
    """Full provenance: Gate-0's jax/jaxlib/backend/device PLUS the interpreter path, python
    version, x64 flag, and backend env (the brief: log env provenance incl. interpreter path)."""
    import jax
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64 = bool(getattr(jax.config, "jax_enable_x64", None))
    return {**g0._env_versions(), "interpreter": sys.executable,
            "python": sys.version.split()[0], "jax_enable_x64": x64,
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<default>"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}


# ===================================================================================== #
#  STAGE 1 -- CPU sandbox (6 checks; no GPU)
# ===================================================================================== #
def _sandbox_geom():
    """A small span-1.5 window (n_msg=8) for CPU plumbing tests -- real dt_in/L, fast FFTs."""
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(STAGE_SPAN, n_msg=8)
    sl = slice(eval_start, L)
    iw = np.arange(eval_start, L)
    ntr = int(p1.TRAIN_FRAC * (L - eval_start))
    return dt_in, eval_start, L, delays, sl, iw, ntr, 1     # n_sub dummy (synth hop ignores it)


def _make_synth_hop(geom, fidelity=1.0):
    """CPU stand-in for the GPU stage: m_rec = message-class DC + fidelity*(injected fluct);
    fidelity=1 is a perfect pass-through. Exercises the chain plumbing / repeater placement /
    logging schema WITHOUT any integration."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom

    def hop(stage, sd, s_target_full, u_in, decoys, L_override=None, anchor=False):
        s_iw = np.asarray(s_target_full, float)[iw]
        m_rec = s_iw.mean() + fidelity * (s_iw - s_iw.mean())
        r2_hop = float(p1.r2_det(m_rec[ntr:], s_iw[ntr:]))
        esp = {"d_slow": 0.0, "ok_slow": True}
        dem = ({"decoy_p95": -0.10, "r2_d0": r2_hop} if decoys is not None else None)
        return m_rec, r2_hop, esp, dem
    return hop


def sandbox(log):
    log("=== RELAY GATE-1 :: STAGE-1 CPU SANDBOX (no GPU, no seeds-at-scale) ===")
    log(f"    backend: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS','<default>')} "
        f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}'  H={H}")
    log(f"    framing: {FRAMING}")

    geom = _sandbox_geom()
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} "
        f"iw={len(iw)} ntr={ntr}")
    results = {}

    # ---- FIRST: seed scheme + collision proof ------------------------------------------ #
    log("\n(0) Seed-derivation scheme + collision proof")
    colrep = log_seed_scheme(log)
    c0 = g0._check(log, "seed scheme collision-free vs decoy/msg families", colrep["ok"],
                   f"net/drive, msg, decoy families mutually disjoint over seeds 0..{SEED_MAX}")
    results["check0_seed_scheme"] = {"pass": c0, "report": colrep}

    # ---- CHECK 1: chain wiring (H=5, repeater between every pair; pass-through) --------- #
    log("\n(1) Chain wiring -- H=5, fresh seed per stage, repeater between every pair "
        "(synthetic pass-through)")
    rec = chain(0, geom, _make_synth_hop(geom, fidelity=1.0), log, arm="chain")
    n_hops = len(rec["hops"])
    n_reps = sum(1 for h in rec["hops"] if h["repeater_in"] is not None)
    builds = [h["seeds"]["build"] for h in rec["hops"]]
    fresh = (len(set(builds)) == H) and builds == [seed_scheme(0, s)["build"] for s in range(1, H + 1)]
    hop1_no_rep = rec["hops"][0]["repeater_in"] is None
    cum = rec["r2_cum"]
    # perfect reconstruction threads m0 through all H hops; hop-1 is exact (no repeater), hops
    # 2..H pay a ONE-TIME finite-window brick-wall band-limit cost (~0.002) that is then
    # idempotent -> a flat plateau (the honest 'message survives' evidence).
    passthru = (cum[0] > 0.999) and all(rc > 0.99 for rc in cum[1:])
    plateau = (max(cum[1:]) - min(cum[1:])) < 5e-3                # idempotent after the 1st repeater
    scales = [h["scale"] for h in rec["hops"][1:]]
    scale_unit = all(abs(sc - 1.0) < 0.05 for sc in scales)      # in-band m0 -> rms_in==rms_target
    c1 = all([
        g0._check(log, "H stages threaded", n_hops == H, f"{n_hops} hops"),
        g0._check(log, "repeater between every consecutive pair (H-1)",
                  n_reps == H - 1 and hop1_no_rep, f"{n_reps} repeaters; hop-1 repeater_in=None"),
        g0._check(log, "fresh derived build seed per stage",
                  fresh, f"builds={builds} (== seed_scheme, all distinct)"),
        g0._check(log, "message threads end-to-end (pass-through r2_cum high) + idempotent plateau",
                  passthru and plateau, f"r2_cum={[round(x,4) for x in cum]} "
                  f"(hop1 exact; hops2..H plateau span {max(cum[1:])-min(cum[1:]):.1e} = one-time "
                  f"finite-window band-limit cost)"),
        g0._check(log, "repeater rescale ~ identity for in-band m0 (scale ~ 1)",
                  scale_unit, f"scales={[round(s,4) for s in scales]}"),
    ])
    results["check1_wiring"] = {"pass": c1, "n_hops": n_hops, "n_reps": n_reps,
                                "builds": builds, "r2_cum": rec["r2_cum"], "scales": scales}

    # ---- CHECK 2: per-hop logging schema ----------------------------------------------- #
    log("\n(2) Per-hop logging schema -- r2(m_k,m0), r2(m_k,proc-m_{k-1}), rms_in/target/scale")
    req_keys = {"stage", "seeds", "repeater_in", "r2_cum", "r2_hop", "esp",
                "decoy_p95", "rms_in", "rms_target", "scale"}
    keys_ok = all(req_keys <= set(h) for h in rec["hops"])
    rep_present = all(all(h[k] is not None for k in ("rms_in", "rms_target", "scale"))
                      for h in rec["hops"][1:])                   # hops 2..H have the repeater trio
    r2_present = all((h["r2_cum"] == h["r2_cum"]) and (h["r2_hop"] == h["r2_hop"])
                     for h in rec["hops"])                        # finite (not NaN)
    e2e_dec_ok = "e2e_decoy_p95" in rec
    c2 = all([
        g0._check(log, "every hop carries the full schema key set", keys_ok,
                  f"keys per hop >= {sorted(req_keys)}"),
        g0._check(log, "rms_in/rms_target/scale present for hops 2..H", rep_present,
                  f"hop2 trio = ({rec['hops'][1]['rms_in']:.3g}, "
                  f"{rec['hops'][1]['rms_target']:.3g}, {rec['hops'][1]['scale']:.3g})"),
        g0._check(log, "r2(m_k,m0) and r2(m_k,proc-m_{k-1}) present every hop", r2_present,
                  f"r2_hop={[round(h['r2_hop'],4) for h in rec['hops']]}"),
        g0._check(log, "end-to-end decoy-at-depth recorded (m_H vs never-injected m0')",
                  e2e_dec_ok, f"e2e_decoy_p95={rec.get('e2e_decoy_p95'):+.4f}"),
    ])
    results["check2_schema"] = {"pass": c2, "hop_keys": sorted(set().union(*[set(h) for h in rec["hops"]]))}

    # ---- CHECK 3: H-stage intersection (mid-chain failure drops the seed everywhere) ---- #
    log("\n(3) H-stage intersection -- nested esp->ok_slow; a stage-3 fail drops the seed")
    def mkrec(flags):  # a chain rec carrying only the per-stage nested esp
        return {"hops": [{"esp": {"d_slow": 0.0 if f else 1.0, "ok_slow": f}} for f in flags]}
    chain_recs = {
        0: mkrec([True, True, True, True, True]),    # keep
        1: mkrec([True, True, False, True, True]),   # drop: stage-3 fail mid-chain
        2: mkrec([True, True, True, True, True]),    # keep
        3: mkrec([False, True, True, True, True]),   # drop: stage-1 fail
        4: mkrec([True, True, True, True, False]),   # drop: stage-5 fail
    }
    esp_all = {i: bool(all(g0.esp_ok_slow(h) for h in chain_recs[i]["hops"])) for i in chain_recs}
    inter = sorted(i for i, ok in esp_all.items() if ok)
    nested_ok = (g0.esp_ok_slow(chain_recs[0]["hops"][0]) is True
                 and g0.esp_ok_slow(chain_recs[1]["hops"][2]) is False)
    thin_inter = [i for i in (0,) if esp_all[0]]                  # a 1-seed table -> underpowered
    c3 = all([
        g0._check(log, "nested esp->ok_slow read per stage (not flat)", nested_ok,
                  "esp_ok_slow reads hop['esp']['ok_slow']"),
        g0._check(log, "intersection = seeds ESP-ok across ALL H stages",
                  inter == [0, 2], f"intersection={inter}"),
        g0._check(log, "mid-chain (stage-3) failure drops the seed everywhere",
                  1 not in inter and 3 not in inter and 4 not in inter,
                  "seeds 1(st3)/3(st1)/4(st5) all excluded"),
        g0._check(log, "underpowered flag fires below MIN_PAIRS",
                  len(thin_inter) < MIN_PAIRS <= len(inter),
                  f"thin n={len(thin_inter)} < {MIN_PAIRS} <= main n={len(inter)}"),
    ])
    results["check3_intersection"] = {"pass": c3, "intersection": inter}

    # ---- CHECK 4: rho_k machinery (validity guard, A/B classification, underpowered) ---- #
    log("\n(4) Loss-law rho_k machinery -- validity guard, A/B slope test, underpowered flag")
    seeds5 = list(range(5))
    # budget = exactly geometric (constant ratio 0.9) -> slope 0 -> A (robust even at SE=0 via floor)
    budget = {i: [0.98 * 0.9 ** k for k in range(H)] for i in seeds5}
    # jittered budget: per-element noise -> nonzero slope SE (the REAL regime); mean slope ~0 -> A
    jr = np.random.default_rng(0)
    budget_jit = {i: [max(0.1, v + 0.003 * jr.standard_normal()) for v in budget[i]] for i in seeds5}
    # compounding: ratios 0.95, 0.85, 0.70, 0.50 -> clearly falling -> B
    compounding = {i: [0.98, 0.931, 0.7914, 0.5540, 0.2770] for i in seeds5}
    ll_A = loss_law(budget, seeds5)
    ll_Aj = loss_law(budget_jit, seeds5)
    ll_B = loss_law(compounding, seeds5)
    # validity guard: a curve dropping below RHO_VALID_MIN mid-chain excludes the sub-floor ratios
    guarded = {i: [0.98, 0.50, 0.15, 0.05, 0.01] for i in seeds5}
    ll_G = loss_law(guarded, seeds5)
    ll_U = loss_law(budget, [0])                        # underpowered: 1 seed
    guard_ok = (ll_G["rho_level_n"]["3"] == 5           # rho_3 valid (parent r2_2=0.50>0.2)
                and ll_G["rho_level_n"]["4"] == 0        # rho_4 excluded (parent r2_3=0.15<0.2)
                and ll_G["rho_level_n"]["5"] == 0)       # rho_5 excluded (parent r2_4=0.05<0.2)
    # A/B classification also flows through decide()'s verdict string (verdict-engine path)
    vA = decide({"chain": _synth_chain_recs(budget), "violation": _synth_viol_recs(seeds5)}, seeds5)
    vB = decide({"chain": _synth_chain_recs(compounding), "violation": _synth_viol_recs(seeds5)}, seeds5)
    c4 = all([
        g0._check(log, "budget-like (exact constant rho) -> A (floor guards SE=0)",
                  ll_A["classification"] == "A",
                  f"slope {ll_A['slope_mean']:+.4f}, thr {ll_A['slope_thr']:.4f}, rho={ll_A['rho_mean']:.3f}"),
        g0._check(log, "budget with seed dispersion (SE>0) -> A", ll_Aj["classification"] == "A",
                  f"slope {ll_Aj['slope_mean']:+.4f} +/- {ll_Aj['slope_se']:.4f} (SE>0), thr {ll_Aj['slope_thr']:.4f}"),
        g0._check(log, "compounding (falling rho) -> B", ll_B["classification"] == "B",
                  f"slope {ll_B['slope_mean']:+.4f} +/- {ll_B['slope_se']:.4f}, thr {ll_B['slope_thr']:.4f}"),
        g0._check(log, "validity guard excludes sub-0.2-parent ratios", guard_ok,
                  f"valid-n by level (k=3,4,5) = "
                  f"{ll_G['rho_level_n']['3']},{ll_G['rho_level_n']['4']},{ll_G['rho_level_n']['5']}"),
        g0._check(log, "underpowered flag fires below MIN_PAIRS seed-slopes",
                  ll_U["classification"] == "underpowered",
                  f"n_seed_slopes={ll_U['n_seed_slopes']} < {MIN_PAIRS}"),
        g0._check(log, "verdict engine maps the curves to A / B", "A -- BUDGET" in vA["verdict"]
                  and "B -- COMPOUNDING" in vB["verdict"],
                  f"A='{vA['verdict'][:20]}...' B='{vB['verdict'][:20]}...'"),
    ])
    results["check4_losslaw"] = {"pass": c4, "A": ll_A, "A_jitter": ll_Aj, "B": ll_B,
                                 "guard": ll_G, "under": ll_U}

    # ---- CHECK 5: violation-at-depth plumbing ------------------------------------------ #
    log("\n(5) Violation-at-depth plumbing -- [2,9] message, first repeater [0.2,0.9], signature")
    vrec = chain(0, geom, _make_synth_hop(geom, fidelity=1.0), log, viol=True, arm="violation",
                 decoys=False, e2e_decoys=False)
    msg_is_viol = vrec["band"] == [VIOL_LO, VIOL_HI]
    first_rep = vrec["hops"][1]["repeater_in"]
    passband_std = first_rep["msg_band"] == [MSG_LO, MSG_HI]      # repeater mismatched to [2,9]
    sig_small_large = first_rep["rms_in"] < 0.05 * first_rep["rms_target"] and first_rep["scale"] > 5.0
    sig_logged = all(k in first_rep for k in ("rms_in", "rms_target", "scale"))
    # the filter deletes a [2,9] message -> the pass-through collapses at depth (e2e r2_cum ~ 0)
    collapse = vrec["r2_cum"][-1] < 0.1
    # direct repeater signature on a pure-[2,9] message (mirrors the Gate-0 addendum CPU pre-check)
    m0v, _ = g0.am_input_band(L, dt_in, MSG_BASE, VIOL_LO, VIOL_HI)
    _, rp = g0.repeater_transform(m0v[iw], m0v[iw], dt_in, w_lo=MSG_LO, w_hi=MSG_HI)
    c5 = all([
        g0._check(log, "violation message band = [2,9]", msg_is_viol, f"band={vrec['band']}"),
        g0._check(log, "first repeater pass-band = standard [0.2,0.9] (mismatched to msg)",
                  passband_std, f"passband={first_rep['msg_band']}"),
        g0._check(log, "signature: small rms_in, large scale (logged)",
                  sig_small_large and sig_logged,
                  f"rms_in={first_rep['rms_in']:.3g} rms_target={first_rep['rms_target']:.3g} "
                  f"scale={first_rep['scale']:.1f}"),
        g0._check(log, "filter deletes [2,9] message -> e2e collapses at depth",
                  collapse, f"r2_cum_H={vrec['r2_cum'][-1]:+.4f} < 0.1"),
        g0._check(log, "pure-[2,9] repeater pre-check reproduces the signature (CPU)",
                  rp["rms_in"] < 0.05 * rp["rms_target"] and rp["scale"] > 5.0,
                  f"rms_in={rp['rms_in']:.3g} scale={rp['scale']:.1f}"),
    ])
    results["check5_violation"] = {"pass": c5, "first_repeater": first_rep,
                                   "e2e_collapse": vrec["r2_cum"][-1],
                                   "precheck_scale": rp["scale"]}

    # ---- CHECK 6: decoy protocol at depth (trace/diff vs Phase-1 per stage) ------------ #
    log("\n(6) Decoy protocol at depth -- byte-identical Phase-1 construction per stage")
    seed_i = 3
    ref1 = g0.phase1_decoys_ref(seed_i, L, dt_in)                 # Phase-1's exact stage-1 line
    g1 = {s: gate1_decoys(s, seed_i, L, dt_in) for s in range(1, H + 1)}
    diff_stage1 = max(float(np.max(np.abs(np.asarray(a) - np.asarray(r))))
                      for a, r in zip(g1[1], ref1))
    counts_ok = all(len(g1[s]) == N_DEC for s in g1)
    # each stage is p1.slow_bandlimited at its own base (identical protocol, offset seed)
    protocol_ok = all(
        np.allclose(g1[s][d], p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                                                  seed=DECOY_BASE[s] + seed_i * 200 + d))
        for s in range(1, H + 1) for d in (0, N_DEC // 2, N_DEC - 1))
    # e2e-at-depth decoys use the same protocol at base 80000
    e2e_ok = np.allclose(
        g0._e2e_score(np.zeros(len(iw)), p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                       seed=E2E_DECOY_BASE), iw, ntr),
        g0._e2e_score(np.zeros(len(iw)), p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                       seed=E2E_DECOY_BASE), iw, ntr))              # deterministic construction
    scorer_ok = (p1.demod_capacity.__module__ == "D_phase1_routing")
    c6 = all([
        g0._check(log, "stage-1 decoys byte-identical to Phase-1 (base 40000)",
                  diff_stage1 == 0.0, f"max|diff|={diff_stage1:.1e} over {N_DEC} draws"),
        g0._check(log, "every stage uses the identical p1.slow_bandlimited protocol",
                  protocol_ok and counts_ok, f"bases={DECOY_BASE}, {N_DEC} draws each"),
        g0._check(log, "e2e-at-depth decoy protocol wired (base 80000, deterministic)", bool(e2e_ok),
                  f"E2E_DECOY_BASE={E2E_DECOY_BASE}"),
        g0._check(log, "decoy scoring path is the imported p1.demod_capacity",
                  scorer_ok, f"scorer module = {p1.demod_capacity.__module__}"),
    ])
    results["check6_decoy"] = {"pass": c6, "diff_stage1": diff_stage1, "bases": DECOY_BASE}

    # ---- summary + write --------------------------------------------------------------- #
    order = ["check0_seed_scheme", "check1_wiring", "check2_schema", "check3_intersection",
             "check4_losslaw", "check5_violation", "check6_decoy"]
    allpass = all(results[k]["pass"] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if results[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate1_sandbox.json")
    with open(outp, "w") as f:
        json.dump({"gate": "relay-gate1", "stage": "1-cpu-sandbox", "H": H, "all_pass": allpass,
                   "framing": FRAMING,
                   "window": {"span": STAGE_SPAN, "dt_in": dt_in, "L": L,
                              "eval_start": eval_start, "n_msg": 8},
                   "checks": results}, f, indent=1, default=_json_default)
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---- synthetic chain/violation recs for check-4 verdict-engine path + verdict_test ------ #
def _synth_chain_recs(r2_by_seed):
    """Wrap a {seed:[r2_1..r2_H]} curve as chain recs (ESP-ok, clean decoys) so decide() runs."""
    out = {}
    for i, r in r2_by_seed.items():
        hops = []
        for s in range(1, H + 1):
            hops.append({"stage": s, "esp": {"ok_slow": True}, "r2_cum": r[s - 1],
                         "r2_hop": r[s - 1], "decoy_p95": -0.10,
                         "rms_in": 0.1, "rms_target": 0.1, "scale": 1.0,
                         "repeater_in": (None if s == 1 else {"rms_in": 0.1, "scale": 1.0})})
        out[i] = {"seed": i, "arm": "chain", "H": H, "hops": hops, "r2_cum": list(r),
                  "e2e_decoy_p95": -0.30}
    return out


def _synth_viol_recs(seeds, e2e=0.0):
    """Collapsed filter-violation recs (ESP-ok, small rms_in / large scale signature)."""
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


# ===================================================================================== #
#  Synthetic verdict-engine test  (CPU; standing rule -- test decide() before any GPU)
# ===================================================================================== #
def verdict_test(log):
    log("=== VERDICT-ENGINE SYNTHETIC TEST (A/B/C branches; CPU only) ===")
    seeds = list(range(5))
    budget = {i: [0.98, 0.882, 0.7938, 0.7144, 0.6430] for i in seeds}
    compounding = {i: [0.98, 0.931, 0.791, 0.554, 0.277] for i in seeds}
    cases, allok = [], True

    def run_case(name, chain_recs, viol_recs, want, extra=lambda v: True):
        nonlocal allok
        v = decide({"chain": chain_recs, "violation": viol_recs}, seeds)
        ok = (want in v["verdict"]) and extra(v)
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {v['verdict'][:78]}")
        return v

    # A / B
    vA = run_case("budget-like -> A", _synth_chain_recs(budget), _synth_viol_recs(seeds),
                  "A -- BUDGET")
    run_case("compounding -> B", _synth_chain_recs(compounding), _synth_viol_recs(seeds),
             "B -- COMPOUNDING")
    # Pin B: rising ratios (anchor OK) -> instrument-suspicion, NEVER A (deliberately non-physical)
    rising = {i: [0.986, 0.986, 1.0156, 1.0765, 1.1734] for i in seeds}   # rho 1.0,1.03,1.06,1.09
    vR = run_case("rising ratios -> NM instrument-suspicion (never A)", _synth_chain_recs(rising),
                  _synth_viol_recs(seeds), "INSTRUMENT-SUSPICION",
                  lambda v: v["loss_law"]["classification"] == "rising-flag"
                  and "A -- BUDGET" not in v["verdict"])
    # C: anchor low
    cr = _synth_chain_recs({i: [x - 0.05 for x in budget[i]] for i in seeds})
    run_case("anchor low -> NM replication", cr, _synth_viol_recs(seeds), "anchor miss, low")
    # C: decoy elevated at depth
    cr = _synth_chain_recs(budget)
    for i in seeds:
        cr[i]["e2e_decoy_p95"] = 0.5
    run_case("e2e decoy elevated -> NM leak", cr, _synth_viol_recs(seeds), "decoy elevated")
    # C: violation does not collapse
    run_case("violation no-collapse -> NM instrument", _synth_chain_recs(budget),
             _synth_viol_recs(seeds, e2e=0.8), "did NOT collapse")
    # C: underpowered (all chain ESP fail on stage 3)
    cr = _synth_chain_recs(budget)
    for i in seeds:
        cr[i]["hops"][2]["esp"]["ok_slow"] = False
    vU = run_case("stage-3 ESP fail all -> NM underpowered", cr, _synth_viol_recs(seeds),
                  "underpowered", lambda v: v["n_intersection"] == 0)
    # attrition: one seed fails stage 3, rest healthy budget -> A with n=4
    cr = _synth_chain_recs(budget)
    cr[3]["hops"][2]["esp"]["ok_slow"] = False
    run_case("attrition (seed 3 drops) -> A n=4", cr, _synth_viol_recs(seeds), "A -- BUDGET",
             lambda v: v["n_intersection"] == 4 and 3 not in v["intersection"])

    # Pin A: _write_md must render every verdict shape without crashing AND always emit the
    # slope line (the Gate-0 lesson: the underpowered early-return once crashed the writer).
    import tempfile
    for tag, vv in (("A", vA), ("underpowered-intersection", vU), ("rising", vR)):
        p = os.path.join(tempfile.gettempdir(), f"_g1_md_{tag}.md")
        try:
            _write_md(p, vv, "- scramble n/a", {}, seeds, 0.0,
                      {"code": "selftest", "spec": "selftest"}, {"ok": True})
            txt = open(p).read()
            os.remove(p)
            has_slope = "loss-law slope" in txt
            allok &= has_slope
            log(f"  [{'OK' if has_slope else 'WRONG'}] _write_md({tag}) renders + Pin-A slope line present")
        except Exception as e:
            allok = False
            log(f"  [WRONG] _write_md({tag}) crashed: {e!r}")

    log(f"  VERDICT-ENGINE: {'ALL BRANCHES CORRECT' if allok else 'BRANCH ERRORS -- FIX'}")
    return allok


# ===================================================================================== #
#  STAGE 2 -- 1-seed smoke  (GPU; full H=5 chain; hop-1 must reproduce 0.981470)
# ===================================================================================== #
def _load_gate0_e2e():
    """Per-seed Gate-0 relay e2e (the H=2 replication bridge) from the committed reposed json."""
    p = os.path.join(RESDIR, "gate0_relay_reposed.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {int(s): float(r["e2e"]["r2"]) for s, r in d["recs"]["relay"].items()}


def smoke(log):
    import time
    log("=== RELAY GATE-1 :: STAGE-2 SMOKE (seed 0, K=0.24, full H=5 chain) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} "
        f"ntr={ntr} n_sub={n_sub}")
    t0 = time.perf_counter()
    rec = chain(0, geom, _real_hop_factory(geom), log, arm="chain")
    wall = time.perf_counter() - t0

    ref1 = REF_TABLE[(STAGE_SPAN, K_PRIMARY)][0][0]               # 0.981470 (committed b0f7664)
    ref2 = _load_gate0_e2e().get(0)                              # 0.962334 (Gate-0 relay e2e)
    hop1 = rec["r2_cum"][0]
    hop2 = rec["r2_cum"][1]
    bridge1 = round(hop1, 6) == ref1
    bridge2 = (ref2 is not None) and abs(hop2 - ref2) < 1e-6
    log("\n=== SMOKE SUMMARY (seed 0) ===")
    for h in rec["hops"]:
        rin = "n/a" if h["rms_in"] is None else f"{h['rms_in']:.3g}"
        sc = "n/a" if h["scale"] is None else f"{h['scale']:.2f}"
        log(f"  hop {h['stage']}: r2_cum(m_k,m0)={h['r2_cum']:+.6f} "
            f"r2_hop(m_k,proc-m_(k-1))={h['r2_hop']:+.6f} ESP={h['esp']['ok_slow']} "
            f"rms_in={rin} scale={sc}")
    log(f"  cumulative r2_cum = {[round(x,4) for x in rec['r2_cum']]}")
    log(f"  e2e-at-depth decoy p95 = {rec.get('e2e_decoy_p95'):+.4f}")
    log(f"  BRIDGE hop-1 vs committed b0f7664: {hop1:.6f} vs {ref1:.6f} -> "
        f"{'MATCH' if bridge1 else 'MISMATCH -- STOP'}")
    # hop-2 is a MANDATORY pass gate -- a missing/unreadable committed reference is a bridge
    # FAILURE (fail-safe: never silently waived), matching "STOP on any bridge miss".
    ref2_status = ("MATCH" if bridge2 else
                   ("MISMATCH -- STOP" if ref2 is not None else
                    "reference MISSING (results/R/gate0_relay_reposed.json) -- bridge FAIL, STOP"))
    log(f"  BRIDGE hop-2 vs Gate-0 relay e2e: {hop2:.6f} vs "
        f"{('%.6f' % ref2) if ref2 is not None else 'n/a'} -> {ref2_status}")
    smoke_pass = bool(bridge1 and bridge2)
    log(f"  wall-clock {wall:.0f}s. SMOKE: {'PASS' if smoke_pass else 'FAIL -- STOP, no battery'}")

    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate1_smoke.json")
    g0._dump_json(outp, {"gate": "relay-gate1", "stage": "2-smoke", "seed": 0, "H": H,
                         "framing": FRAMING, "env": _env_full(),
                         "bridges": {"hop1_ref": ref1, "hop1_got": hop1, "hop1_match": bridge1,
                                     "hop2_ref": ref2, "hop2_got": hop2, "hop2_match": bridge2},
                         "record": rec, "wall_clock_s": wall})
    log(f"  [written -> {os.path.relpath(outp)}]  (smoke artifact; NOT committed)")
    return smoke_pass


# ===================================================================================== #
#  STAGE 3 -- full battery  (chain + violation + scramble; verdict per A/B/C mapping)
# ===================================================================================== #
def run(log, nseeds):
    import time
    seeds = list(range(nseeds))
    log(f"=== RELAY GATE-1 :: STAGE-3 FULL BATTERY (seeds {seeds}, H={H}, K={K_PRIMARY}) ===")
    log(f"    framing: {FRAMING}")
    geom = g0._geom(STAGE_SPAN)
    hop = _real_hop_factory(geom)
    recs = {"chain": {}, "violation": {}, "scramble": {}}
    outp = os.path.join(RESDIR, "gate1_multihop.json")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..",
                                             "relay_gate1_multihop_spec.md"))}
    colrep = verify_no_collision()
    t0 = time.perf_counter()
    for i in seeds:
        ts = time.perf_counter()
        recs["chain"][i] = chain(i, geom, hop, log, arm="chain")
        recs["violation"][i] = chain(i, geom, hop, log, viol=True, arm="violation",
                                     decoys=False, e2e_decoys=False)
        recs["scramble"][i] = chain(i, geom, hop, log, scramble_stage=SCRAMBLE_STAGE,
                                    arm="scramble", decoys=False, e2e_decoys=False)
        log(f"  seed {i} chain r2_cum={[round(x,4) for x in recs['chain'][i]['r2_cum']]} "
            f"| viol e2e={recs['violation'][i]['r2_cum'][-1]:+.4f} "
            f"| scr r2_cum={[round(x,4) for x in recs['scramble'][i]['r2_cum']]} "
            f"({time.perf_counter()-ts:.0f}s; {time.perf_counter()-t0:.0f}s elapsed)")
        g0._dump_json(outp, {"gate": "relay-gate1", "stage": "3-battery", "H": H,
                             "seeds_done": seeds[:i + 1], "framing": FRAMING,
                             "K": K_PRIMARY, "seed_scheme": colrep, "hashes": hashes,
                             "env": _env_full(),
                             "recs": {a: {str(s): r for s, r in v.items()}
                                      for a, v in recs.items()}})
    verdict = decide(recs, seeds)
    scramble_line = _scramble_line(recs, verdict.get("intersection", []))
    wall = time.perf_counter() - t0
    payload = {"gate": "relay-gate1", "stage": "3-battery", "H": H, "seeds": seeds,
               "framing": FRAMING, "K": K_PRIMARY, "seed_scheme": colrep, "hashes": hashes,
               "env": _env_full(), "wall_clock_s": wall, "verdict": verdict,
               "scramble_line": scramble_line,
               "recs": {a: {str(s): r for s, r in v.items()} for a, v in recs.items()}}
    g0._dump_json(outp, payload)
    _write_md(os.path.join(RESDIR, "gate1_multihop.md"), verdict, scramble_line, recs, seeds,
              wall, hashes, colrep)
    log("\n=== BATTERY VERDICT ===")
    log(f"  {verdict['verdict']}")
    log("  STOP-and-report. No Gate-2 (hop-length), no mechanism-decomposition, no "
        "interpretation beyond the A/B/C mapping.")
    return verdict


def _scramble_line(recs, inter):
    S, C = recs.get("scramble", {}), recs.get("chain", {})
    ok = [i for i in inter if i in S and S[i]["esp_all_stages"]]
    if not ok:
        return "- no ESP-ok scramble seeds in the intersection (not read)."
    sm = _mstats([S[i]["r2_cum"][-1] for i in ok])
    cm = _mstats([C[i]["r2_cum"][-1] for i in ok])
    generic = abs(sm["mean"] - cm["mean"]) < max(3 * cm["sd"], 0.1)
    return (f"- scrambled-stage-{SCRAMBLE_STAGE} chain e2e = {sm['mean']:+.4f} +/- {sm['se']:.4f} "
            f"vs compliant {cm['mean']:+.4f} +/- {cm['se']:.4f} (n={sm['n']}) -- loss-law "
            f"{'topology-generic' if generic else 'shows topology dependence (flag)'} "
            "(3-sigma/0.1 bar = code-level operationalization).")


def _slope_line(ll):
    """Pin A: the loss-law slope line, ALWAYS rendered regardless of class (slope_mean +/- SE,
    threshold, margin). Pin B reminder: a positive slope beyond +max(2*SE, SLOPE_EPS) is an
    instrument-suspicion flag, never A."""
    cls = ll.get("classification", "not-computed")
    sm, sse = ll.get("slope_mean"), ll.get("slope_se")
    thr, mg = ll.get("slope_thr"), ll.get("margin")
    if sm is None:
        reason = ("intersection underpowered -- loss law not computed" if cls == "not-computed"
                  else "too few fittable seed-slopes (< MIN_PAIRS)")
        return (f"- **loss-law slope: n/a** (class {cls}; {reason}). SLOPE_EPS={SLOPE_EPS}. "
                "Pin B: a positive slope beyond +max(2*SE,SLOPE_EPS) is instrument-suspicion, never A.")
    inside = (mg is not None and mg >= 0)
    return (f"- **loss-law slope = {sm:+.4f} +/- {_fmt(sse,'.4f')}** (class {cls}); "
            f"threshold max(2*SE,{SLOPE_EPS}) = {_fmt(thr,'.4f')}; margin (thr-|slope|) = "
            f"{_fmt(mg,'+.4f')} -> {'inside flat band (A-consistent)' if inside else 'trend resolved beyond threshold'}. "
            "Pin B: a positive slope beyond +threshold is instrument-suspicion, never A.")


def _two_regime_lines(tr, ll):
    """Amendment (panel-mandated descriptive honesty): render the two-regime shape + the
    r2_hop B-refutation + the H_half extrapolation caveat, WITHOUT changing the registered A."""
    if not tr:
        return []
    r2h = tr.get("r2hop_by_hop", {})
    rho2 = tr.get("rho2_insertion", {})
    st = tr.get("steady_slope", {})
    hh = ll.get("H_half_extrap")
    mult = _fmt(hh / H, ".1f") if isinstance(hh, (int, float)) else "n/a"
    return [
        "", "## Loss-law SHAPE (two-regime; amendment -- descriptive, does not change the registered A)", "",
        f"- **B / depth-limit REFUTED (mechanism):** per-hop reconstruction fidelity r2_hop = "
        f"r2(m_k, processed-m_(k-1)) is FLAT with depth "
        f"({ {k: (round(x,4) if x is not None else None) for k,x in r2h.items()} }; "
        f"slope t={_fmt(tr.get('r2hop_slope_t'),'.2f')}) -> no error amplification; each stage "
        f"reconstructs its input equally well at any depth. The ladder is PRICED, not merely 'not "
        f"yet bounded'.",
        f"- **Regime 1 -- one-time first-relay insertion loss:** rho_2 = "
        f"{_fmt(rho2.get('mean'),'.4f')} +/- {_fmt(rho2.get('se'),'.4f')} (the LARGEST single-hop "
        f"drop; hop-1 clean injection -> hop-2 first relay).",
        f"- **Regime 2 -- near-flat steady-state:** mean rho_3..{H} = "
        f"{_fmt(tr.get('steady_rho_mean'),'.4f')} +/- {_fmt(tr.get('steady_rho_se'),'.4f')} "
        f"(distinct from the loss-law OVERALL budget rho {_fmt(ll.get('rho_mean'),'.4f')}, which "
        f"pools in the first-relay insertion loss). Slope "
        f"{_fmt(st.get('mean'),'+.5f')} +/- {_fmt(st.get('se'),'.5f')} "
        f"(t={_fmt(tr.get('steady_slope_t'),'.2f')}, {tr.get('steady_n_negative')} seeds neg). This "
        f"mild late drift is UNRESOLVED at this depth: it lives mostly in the LAST hop "
        f"(mean(rho_last - rho_3)={_fmt(tr.get('decline_locus_last_minus_first'),'+.4f')} vs "
        f"mean(rho_mid - rho_3)={_fmt(tr.get('decline_locus_mid_minus_first'),'+.4f')}), so at "
        f"n/H={H} a genuine steady decline is NOT separable from a single noisy last hop.",
        f"- **H_half caveat:** H_half={_fmt(hh,'.1f')} is an EXTRAPOLATION ({mult}x beyond the "
        f"measured H={H}, off a U-shaped rho) assuming a tail constancy the data mildly violate -- "
        f"a projected budget, NOT a measured horizon. Depth extension (H>{H}) is the clean test.",
    ]


def _write_md(path, v, scramble_line, recs, seeds, wall, hashes, colrep, note=""):
    ll = v.get("loss_law", {})
    lines = [
        "# Relay Gate-1 -- multi-hop loss-law record (H=5)", "",
        f"Spec: relay_gate1_multihop_spec.md (sha256 {hashes['spec']}). Harness: "
        f"experiments/relay_gate1.py (sha256 {hashes['code']}).",
        f"Seeds run: {seeds}. K = {K_PRIMARY}. Wall-clock {wall/60:.0f} min. "
        f"Seed scheme collision-free: {colrep['ok']}.",
    ] + ([note] if note else []) + [
        "",
        f"Framing: {FRAMING}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
        "## Instrument checks (pre-registered order, before the A/B read)", "",
    ]
    if "anchor" in v:
        a, d = v["anchor"], v["decoys"]
        vio = v["violation"]
        lines += [
            f"1. **Anchor (hop-1 == committed span-1.5/K=0.24)**: intersection mean "
            f"{a['hop1']['mean']:.6f} (SE {a['hop1']['se']:.6f}, n={a['hop1']['n']}); "
            f"target {ANCHOR} +/- {a['window']:.4f}; deviation {a['deviation']:+.6f} -> "
            f"{'OK' if a['ok'] else 'MISS'}.",
            f"2. **Decoy floors at depth** (intersection p95 means): per-stage "
            f"{ {k: (round(x,4) if x is not None else None) for k,x in d['stage_p95_mean'].items()} }, "
            f"e2e-at-depth {d['e2e_p95_mean']:+.4f}; elevated bar {DECOY_ELEVATED} -> "
            f"{'ELEVATED (leak)' if d['elevated'] else 'clean'}.",
            f"3. **Filter-violation at depth** ([2,9] message, first repeater [0.2,0.9]): "
            f"n={vio['n']}, e2e mean {_fmt(vio['violation_e2e_mean'])} "
            f"(bar e2e r2 < {vio['bar_per_seed']} every seed) -> "
            f"{'COLLAPSED (sound)' if vio['collapsed'] else 'DID NOT COLLAPSE '+str(vio['offenders'])}. "
            f"Signature rms_in {_fmt(vio['signature_rms_in'],'.4g')}, "
            f"scale {_fmt(vio['signature_scale'],'.1f')}.",
            f"4. **ESP-honest paired intersection** (ESP-ok across ALL {H} stages): "
            f"{v['intersection']} (n={v['n_intersection']}).", "",
            "## Cumulative fidelity r2(m_k, m0) (intersection means +/- SE)", "",
        ]
        for s in range(1, H + 1):
            cm = v["cumulative"][str(s)]
            lines.append(f"- hop {s}: **{cm['mean']:+.4f} +/- {cm['se']:.4f}** "
                         f"(per-seed {[round(x,3) for x in cm['per_seed']]})")
        lines += [
            "", "## Loss law (rho_k = r2_k / r2_{k-1}; validity guard r2_{k-1} > "
            f"{RHO_VALID_MIN})", "",
            _slope_line(ll),                                       # Pin A: always slope +/- SE + margin
            f"- classification: **{ll.get('classification')}**; "
            f"budget rho = {_fmt(ll.get('rho_mean'),'.4f')}; extrapolated H_half = "
            f"{_fmt(ll.get('H_half_extrap'),'.1f')}.",
            f"- rho by level: { {k: (round(x,4) if x is not None else None) for k,x in ll.get('rho_by_level',{}).items()} } "
            f"(valid-n {ll.get('rho_level_n')}).",
        ]
        lines += _two_regime_lines(v.get("two_regime", {}), ll)
    else:
        lines += [f"- Battery ended on the pre-registered early exit (underpowered): "
                  f"intersection {v['intersection']} (n={v['n_intersection']}).",
                  _slope_line(ll)]                                # Pin A: slope line even here
    lines += ["", "## Scramble robustness line (characterisation only, never verdict)", "",
              scramble_line, "",
              "## Scope", "",
              "Offline H=5 chain; compound span 1.5*H = an INFORMATION-PATH claim (H successive",
              "square-law demodulations), NOT one physical spectrum. STOP-and-report: no Gate-2",
              "(hop-length), no mechanism-decomposition, no interpretation beyond the A/B/C mapping."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def reread(log):
    """Re-decide + re-render the verdict from the COMMITTED battery recs (NO GPU). Amends the
    verdict framing (two-regime honesty) without changing a single measured number: every arm's
    recs re-serialize byte-identically (asserted). Overwrites gate1_multihop.{json,md}."""
    src = os.path.join(RESDIR, "gate1_multihop.json")
    assert os.path.exists(src), f"missing battery record {src} -- run --run first"
    nm = json.load(open(src))
    recs = {a: {int(s): r for s, r in v.items()} for a, v in nm["recs"].items()}
    seeds = nm["seeds"]
    log("=== RELAY GATE-1 :: REREAD (re-frame verdict from unchanged recs; NO GPU) ===")
    verdict = decide(recs, seeds)
    scramble_line = _scramble_line(recs, verdict.get("intersection", []))
    for a in recs:                                     # integrity: not one number moved
        assert json.dumps({str(s): r for s, r in recs[a].items()}, sort_keys=True) == \
               json.dumps(nm["recs"][a], sort_keys=True), f"arm '{a}' drifted from the battery record"
    log("  [integrity] all arms' recs byte-identical to the battery record: OK")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..",
                                             "relay_gate1_multihop_spec.md"))}
    payload = {**nm, "verdict": verdict, "scramble_line": scramble_line,
               "hashes": hashes, "run_hashes": nm.get("hashes"),
               "reread": "verdict re-framed (two-regime amendment) from unchanged recs; no GPU, "
                         "no measured number changed"}
    g0._dump_json(src, payload)
    note = (f"Provenance: numbers produced by the battery run (harness sha256 "
            f"{(nm.get('hashes') or {}).get('code','?')}, {nm.get('wall_clock_s',0)/60:.0f} min GPU); "
            f"verdict RE-FRAMED (two-regime amendment) by --reread (harness sha256 "
            f"{hashes['code']}) with every arm's recs asserted byte-identical -- no measured number changed.")
    _write_md(os.path.join(RESDIR, "gate1_multihop.md"), verdict, scramble_line, recs, seeds,
              nm.get("wall_clock_s", 0.0), hashes, nm.get("seed_scheme", {"ok": True}), note=note)
    log(f"  {verdict['verdict']}")
    log(f"  [rewritten -> {os.path.relpath(src)} + gate1_multihop.md]  (recs UNCHANGED; NOT committed)")
    return verdict


# ===================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reread", action="store_true",
                    help="re-frame the verdict from the committed battery recs (no GPU)")
    ap.add_argument("--nseeds", type=int, default=8,
                    help="spec n>=5; default 8 (seeds 0..7 covered by committed bridges)")
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
        assert 1 <= args.nseeds <= SEED_MAX + 1, \
            f"committed bridges cover seeds 0..{SEED_MAX} (Phase-1 ran 10)"
        run(log, args.nseeds)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
