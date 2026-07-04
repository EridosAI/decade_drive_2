"""
experiments/relay_gate0.py
==========================

Relay Gate-0: offline two-stage square-law relay probe.
Per the committed relay_gate0_spec.md. Built STRICTLY on the byte-identical Phase-1
machinery (imported from D_phase1_routing); core/integrator_corotating.py is NOT touched.

Question: does an offline two-stage square-law relay -- stage A's slow-band |z|^2
reconstruction, band-limited + re-injected as stage B's message -- transfer usable
information across a compound 3-decade information path where direct passive transfer is
dead (committed Phase-1 direct span-3.0 r2 = -0.003, ESP-robust @ K=0.24)?

Modes:
  --sandbox : Stage 1. CPU-ONLY load-bearing checks (no GPU, no seeds-at-scale):
              (1) repeater transform, (2) decoy-protocol match to Phase-1,
              (3) paired-intersection logic, (4) violation wiring.
  --smoke   : Stage 2 (separate go). 1-seed integration smoke.
  --run     : Stage 3 (separate go). Full gate battery.

STOP-and-report after --sandbox. Nothing committed.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

# CPU-only sandbox: force the JAX CPU backend BEFORE jax is imported (via D_phase1_routing),
# so a Stage-1 sandbox run never touches the GPU. --smoke/--run leave the default backend.
if "--sandbox" in sys.argv or "--verdict-test" in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))                        # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root

import numpy as np                                                   # noqa: E402
import D_phase1_routing as p1                                        # noqa: E402  (jax x64 set on import)
from core.reservoir import build_system                             # noqa: E402
from core.bands import band_indices, masked_encoding, band_summary  # noqa: E402
from core.consistency import replica_spec, consistency_distance, ESP_EPS  # noqa: E402

RESDIR = os.path.join(os.path.dirname(__file__), "..", "results", "R")

# --- reconciled from the committed spec + Phase-1 (imported) --------------------------- #
ANCHOR = 0.986              # committed Phase-1 span-1.5 @ K=0.24 (b0f7664)
ANCHOR_SE_K = 2.0           # anchor window: |mean - ANCHOR| <= max(ANCHOR_SE_K*SE, ANCHOR_FLOOR)
ANCHOR_FLOOR = 0.02         # floor keeps the window clear of 0.945 (the K=0.16 value at span 1.5)
STAGE_SPAN = 1.5            # each hop spans 1.5 decades
DIRECT_SPAN = 3.0           # the paired direct baseline (compound-span comparison target)
K_PRIMARY = 0.24            # verdict row (ESP-robust); single-variable
K_SECONDARY = 0.16          # optional bracket (K_OP)
VIOL_LO, VIOL_HI = 2.0, 9.0  # bandwidth-violation message band (10x standard; at/above omega_min=1)
MIN_PAIRS = 2              # >=2 seeds in the ESP-honest intersection or the gate is underpowered

MSG_LO, MSG_HI = p1.MSG_LO, p1.MSG_HI     # [0.2, 0.9] rad/s standard message band
N_DEC = p1.N_DEC


# ===================================================================================== #
#  Repeater transform  F  (the architecture's detect-filter-remodulate, made explicit)
# ===================================================================================== #
def bandlimit(x, dt_in, w_lo, w_hi):
    """Zero-phase brick-wall band-pass to angular band [w_lo, w_hi] rad/s (an auditable F:
    zero every rFFT bin outside the band). DC (w=0) is outside [0.2,0.9] so the result is
    zero-mean by construction."""
    x = np.asarray(x, float)
    L = len(x)
    X = np.fft.rfft(x)
    w = 2.0 * np.pi * np.fft.rfftfreq(L, d=dt_in)      # angular frequency per bin (rad/s)
    keep = (w >= w_lo) & (w <= w_hi)
    return np.fft.irfft(X * keep, n=L)


def repeater_transform(m1, m0, dt_in, w_lo=MSG_LO, w_hi=MSG_HI):
    """F(m1): band-limit m1 to MSG_BAND, then affine-rescale to the m0 message class
    (zero-mean, RMS matched to m0's zero-mean fluctuation). Returns (out, params).
    The re-added message-class DC for stage-B injection lives in remodulate_for_stage_b."""
    bl = bandlimit(m1, dt_in, w_lo, w_hi)
    bl = bl - bl.mean()                                # enforce exact zero-mean
    rms_in = float(bl.std())
    rms_target = float((np.asarray(m0, float) - np.mean(m0)).std())   # m0-class fluctuation RMS
    scale = rms_target / (rms_in + 1e-300)
    out = bl * scale
    # out-of-band residual: fraction of OUT's spectral power outside [w_lo, w_hi] (should be ~0)
    P = np.abs(np.fft.rfft(out)) ** 2
    w = 2.0 * np.pi * np.fft.rfftfreq(len(out), d=dt_in)
    inb = (w >= w_lo) & (w <= w_hi)
    oob_resid = float(P[~inb].sum() / (P.sum() + 1e-300))
    params = {"msg_band": [w_lo, w_hi], "filter": "zero-phase brick-wall rFFT band-pass",
              "rms_in": rms_in, "rms_target": rms_target, "scale": scale,
              "mean_out": float(out.mean()), "rms_out": float(out.std()),
              "oob_residual_frac": oob_resid}
    return out, params


def remodulate_for_stage_b(processed, m0, floor=1e-6):
    """Re-add the m0 message-class DC so stage B has a valid (>=0) AM message s_B; the
    information rides in the zero-mean fluctuation `processed`. (Called by smoke()'s
    stage-B path; the DC and the repeater's RMS target are the ONLY two m0 scalars that
    reach stage B -- the spec's message-class statistics.)"""
    s_b = float(np.mean(m0)) + np.asarray(processed, float)
    return np.clip(s_b, floor, None)


# ===================================================================================== #
#  Demod reconstruction  (fit byte-identical to p1.demod_capacity; returns the message m1)
# ===================================================================================== #
def demod_fit(X, idx, s_real, sl, mode="full"):
    """Replicates p1.demod_capacity's delay-0 fit EXACTLY (bias-protected standardize,
    inner-validation lambda, ridge G) and returns (m1 reconstruction over the eval window,
    r2_d0, lambda). r2_d0 is computed identically to demod_capacity -> must match it."""
    iw = np.arange(sl.start, sl.stop)
    F = p1.band_features(X, idx, mode)[iw]
    ntr = int(p1.TRAIN_FRAC * F.shape[0])
    Ftr, Fte = F[:ntr], F[ntr:]
    mu, sd = Ftr.mean(0), Ftr.std(0)
    const = sd < 1e-7
    sd = sd + 1e-8
    sd[const] = 1.0
    mu[const] = 0.0
    Ftr_s, Fte_s, Fall_s = (Ftr - mu) / sd, (Fte - mu) / sd, (F - mu) / sd
    Fp = Ftr_s.shape[1]
    y0 = s_real[iw][:ntr]
    nval = max(1, int(0.2 * ntr))
    inn, va = slice(0, ntr - nval), slice(ntr - nval, ntr)
    GTG, FTy = Ftr_s[inn].T @ Ftr_s[inn], Ftr_s[inn].T @ y0[inn]
    best_lam, best = p1.ENV_LAMS[0], np.inf
    for lam in p1.ENV_LAMS:
        W = np.linalg.solve(GTG + lam * np.eye(Fp), FTy)
        err = np.mean((Ftr_s[va] @ W - y0[va]) ** 2) / (np.var(y0[va]) + 1e-12)
        if err < best:
            best, best_lam = err, lam
    G = np.linalg.inv(Ftr_s.T @ Ftr_s + best_lam * np.eye(Fp))
    W = G @ (Ftr_s.T @ s_real[iw][:ntr])               # == demod_capacity's delay-0 W
    r2_d0 = p1.r2_det(Fte_s @ W, s_real[iw][ntr:])     # == demod_capacity real[0]
    m1 = Fall_s @ W                                    # reconstruction over the full eval window
    return m1, float(r2_d0), float(best_lam)


# ===================================================================================== #
#  Decoy construction  (byte-identical protocol to Phase-1 D_phase1_routing.run)
# ===================================================================================== #
def stage_decoys(stage, seed_i, L, dt_in):
    """Per-stage decoy list, using Phase-1's EXACT construction (p1.slow_bandlimited, same
    band, N_DEC draws). Stage A reproduces Phase-1's seeds verbatim (base 40000); stage B
    uses the identical protocol at an independent base (60000 -- OUTSIDE the Phase-1 decoy
    seed range 40000..41859, so no collision at any seed count; review finding)."""
    base = {"A": 40000, "B": 60000}[stage]
    return [p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=base + seed_i * 200 + d)
            for d in range(N_DEC)]


def phase1_decoys_ref(seed_i, L, dt_in):
    """Verbatim copy of D_phase1_routing.run's decoy line -- the reference to diff against."""
    return [p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=40000 + seed_i * 200 + d)
            for d in range(N_DEC)]


# ===================================================================================== #
#  Paired-intersection logic  (ESP-honest; nested esp -> ok_slow read)
# ===================================================================================== #
def esp_ok_slow(rec):
    """Read the Phase-1 nested ESP flag (esp -> ok_slow), NOT a flat rec['ok_slow']."""
    return bool(rec["esp"]["ok_slow"])


def paired_intersection(esp_table, conds):
    """esp_table: {seed: {cond: bool}}. Return sorted seeds ESP-ok across EVERY cond in
    `conds` (a seed failing ANY compared condition is dropped from ALL deltas)."""
    return sorted(s for s, flags in esp_table.items()
                  if all(flags.get(c, False) for c in conds))


def intersection_status(esp_table, conds, min_pairs=MIN_PAIRS):
    inter = paired_intersection(esp_table, conds)
    dropped = {s: [c for c in conds if not esp_table[s].get(c, False)]
               for s in esp_table if s not in inter}
    return {"intersection": inter, "n": len(inter),
            "underpowered": len(inter) < min_pairs,
            "dropped": dropped}


# ===================================================================================== #
#  Violation wiring  (message at/above the slow tertile's natural-frequency floor)
# ===================================================================================== #
def am_input_band(L, dt_in, seed, w_lo, w_hi):
    """am_input generalized to an arbitrary message band. w_lo=MSG_LO,w_hi=MSG_HI reproduces
    p1.am_input exactly (same slow_bandlimited + same Rademacher carrier seed offset)."""
    s = p1.slow_bandlimited(L, dt_in, w_lo, w_hi, seed=seed)
    w = np.random.default_rng(seed + 777).choice([-1.0, 1.0], size=L)
    return s, 0.5 * np.sqrt(s) * w


def slow_tertile_omega(span=STAGE_SPAN):
    """[min,max] natural frequency of the slow tertile at `span` (CPU: omega + partition
    only, no integration)."""
    om = build_system(0, p1.N, span).omega
    sl_idx = band_indices(om)["slow"]
    return float(om[sl_idx].min()), float(om[sl_idx].max())


def dominant_band(sig, dt_in):
    """[approx] angular-frequency support of a signal: the 5th/95th percentile of spectral
    power mass. For reporting that message content sits in the intended band."""
    P = np.abs(np.fft.rfft(sig - np.mean(sig))) ** 2
    w = 2.0 * np.pi * np.fft.rfftfreq(len(sig), d=dt_in)
    c = np.cumsum(P) / (P.sum() + 1e-300)
    lo = float(w[np.searchsorted(c, 0.05)])
    hi = float(w[min(np.searchsorted(c, 0.95), len(w) - 1)])
    return lo, hi


# ===================================================================================== #
#  STAGE 1 -- CPU sandbox
# ===================================================================================== #
def _check(log, name, passed, detail):
    log(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return bool(passed)


def sandbox(log):
    log("=== RELAY GATE-0 :: STAGE-1 CPU SANDBOX (no GPU, no seeds-at-scale) ===")
    log(f"    backend: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS','<default>')} "
        f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}'")
    # a small stage-A-like window (Phase-1 am_window at the stage span; small n_msg for speed)
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(STAGE_SPAN, n_msg=8)
    sl = slice(eval_start, L)
    log(f"    window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} "
        f"delays={delays}")
    results = {}

    # ---- CHECK 1: repeater transform -------------------------------------------------- #
    log("\n(1) Repeater transform -- band-limit to MSG_BAND [0.2,0.9] + affine rescale")
    t = (np.arange(L) + 1.0) * dt_in
    # synthetic m1: DC + in-band tone (0.5) + OOB-low tone (0.05) + OOB-high tone (5.0)
    A_in, A_lo, A_hi, DC = 1.0, 0.7, 0.5, 0.3
    m1 = DC + A_in * np.sin(0.5 * t) + A_lo * np.sin(0.05 * t) + A_hi * np.sin(5.0 * t)
    m0 = p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=123)     # a real message-class signal
    out, params = repeater_transform(m1, m0, dt_in)

    def tone_amp(sig, f):                                            # amplitude at angular freq f
        c, s = np.cos(f * t), np.sin(f * t)
        return 2.0 * np.hypot((sig * c).mean(), (sig * s).mean())
    a_in_out, a_lo_out, a_hi_out = tone_amp(out, 0.5), tone_amp(out, 0.05), tone_amp(out, 5.0)
    a_in_in = tone_amp(m1, 0.5)
    zero_mean = abs(out.mean()) < 1e-9 * (out.std() + 1e-300)
    rms_match = abs(out.std() - params["rms_target"]) / (params["rms_target"] + 1e-300) < 1e-9
    oob_rej = (a_lo_out < 0.01 * A_lo) and (a_hi_out < 0.01 * A_hi) and params["oob_residual_frac"] < 1e-6
    in_kept = a_in_out > 0.5 * (a_in_in * params["scale"])          # in-band tone survives (scaled)
    c1 = all([
        _check(log, "out-of-band rejection",
               oob_rej, f"OOB-low amp {a_lo_out:.2e} (in {A_lo}), OOB-high {a_hi_out:.2e} "
               f"(in {A_hi}), OOB spectral residual {params['oob_residual_frac']:.2e}"),
        _check(log, "in-band content preserved",
               in_kept, f"0.5 rad/s tone amp {a_in_out:.4f} present (scaled input "
               f"{a_in_in*params['scale']:.4f})"),
        _check(log, "affine rescale -> zero-mean",
               zero_mean, f"mean_out={out.mean():.2e}"),
        _check(log, "affine rescale -> RMS match to m0",
               rms_match, f"rms_out={out.std():.6f} vs rms_target(m0 fluct)={params['rms_target']:.6f}"),
    ])
    log(f"    transform params (logged to json): {json.dumps(params)}")
    results["check1_repeater"] = {"pass": c1, "params": params,
                                  "oob_low_amp": a_lo_out, "oob_high_amp": a_hi_out,
                                  "in_band_amp": a_in_out}

    # ---- CHECK 2: decoy protocol match (both stages) ---------------------------------- #
    log("\n(2) Decoy protocol -- diff vs the Phase-1 decoy path (both stages)")
    seed_i = 3
    ref = phase1_decoys_ref(seed_i, L, dt_in)                        # Phase-1's exact construction
    a_dec = stage_decoys("A", seed_i, L, dt_in)
    b_dec = stage_decoys("B", seed_i, L, dt_in)
    diff_A = max(float(np.max(np.abs(np.asarray(a) - np.asarray(r)))) for a, r in zip(a_dec, ref))
    # stage A must be byte-identical to Phase-1; stage B uses the identical protocol (offset seed)
    b_is_slowbl = all(np.allclose(b, p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                                  seed=60000 + seed_i * 200 + d)) for d, b in enumerate(b_dec))
    n_match = (len(a_dec) == N_DEC == len(b_dec))
    # scoring path identity: the gate scores decoys via the IMPORTED p1.demod_capacity
    scorer_is_phase1 = (demod_fit.__module__ != p1.demod_capacity.__module__) and \
                       (p1.demod_capacity.__module__ == "D_phase1_routing")
    # and the delay-0 fit that produces m1 reproduces demod_capacity's r2_d0 exactly
    rng = np.random.default_rng(7)
    Nc = 24
    sp_small = build_system(0, Nc, STAGE_SPAN)
    bands_s = band_indices(sp_small.omega)
    s_syn, _ = am_input_band(L, dt_in, 55, MSG_LO, MSG_HI)
    Xsyn = (rng.standard_normal((L, Nc)) + 1j * rng.standard_normal((L, Nc)))
    Xsyn[:, bands_s["slow"]] += (np.sqrt(np.clip(s_syn, 0, None))[:, None]
                                 * np.exp(1j * rng.uniform(0, 2 * np.pi, len(bands_s["slow"]))))
    dc = phase1_decoys_ref(0, L, dt_in)
    r2_ref = p1.demod_capacity(Xsyn, bands_s["slow"], s_syn, dc, delays, sl, "full")["r2_d0"]
    _, r2_mine, _ = demod_fit(Xsyn, bands_s["slow"], s_syn, sl, "full")
    fit_match = abs(r2_ref - r2_mine) < 1e-10
    c2 = all([
        _check(log, "stage-A decoys byte-identical to Phase-1",
               diff_A == 0.0 and n_match, f"max|diff|={diff_A:.1e} over {N_DEC} draws"),
        _check(log, "stage-B decoys use the identical protocol (offset seed)",
               b_is_slowbl, "p1.slow_bandlimited, same band [0.2,0.9], N_DEC draws, base 60000 "
               "(outside Phase-1's 40000..41859 -- no collision at any seed count)"),
        _check(log, "decoy scoring is the imported p1.demod_capacity",
               scorer_is_phase1, f"scorer module = {p1.demod_capacity.__module__}"),
        _check(log, "m1-producing fit reproduces demod_capacity r2_d0 exactly",
               fit_match, f"demod_capacity={r2_ref:.10f} vs demod_fit={r2_mine:.10f} "
               f"(|diff|={abs(r2_ref-r2_mine):.1e})"),
    ])
    results["check2_decoy"] = {"pass": c2, "diff_A": diff_A, "fit_match_r2": [r2_ref, r2_mine]}

    # ---- CHECK 3: paired-intersection logic ------------------------------------------- #
    log("\n(3) Paired-intersection -- ESP-honest, incl. asymmetric attrition")
    conds = ["stageA", "stageB", "direct"]
    # build recs with the NESTED Phase-1 esp structure, then derive the flag table from them
    def rec(ok):  # a minimal Phase-1-shaped rec carrying esp->ok_slow
        return {"esp": {"d_slow": 0.0 if ok else 1.0, "ok_slow": ok}}
    recs = {  # seed -> cond -> rec        (the asymmetric-attrition cases are 1,2,3)
        0: {"stageA": rec(True),  "stageB": rec(True),  "direct": rec(True)},   # keep
        1: {"stageA": rec(True),  "stageB": rec(False), "direct": rec(True)},   # drop: fails B
        2: {"stageA": rec(True),  "stageB": rec(True),  "direct": rec(False)},  # drop: fails direct
        3: {"stageA": rec(False), "stageB": rec(True),  "direct": rec(True)},   # drop: fails A
        4: {"stageA": rec(True),  "stageB": rec(True),  "direct": rec(True)},   # keep
        5: {"stageA": rec(False), "stageB": rec(False), "direct": rec(False)},  # drop: fails all
    }
    table = {s: {c: esp_ok_slow(recs[s][c]) for c in conds} for s in recs}
    st = intersection_status(table, conds)
    nested_ok = esp_ok_slow(recs[0]["stageA"]) is True and esp_ok_slow(recs[3]["stageA"]) is False
    # a seed dropped for ANY condition must appear in NO pairwise delta (i.e. not in intersection)
    all_dropped_everywhere = all(s not in st["intersection"] for s in (1, 2, 3, 5))
    # underpowered flag: a table with intersection<MIN_PAIRS must flag
    thin = {0: {c: True for c in conds}, 1: {"stageA": True, "stageB": False, "direct": True}}
    thin_status = intersection_status(thin, conds)
    c3 = all([
        _check(log, "nested esp->ok_slow read (not flat)",
               nested_ok, "esp_ok_slow reads rec['esp']['ok_slow']"),
        _check(log, "intersection = seeds ok across ALL conds",
               st["intersection"] == [0, 4], f"intersection={st['intersection']}"),
        _check(log, "asymmetric attrition drops seed everywhere",
               all_dropped_everywhere, f"dropped {sorted(st['dropped'])} "
               f"(reasons: {{s: {{c}} }} -> {st['dropped']})"),
        _check(log, "underpowered flag fires when intersection < MIN_PAIRS",
               thin_status["underpowered"] and st["underpowered"] is False,
               f"thin n={thin_status['n']} -> underpowered={thin_status['underpowered']}; "
               f"main n={st['n']} -> underpowered={st['underpowered']}"),
    ])
    results["check3_intersection"] = {"pass": c3, "intersection": st["intersection"],
                                      "dropped": {str(k): v for k, v in st["dropped"].items()}}

    # ---- CHECK 4: violation wiring ---------------------------------------------------- #
    log("\n(4) Violation wiring -- MSG_BAND [2,9] rad/s flows through injection")
    s_std, u_std = am_input_band(L, dt_in, 9, MSG_LO, MSG_HI)
    s_vio, u_vio = am_input_band(L, dt_in, 9, VIOL_LO, VIOL_HI)
    # the message rides in the injected POWER u^2 (= 0.25*s exactly); check its band support
    std_band = dominant_band(u_std ** 2, dt_in)
    vio_band = dominant_band(u_vio ** 2, dt_in)
    om_lo, om_hi = slow_tertile_omega(STAGE_SPAN)                    # slow-tertile omega [min,max]
    # standard is sub-omega_min (trackable); violation is at/above omega_min=1 (untrackable)
    std_sub = MSG_HI < om_lo
    vio_above = VIOL_LO >= om_lo
    vio_in_band = (vio_band[0] >= VIOL_LO * 0.5) and (vio_band[1] <= VIOL_HI * 1.5)
    flows = np.isfinite(u_vio).all() and (u_vio.shape == u_std.shape == (L,))
    log(f"    slow-tertile omega range @span {STAGE_SPAN}: [{om_lo:.3f}, {om_hi:.3f}] "
        f"(omega_min={om_lo:.3f})")
    log(f"    injected-power band support: standard u^2 ~ {std_band[0]:.2f}-{std_band[1]:.2f} "
        f"rad/s | violation u^2 ~ {vio_band[0]:.2f}-{vio_band[1]:.2f} rad/s")
    c4 = all([
        _check(log, "violation message flows through the AM injection",
               flows, f"u_vio shape {u_vio.shape}, finite, = 0.5*sqrt(s)*w in [{VIOL_LO},{VIOL_HI}]"),
        _check(log, "violation content sits AT/ABOVE the slow-tertile floor",
               vio_above and vio_in_band, f"VIOL_LO={VIOL_LO} >= omega_min={om_lo:.3f}; "
               f"power support {vio_band[0]:.2f}-{vio_band[1]:.2f} rad/s"),
        _check(log, "standard message is sub-omega_min (trackable) -- contrast",
               std_sub, f"MSG_HI={MSG_HI} < omega_min={om_lo:.3f}"),
    ])
    viol_log = {"violation_band": [VIOL_LO, VIOL_HI], "standard_band": [MSG_LO, MSG_HI],
                "slow_tertile_omega": [om_lo, om_hi],
                "violation_power_support": list(vio_band), "standard_power_support": list(std_band)}
    log(f"    json log block (check 4): {json.dumps(viol_log)}")
    results["check4_violation"] = {"pass": c4, **viol_log}

    # ---- summary + write --------------------------------------------------------------- #
    order = ["check1_repeater", "check2_decoy", "check3_intersection", "check4_violation"]
    allpass = all(results[k]["pass"] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if results[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate0_sandbox.json")
    with open(outp, "w") as f:
        json.dump({"stage": "1-cpu-sandbox", "all_pass": allpass,
                   "window": {"span": STAGE_SPAN, "dt_in": dt_in, "L": L,
                              "eval_start": eval_start, "delays": delays},
                   "checks": results}, f, indent=1)
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


# ===================================================================================== #
#  STAGE 2 -- 1-seed smoke  (GPU; full end-to-end plumbing; anchor-neighborhood gated)
# ===================================================================================== #
# Smoke pass criterion (Stage-2 brief): plumbing completes, per-stage ESP flags recorded,
# and stage-A r2(m1,m0) lands in the committed per-seed NEIGHBORHOOD below. The n>=5 anchor
# rule (|mean-0.986| <= max(2*SE, 0.02)) evaluates only at the full gate.
SMOKE_LO, SMOKE_HI = 0.975, 0.995
# Committed per-seed references (decade_drive b0f7664, results/D/phase1_routing.json):
REF_SEED0_STAGE = 0.981470     # span 1.5, K=0.24, seed 0 (ESP ok, lam=1e-4)
REF_SEED0_DIRECT = 0.000724    # span 3.0, K=0.24, seed 0 (ESP ok)

# Stage-B seed scheme: same construction as stage A, INDEPENDENT network within pair i
# (spec: "fresh network, independent seed"). Stage A uses Phase-1's exact bases
# (build i / enc 5000+i / msg 1000+i / rep 9000+i); stage B offsets every base.
STAGE_B_NET_OFF = 100        # build_system(STAGE_B_NET_OFF + i)
STAGE_B_ENC_OFF = 5100       # masked-encoding rng base
STAGE_B_REP_OFF = 9100       # ESP-replica base
STAGE_B_CAR_OFF = 2000       # Rademacher carrier base (stage A's carrier = (1000+i)+777)


def _env_versions():
    """jax/jaxlib/backend/device provenance, logged so a last-decimal anchor discrepancy is
    attributable to a library/kernel change rather than a protocol bug (review finding)."""
    import jax
    try:
        import jaxlib
        jl = jaxlib.__version__
    except Exception:
        jl = "unknown"
    try:
        dev = jax.devices()[0]
        devk = f"{dev.platform}:{getattr(dev, 'device_kind', '?')}"
    except Exception:
        devk = "unknown"
    return {"jax": jax.__version__, "jaxlib": jl, "backend": jax.default_backend(),
            "device": devk}


def am_from_message(s, seed):
    """u = 0.5*sqrt(s)*w for a GIVEN message s (stage-B injection). Same carrier
    construction as p1.am_input (Rademacher from seed+777); s must be >= 0."""
    w = np.random.default_rng(seed + 777).choice([-1.0, 1.0], size=len(s))
    return 0.5 * np.sqrt(np.asarray(s, float)) * w


def _hop(spec, enc_seed, rep_seed, s_target, u_in, K, dt_in, n_sub, delays, sl,
         decoys=None, Ks=None, L_override=None):
    """One integration hop (main + ESP replica) + slow-band reconstruction of s_target.
    Byte-identical machinery: p1.integrate_Ks / replica_spec / consistency_distance /
    demod_fit (== p1.demod_capacity's delay-0 fit; sandbox check 2). Returns
    (m_rec over the eval window, r2_d0, esp, demod_dict-or-None).

    Ks: optional K-batch to integrate (K selected out of it). The stage-A ANCHOR hop must
    pass Ks=p1.K_GRID so the integration reproduces Phase-1's compiled batch shape (the
    committed 0.981470 was element 4 of ONE batch-of-5 vmap; a batch-of-1 compiles a
    different XLA program whose kernel/reduction order can drift at the ULP level --
    adversarial-review finding). Direct span-3.0 stays Ks=None: Phase-1's chunking there
    (maxb=2) made its K=0.24 chunk a batch-of-1 already, the same shape as [K]."""
    bands = band_indices(spec.omega)
    m_fast = masked_encoding(spec.omega, bands["fast"], np.random.default_rng(enc_seed))
    outside = np.concatenate([bands["slow"], bands["guard"]])
    assert float(np.abs(m_fast[outside]).max()) == 0.0        # fast-band-only injection
    Ks_run = [K] if Ks is None else list(Ks)
    ki = Ks_run.index(K)
    Lmat = spec.L if L_override is None else L_override      # scramble arm: degree-matched L
    X = p1.integrate_Ks(spec.omega, Lmat, m_fast, spec.z0, u_in, Ks_run, dt_in, n_sub)[ki]
    rep = replica_spec(spec, rep_seed)
    Xr = p1.integrate_Ks(spec.omega, Lmat, m_fast, rep.z0, u_in, Ks_run, dt_in, n_sub)[ki]
    d_slow = consistency_distance(X[:, bands["slow"]], Xr[:, bands["slow"]], sl)
    esp = {"d_slow": float(d_slow), "ok_slow": bool(d_slow < ESP_EPS)}
    dem = (p1.demod_capacity(X, bands["slow"], s_target, decoys, delays, sl, "full")
           if decoys is not None else None)
    m_rec, r2, lam = demod_fit(X, bands["slow"], s_target, sl, "full")
    if dem is not None:
        assert abs(r2 - dem["r2_d0"]) < 1e-9                  # fit identity (sandbox check 2)
    del X, Xr
    return m_rec, float(r2), esp, dem


def smoke(log):
    import time
    log("=== RELAY GATE-0 :: STAGE-2 SMOKE (seed 0, K=0.24, full end-to-end path) ===")
    i, K = 0, K_PRIMARY
    t0 = time.perf_counter()

    # ---- shared stage geometry (span 1.5; identical for stages A and B) --------------- #
    n_sub = p1.n_sub_for(STAGE_SPAN)
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(STAGE_SPAN)
    sl = slice(eval_start, L)
    iw = np.arange(eval_start, L)
    n_eval = L - eval_start
    ntr = int(p1.TRAIN_FRAC * n_eval)      # shared 70/30 split point: stage-A's out-of-sample
    #                                        region feeds stage-B's out-of-sample region.
    log(f"  window: span={STAGE_SPAN} dt_in={dt_in:.5g} L={L} eval_start={eval_start} "
        f"n_eval={n_eval} ntr={ntr} n_sub={n_sub}")

    # ---- STAGE A: Phase-1 replica, EXACT seed scheme (build i/enc 5000+i/msg 1000+i/
    #      rep 9000+i/decoys 40000-base) -- this is the replication-anchor hop ----------- #
    tA = time.perf_counter()
    sp_A = build_system(i, p1.N, STAGE_SPAN)
    m0, u_A = p1.am_input(L, dt_in, 1000 + i)
    decoys_A = stage_decoys("A", i, L, dt_in)          # byte-identical to Phase-1 (sandboxed)
    m1, r2_A, esp_A, dem_A = _hop(sp_A, 5000 + i, 9000 + i, m0, u_A, K, dt_in, n_sub,
                                  delays, sl, decoys=decoys_A, Ks=p1.K_GRID)
    wall_A = time.perf_counter() - tA
    in_nbhd = bool(SMOKE_LO <= r2_A <= SMOKE_HI)
    log(f"  [stage A] r2(m1,m0)={r2_A:+.6f}  (committed seed-0 ref {REF_SEED0_STAGE:+.6f}; "
        f"neighborhood [{SMOKE_LO},{SMOKE_HI}] -> {'IN' if in_nbhd else 'OUT'})")
    log(f"            decoy_p95={dem_A['decoy_p95']:+.4f} lam={dem_A['lam']:g} "
        f"ESP d_slow={esp_A['d_slow']:.2e} ok_slow={esp_A['ok_slow']}  ({wall_A:.0f}s)")

    # ---- REPEATER: F(m1) = band-limit to MSG_BAND + affine rescale to m0-class stats.
    #      Uses two SCALARS of m0[iw] (mean, fluctuation RMS) -- the spec's message-class
    #      statistics -- never the m0 waveform (no oracle path into stage B). ------------ #
    processed, rparams = repeater_transform(m1, m0[iw], dt_in)
    dc_B = float(np.mean(m0[iw]))
    s_B_eval = remodulate_for_stage_b(processed, m0[iw])      # = clip(dc_B + processed, 1e-6)
    clip_frac = float(np.mean((dc_B + processed) < 1e-6))
    s_B_full = np.full(L, dc_B)                        # washout warmup: message-free DC carrier
    s_B_full[eval_start:] = s_B_eval
    u_B = am_from_message(s_B_full, STAGE_B_CAR_OFF + i)
    log(f"  [repeater] params: {json.dumps(rparams)}")
    log(f"             stage-B DC={dc_B:.4f} clip_frac={clip_frac:.2e} "
        f"(washout region = constant DC, carries no message)")

    # ---- STAGE B: fresh independent network, identical construction; injected message =
    #      the processed relay of m1. r2 target = s_B_full (its own injected message). --- #
    tB = time.perf_counter()
    sp_B = build_system(STAGE_B_NET_OFF + i, p1.N, STAGE_SPAN)
    m2, r2_hop2, esp_B, _ = _hop(sp_B, STAGE_B_ENC_OFF + i, STAGE_B_REP_OFF + i,
                                 s_B_full, u_B, K, dt_in, n_sub, delays, sl, decoys=None)
    wall_B = time.perf_counter() - tB
    # end-to-end: the FIXED m2 series against the ORIGINAL m0, on the shared out-of-sample
    # test split (no refit on m0 -- stage B never sees the m0 waveform).
    r2_e2e = p1.r2_det(m2[ntr:], m0[iw][ntr:])
    log(f"  [stage B] r2(m2,processed-m1)={r2_hop2:+.6f}  ESP d_slow={esp_B['d_slow']:.2e} "
        f"ok_slow={esp_B['ok_slow']}  ({wall_B:.0f}s)")
    log(f"  [end-to-end] r2(m2,m0)={r2_e2e:+.6f}  (fixed m2 vs m0, test split; no refit)")

    # ---- DIRECT span-3.0 (same seed, same message seed -> same continuous m0) ---------- #
    t3 = time.perf_counter()
    n_sub3 = p1.n_sub_for(DIRECT_SPAN)
    dt3, W03, ev3, L3, delays3, stride3 = p1.am_window(DIRECT_SPAN)
    sl3 = slice(ev3, L3)
    sp_3 = build_system(i, p1.N, DIRECT_SPAN)
    m0_3, u_3 = p1.am_input(L3, dt3, 1000 + i)
    decoys_3 = phase1_decoys_ref(i, L3, dt3)           # Phase-1's exact decoy path at span 3
    _, r2_direct, esp_3, dem_3 = _hop(sp_3, 5000 + i, 9000 + i, m0_3, u_3, K, dt3, n_sub3,
                                      delays3, sl3, decoys=decoys_3)
    wall_3 = time.perf_counter() - t3
    log(f"  [direct 3.0] r2={r2_direct:+.6f}  (committed seed-0 ref {REF_SEED0_DIRECT:+.6f}) "
        f"decoy_p95={dem_3['decoy_p95']:+.4f} ESP d_slow={esp_3['d_slow']:.2e} "
        f"ok_slow={esp_3['ok_slow']}  ({wall_3:.0f}s)")

    wall = time.perf_counter() - t0
    smoke_pass = in_nbhd                                # plumbing completed; ESP recorded
    log("\n=== SMOKE SUMMARY (seed 0, K=0.24) ===")
    log(f"  r2(m1,m0)           = {r2_A:+.6f}   [committed ref {REF_SEED0_STAGE:+.6f}]")
    log(f"  r2(m2,processed-m1) = {r2_hop2:+.6f}")
    log(f"  r2(m2,m0)           = {r2_e2e:+.6f}   [naive 2-hop bound ~ r2_A * r2_hop2]")
    log(f"  direct span-3.0 r2  = {r2_direct:+.6f}   [committed ref {REF_SEED0_DIRECT:+.6f}]")
    log(f"  ESP ok_slow: stageA={esp_A['ok_slow']} stageB={esp_B['ok_slow']} "
        f"direct={esp_3['ok_slow']}")
    log(f"  wall-clock: stageA {wall_A:.0f}s + stageB {wall_B:.0f}s + direct {wall_3:.0f}s "
        f"= {wall:.0f}s total")
    log(f"  SMOKE: {'PASS' if smoke_pass else 'FAIL -- STOP, no battery'} "
        f"(anchor-neighborhood {'IN' if in_nbhd else 'OUT'})")

    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gate0_smoke.json")
    with open(outp, "w") as f:
        json.dump({
            "stage": "2-smoke", "seed": i, "K": K, "smoke_pass": smoke_pass,
            "neighborhood": [SMOKE_LO, SMOKE_HI],
            "committed_refs": {"stageA_seed0": REF_SEED0_STAGE,
                               "direct_seed0": REF_SEED0_DIRECT,
                               "provenance": "decade_drive b0f7664 phase1_routing.json"},
            "env": _env_versions(),
            "r2": {"m1_vs_m0": r2_A, "m2_vs_processed_m1": r2_hop2, "m2_vs_m0": r2_e2e,
                   "direct_span3": r2_direct},
            "esp": {"stageA": esp_A, "stageB": esp_B, "direct": esp_3},
            "stageA_demod": dem_A, "direct_demod": dem_3,
            "repeater": {**rparams, "dc_B": dc_B, "clip_frac": clip_frac},
            "seed_scheme": {"stageA": {"build": i, "enc": 5000 + i, "msg": 1000 + i,
                                       "rep": 9000 + i, "decoys_base": 40000},
                            "stageB": {"build": STAGE_B_NET_OFF + i,
                                       "enc": STAGE_B_ENC_OFF + i,
                                       "carrier": STAGE_B_CAR_OFF + i,
                                       "rep": STAGE_B_REP_OFF + i},
                            "direct": {"build": i, "enc": 5000 + i, "msg": 1000 + i,
                                       "rep": 9000 + i}},
            "window": {"span": STAGE_SPAN, "dt_in": dt_in, "L": L, "eval_start": eval_start,
                       "ntr": ntr, "n_sub": n_sub,
                       "direct": {"span": DIRECT_SPAN, "dt_in": dt3, "L": L3,
                                  "eval_start": ev3, "n_sub": n_sub3}},
            "wall_clock_s": {"stageA": wall_A, "stageB": wall_B, "direct": wall_3,
                             "total": wall},
        }, f, indent=1)
    log(f"  [written -> {os.path.relpath(outp)}]  (smoke artifact; NOT committed)")
    return smoke_pass


# ===================================================================================== #
#  STAGE 3 -- full gate battery  (spec conditions 1-6; verdict per pre-registered mapping)
# ===================================================================================== #
E2E_DECOY_BASE = 80000        # e2e scoring decoys: never injected, never fitted (fresh base)
FRAMING = ("Compound span 3.0 is a claim about the INFORMATION PATH -- the message survives "
           "two successive square-law demodulations end-to-end -- NOT about one physical "
           "spectrum; the comparison target is the fresh paired direct span-3.0 run "
           "(spec 'Honest framing'; stated in all outputs).")


def _dump_json(path, payload):
    """Atomic json write (tmp + replace): a kill mid-dump never corrupts the artifact."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
VIOL_E2E_BAR = 0.1            # RE-POSED collapse bar (addendum, on Jason's word): the re-posed
                              # repeater-filter violation must have e2e r2 < this on EVERY
                              # intersection seed (per-seed absolute -- no ratio ambiguity).
# (The retired rate-limit check's ratio bar -- VIOL_COLLAPSE_FRAC * compliant OR an absolute
#  floor -- is gone with the check itself; see relay_gate0_spec_addendum.md.)
DECOY_ELEVATED = 0.2          # any decoy-p95 intersection mean above this = leakage flag
                              # (committed Phase-1 decoy_p95 range ~[-0.13,+0.03]; real signal ~0.96)
DIRECT16_BATCH = [0.12, 0.16]  # Phase-1's span-3 chunk containing K=0.16 (shape-exact secondary)

# Committed per-seed references, seeds 0-9 (decade_drive b0f7664 phase1_routing.json):
# (span, K) -> {seed: (r2_d0 [6dp], esp ok_slow)} -- the bit-exact replication table targets.
REF_TABLE = {
    (1.5, 0.24): {0: (0.981470, True), 1: (0.987776, True), 2: (0.990025, True),
                  3: (0.982354, True), 4: (0.990056, True), 5: (0.986100, True),
                  6: (0.987164, True), 7: (0.986687, True), 8: (0.988723, True),
                  9: (0.980370, True)},
    (1.5, 0.16): {0: (0.951091, True), 1: (0.960078, True), 2: (0.953189, True),
                  3: (0.923006, True), 4: (0.966711, True), 5: (0.932150, True),
                  6: (0.941515, True), 7: (0.936835, False), 8: (0.925840, False),
                  9: (0.932156, True)},
    (3.0, 0.24): {0: (0.000724, True), 1: (-0.011609, True), 2: (-0.000545, True),
                  3: (0.001416, True), 4: (-0.030314, True), 5: (0.026516, True),
                  6: (0.001817, True), 7: (-0.007683, False), 8: (-0.005132, False),
                  9: (-0.011102, True)},
    (3.0, 0.16): {0: (0.002983, False), 1: (-0.011951, False), 2: (-0.001651, True),
                  3: (-0.000917, True), 4: (-0.030508, False), 5: (0.019823, True),
                  6: (0.002194, False), 7: (-0.007058, False), 8: (-0.008679, False),
                  9: (-0.011302, False)},
}
REF_SEED_MAX = 9              # committed Phase-1 ran 10 seeds; --nseeds must not exceed 10


def _sha12(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _geom(span):
    """(dt_in, eval_start, L, delays, sl, iw, ntr, n_sub) for one span (Phase-1 windows)."""
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(span)
    sl = slice(eval_start, L)
    iw = np.arange(eval_start, L)
    ntr = int(p1.TRAIN_FRAC * (L - eval_start))
    return dt_in, eval_start, L, delays, sl, iw, ntr, p1.n_sub_for(span)


def _e2e_score(m2, target_full, iw, ntr):
    """Score the FIXED m2 against a full-window target on the shared out-of-sample split."""
    return float(p1.r2_det(m2[ntr:], np.asarray(target_full, float)[iw][ntr:]))


def _chain_stage_b(i, m1, m0_iw, band, geom, sp_B, K, log):
    """Repeater + stage B for one arm at coupling K: F(m1) -> remodulate -> inject into
    the fresh stage-B network -> (m2, r2_hop2, esp_B, dem_B, repeater record). K must be
    the ARM'S K row (the K=0.16 secondary runs BOTH stages at 0.16)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    processed, rparams = repeater_transform(m1, m0_iw, dt_in, w_lo=band[0], w_hi=band[1])
    dc_B = float(np.mean(m0_iw))
    s_B_eval = remodulate_for_stage_b(processed, m0_iw)
    clip_frac = float(np.mean((dc_B + processed) < 1e-6))
    s_B_full = np.full(L, dc_B)
    s_B_full[eval_start:] = s_B_eval
    u_B = am_from_message(s_B_full, STAGE_B_CAR_OFF + i)
    decoys_B = stage_decoys("B", i, L, dt_in)
    m2, r2_B, esp_B, dem_B = _hop(sp_B, STAGE_B_ENC_OFF + i, STAGE_B_REP_OFF + i,
                                  s_B_full, u_B, K, dt_in, n_sub, delays, sl,
                                  decoys=decoys_B)
    rep_rec = {**rparams, "dc_B": dc_B, "clip_frac": clip_frac}
    return m2, r2_B, esp_B, dem_B, rep_rec, s_B_full


def _reposed_violation(i, geom, log):
    """RE-POSED bandwidth-violation control (addendum, replacing the retired rate-limit
    check). Tests the REPEATER-FILTER bookkeeping, the one part of the envelope-of-envelope
    machinery that can actually fail:

      * message = [2,9] rad/s (VIOL band); the sweep proved stage A TRACKS it (r2 ~0.88),
        so m1 faithfully carries the [2,9] message -- upstream physics is NOT the cause;
      * repeater pass-band = the STANDARD [0.2,0.9] (NOT the message band) -> the brick-wall
        keeps only the in-band reconstruction RESIDUAL of a [2,9] message (~0), then the
        affine rescale amplifies that residual to full message RMS;
      * stage B therefore receives m0-uncorrelated noise -> e2e r2(m2, m0) ~ 0.

    Signature the check worked for the STATED reason: small rms_in, LARGE scale (a big
    amplification of a tiny residual). Both are logged (rep record). If instead e2e does NOT
    collapse, the repeater filter / rescale bookkeeping is wrong -> NO-MEASUREMENT."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    m0_v, u_v = am_input_band(L, dt_in, 1000 + i, VIOL_LO, VIOL_HI)      # [2,9] message
    sp_Av = build_system(i, p1.N, STAGE_SPAN)
    m1_v, r2_Av, esp_Av, _ = _hop(sp_Av, 5000 + i, 9000 + i, m0_v, u_v, K_PRIMARY,
                                  dt_in, n_sub, delays, sl, decoys=None)
    sp_Bv = build_system(STAGE_B_NET_OFF + i, p1.N, STAGE_SPAN)
    # RE-POSE: repeater pass-band = STANDARD [0.2,0.9], deliberately MISMATCHED to the [2,9]
    # message -> the filter removes the message; only the amplified residual reaches stage B.
    m2_v, r2_Bv, esp_Bv, _, rep_v, _ = _chain_stage_b(
        i, m1_v, m0_v[iw], (MSG_LO, MSG_HI), geom, sp_Bv, K_PRIMARY, log)
    e2e_v = _e2e_score(m2_v, m0_v, iw, ntr)
    rec = {
        "seed": i, "K": K_PRIMARY, "arm": "reposed_violation",
        "message_band": [VIOL_LO, VIOL_HI], "repeater_passband": [MSG_LO, MSG_HI],
        "mechanism": "message [2,9] tracked by stage A; repeater brick-wall keeps only the "
                     "[0.2,0.9] residual; rescale amplifies it to full RMS; stage B gets "
                     "m0-uncorrelated noise -> e2e ~ 0. Signature: small rms_in, large scale.",
        "stageA": {"r2": float(r2_Av), "esp": esp_Av},
        "repeater": rep_v,          # carries rms_in, rms_target, scale (the signature)
        "stageB": {"r2": float(r2_Bv), "esp": esp_Bv},
        # r2(m2, processed-m1): stage B reconstructing its OWN injected (amplified-residual)
        # message. HIGH here WHILE e2e(m2,m0) ~ 0 is the cleanest signature that the repeater
        # FILTER removed the message -- stage B itself worked fine (= stageB.r2, made explicit).
        "r2_m2_vs_processed_m1": float(r2_Bv),
        "e2e": {"r2": e2e_v},
    }
    log(f"  seed {i} reposed-viol stageA={r2_Av:+.4f} r2(m2,proc-m1)={r2_Bv:+.4f} "
        f"e2e(m2,m0)={e2e_v:+.4f} rms_in={rep_v['rms_in']:.3g} scale={rep_v['scale']:.1f} "
        f"ESP A/B={esp_Av['ok_slow']}/{esp_Bv['ok_slow']}")
    return rec


def relay_seed(i, geom, log):
    """All span-1.5 arms for seed i. Stage A is integrated ONCE per (system, input) with the
    full committed K_GRID batch (Phase-1's compiled shape -> bit-exact anchor at BOTH K rows;
    K=0.24 = element 4, K=0.16 = element 3). Returns records for relay24 / relay16 /
    violation24 / scramble24."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    out = {}

    # ---- compliant stage A: ONE K_GRID-batched integration serves both K rows ---------- #
    sp_A = build_system(i, p1.N, STAGE_SPAN)
    bands_A = band_indices(sp_A.omega)
    m_fast_A = masked_encoding(sp_A.omega, bands_A["fast"], np.random.default_rng(5000 + i))
    outside = np.concatenate([bands_A["slow"], bands_A["guard"]])
    assert float(np.abs(m_fast_A[outside]).max()) == 0.0
    m0, u_A = p1.am_input(L, dt_in, 1000 + i)
    decoys_A = stage_decoys("A", i, L, dt_in)                  # Phase-1 byte-identical
    XA = p1.integrate_Ks(sp_A.omega, sp_A.L, m_fast_A, sp_A.z0, u_A, p1.K_GRID, dt_in, n_sub)
    rep_A = replica_spec(sp_A, 9000 + i)
    XAr = p1.integrate_Ks(sp_A.omega, sp_A.L, m_fast_A, rep_A.z0, u_A, p1.K_GRID, dt_in, n_sub)

    for K, tag in ((K_PRIMARY, "relay"), (K_SECONDARY, "relay_k16")):
        ki = p1.K_GRID.index(K)
        X, Xr = XA[ki], XAr[ki]
        d_slow = consistency_distance(X[:, bands_A["slow"]], Xr[:, bands_A["slow"]], sl)
        esp_A = {"d_slow": float(d_slow), "ok_slow": bool(d_slow < ESP_EPS)}
        dem_A = p1.demod_capacity(X, bands_A["slow"], m0, decoys_A, delays, sl, "full")
        m1, r2_A, lam_A = demod_fit(X, bands_A["slow"], m0, sl, "full")
        assert abs(r2_A - dem_A["r2_d0"]) < 1e-9
        sp_B = build_system(STAGE_B_NET_OFF + i, p1.N, STAGE_SPAN)
        m2, r2_B, esp_B, dem_B, rep_rec, s_B_full = _chain_stage_b(
            i, m1, m0[iw], (MSG_LO, MSG_HI), geom, sp_B, K, log)
        e2e = _e2e_score(m2, m0, iw, ntr)
        e2e_dec = [_e2e_score(m2, p1.slow_bandlimited(L, dt_in, MSG_LO, MSG_HI,
                                                      seed=E2E_DECOY_BASE + i * 200 + d),
                              iw, ntr) for d in range(N_DEC)]
        out[tag] = {
            "seed": i, "K": K, "band": [MSG_LO, MSG_HI], "arm": tag,
            "stageA": {"r2": float(r2_A), "demod": dem_A, "esp": esp_A},
            "repeater": rep_rec,
            "stageB": {"r2": float(r2_B), "demod": dem_B, "esp": esp_B},
            "e2e": {"r2": e2e, "decoy_p95": float(np.percentile(e2e_dec, 95)),
                    "decoy_mean": float(np.mean(e2e_dec))},
        }
        ref = REF_TABLE[(STAGE_SPAN, K)][i]
        log(f"  seed {i} {tag:10s} A={r2_A:+.6f} (ref {ref[0]:+.6f}) B={r2_B:+.6f} "
            f"e2e={e2e:+.6f} decp95={out[tag]['e2e']['decoy_p95']:+.3f} "
            f"ESP A/B={esp_A['ok_slow']}/{esp_B['ok_slow']}")
    del XA, XAr

    # ---- RE-POSED violation arm (K=0.24): repeater-filter bookkeeping check -------------- #
    out["violation"] = _reposed_violation(i, geom, log)

    # ---- scramble arm (K=0.24 only): stage A on the degree-matched random coupling ----- #
    Lscr = p1.scramble_laplacian(p1.N, i)                      # Phase-1 exact (rng 70000+i)
    sp_As = build_system(i, p1.N, STAGE_SPAN)
    m1_s, r2_As, esp_As, _ = _hop(sp_As, 5000 + i, 9500 + i, m0, u_A, K_PRIMARY,
                                  dt_in, n_sub, delays, sl, decoys=None, L_override=Lscr)
    sp_Bs = build_system(STAGE_B_NET_OFF + i, p1.N, STAGE_SPAN)
    m2_s, r2_Bs, esp_Bs, _, rep_s, _ = _chain_stage_b(
        i, m1_s, m0[iw], (MSG_LO, MSG_HI), geom, sp_Bs, K_PRIMARY, log)
    e2e_s = _e2e_score(m2_s, m0, iw, ntr)
    out["scramble"] = {
        "seed": i, "K": K_PRIMARY, "band": [MSG_LO, MSG_HI], "arm": "scramble",
        "stageA": {"r2": float(r2_As), "esp": esp_As},
        "repeater": rep_s,
        "stageB": {"r2": float(r2_Bs), "esp": esp_Bs},
        "e2e": {"r2": e2e_s},
    }
    log(f"  seed {i} scramble   A={r2_As:+.6f} B={r2_Bs:+.6f} e2e={e2e_s:+.6f} "
        f"ESP A/B={esp_As['ok_slow']}/{esp_Bs['ok_slow']}")
    return out


def direct_seed(i, geom3, log):
    """Fresh paired direct span-3.0 baseline for seed i, both K rows, Phase-1-exact seeds.
    K=0.24 runs batch-of-1 (Phase-1's chunk shape at span 3 -> bit-exact); K=0.16 runs
    inside Phase-1's [0.12,0.16] chunk (shape-exact secondary)."""
    dt3, ev3, L3, delays3, sl3, iw3, ntr3, n_sub3 = geom3
    sp_3 = build_system(i, p1.N, DIRECT_SPAN)
    m0_3, u_3 = p1.am_input(L3, dt3, 1000 + i)
    decoys_3 = phase1_decoys_ref(i, L3, dt3)
    out = {}
    for K, Ks, tag in ((K_PRIMARY, None, "direct"), (K_SECONDARY, DIRECT16_BATCH, "direct_k16")):
        _, r2, esp, dem = _hop(sp_3, 5000 + i, 9000 + i, m0_3, u_3, K, dt3, n_sub3,
                               delays3, sl3, decoys=decoys_3, Ks=Ks)
        out[tag] = {"seed": i, "K": K, "r2": float(r2), "demod": dem, "esp": esp}
        ref = REF_TABLE[(DIRECT_SPAN, K)][i]
        log(f"  seed {i} {tag:10s} r2={r2:+.6f} (ref {ref[0]:+.6f}) "
            f"decp95={dem['decoy_p95']:+.3f} ESP={esp['ok_slow']}")
    return out


def _mstats(vals):
    v = [float(x) for x in vals]
    n = len(v)
    m = float(np.mean(v)) if n else float("nan")
    sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else float("nan")
    return {"mean": m, "sd": sd, "se": se, "n": n, "per_seed": v}


def decide(recs, seeds):
    """Pre-registered verdict mapping (spec 'Pre-registered outcomes'), instrument checks
    FIRST. Operationalizations (fixed here, before the run):
      * anchor: |stage-A intersection mean - 0.986| <= max(2*SE, 0.02); low-side miss ->
        replication failure; high-side miss -> leakage-suspect (either = NO-MEASUREMENT).
      * leakage: any decoy-p95 intersection mean (stage A, stage B, e2e) > 0.2.
      * RE-POSED violation collapse (addendum): the re-posed repeater-filter check must
        collapse to e2e r2 < VIOL_E2E_BAR (=0.1) on EVERY seed of the shared ESP-ok
        sub-intersection (per-seed absolute; no ratio ambiguity). >= MIN_PAIRS or
        underpowered -> NM. (Supersedes the retired ratio bar; the rate-limit violation
        check itself was retired as empirically void -- see relay_gate0_spec_addendum.md.)
      * PASS: paired deltas (e2e - direct) AND (e2e - e2e_decoy_p95) both have
        mean > sd (the '> seed sigma' bar), on the verdict intersection.
      * FAIL: instruments healthy, PASS bar not met (includes the within-sigma gap).
    Verdict intersection = ESP-ok on relay stage A AND relay stage B AND direct (K=0.24)."""
    R, D, V = recs["relay"], recs["direct"], recs["violation"]
    esp_table = {i: {"relayA": R[i]["stageA"]["esp"]["ok_slow"],
                     "relayB": R[i]["stageB"]["esp"]["ok_slow"],
                     "direct": D[i]["esp"]["ok_slow"]} for i in seeds}
    inter = paired_intersection(esp_table, ["relayA", "relayB", "direct"])
    out = {"esp_table": {str(k): v for k, v in esp_table.items()},
           "intersection": inter, "n_intersection": len(inter),
           "operationalization": {"anchor_window": f"max({ANCHOR_SE_K}*SE, {ANCHOR_FLOOR})",
                                  "decoy_elevated": DECOY_ELEVATED,
                                  "viol_e2e_bar_per_seed": VIOL_E2E_BAR,
                                  "pass_bar": "mean(delta) > sd(delta), both deltas",
                                  "min_pairs": MIN_PAIRS}}
    if len(inter) < MIN_PAIRS:
        out["verdict"] = ("NO-MEASUREMENT (underpowered: intersection "
                          f"n={len(inter)} < {MIN_PAIRS} -- add seeds, do not read)")
        return out

    # ---- anchor ------------------------------------------------------------------------ #
    stA = _mstats([R[i]["stageA"]["r2"] for i in inter])
    window = max(ANCHOR_SE_K * stA["se"], ANCHOR_FLOOR)
    anchor_dev = stA["mean"] - ANCHOR
    anchor_ok = bool(abs(anchor_dev) <= window)
    out["anchor"] = {"stageA": stA, "target": ANCHOR, "window": window,
                     "deviation": anchor_dev, "ok": anchor_ok}

    # ---- decoy health (leakage) --------------------------------------------------------- #
    decA = float(np.mean([R[i]["stageA"]["demod"]["decoy_p95"] for i in inter]))
    decB = float(np.mean([R[i]["stageB"]["demod"]["decoy_p95"] for i in inter]))
    decE = float(np.mean([R[i]["e2e"]["decoy_p95"] for i in inter]))
    leak = bool(max(decA, decB, decE) > DECOY_ELEVATED)
    out["decoys"] = {"stageA_p95_mean": decA, "stageB_p95_mean": decB,
                     "e2e_p95_mean": decE, "elevated": leak}

    # ---- RE-POSED violation collapse: e2e r2 < VIOL_E2E_BAR on EVERY vinter seed --------- #
    vinter = [i for i in inter if V[i]["stageA"]["esp"]["ok_slow"]
              and V[i]["stageB"]["esp"]["ok_slow"]]
    if len(vinter) >= MIN_PAIRS:
        viol_ps = {i: float(V[i]["e2e"]["r2"]) for i in vinter}
        collapsed = all(r < VIOL_E2E_BAR for r in viol_ps.values())      # per-seed, all
        offenders = [i for i, r in viol_ps.items() if r >= VIOL_E2E_BAR]
        viol_mean = float(np.mean(list(viol_ps.values())))
        compl = float(np.mean([R[i]["e2e"]["r2"] for i in vinter]))
        # signature of the re-posed mechanism, averaged: small rms_in, large scale, AND
        # stage B faithfully reconstructing the residual (r2(m2,processed-m1) high) while
        # e2e collapses -> the FILTER removed the message, not an upstream/stage-B failure.
        rms_in = float(np.mean([V[i]["repeater"]["rms_in"] for i in vinter]))
        scale = float(np.mean([V[i]["repeater"]["scale"] for i in vinter]))
        stageB_r2 = float(np.mean([V[i]["stageB"]["r2"] for i in vinter]))
    else:
        viol_ps, collapsed, offenders = {}, False, []
        viol_mean, compl, rms_in, scale, stageB_r2 = None, None, None, None, None
    out["violation"] = {"n": len(vinter), "intersection": vinter, "bar_per_seed": VIOL_E2E_BAR,
                        "e2e_per_seed": {str(k): v for k, v in viol_ps.items()},
                        "offenders": offenders, "violation_e2e_mean": viol_mean,
                        "compliant_e2e_mean": compl, "signature_rms_in": rms_in,
                        "signature_scale": scale, "signature_stageB_r2": stageB_r2,
                        "collapsed": collapsed}

    # ---- paired deltas (the PASS bar) ---------------------------------------------------- #
    d_direct = _mstats([R[i]["e2e"]["r2"] - D[i]["r2"] for i in inter])
    d_decoy = _mstats([R[i]["e2e"]["r2"] - R[i]["e2e"]["decoy_p95"] for i in inter])
    beats_direct = bool(d_direct["mean"] > max(d_direct["sd"], 1e-12))
    beats_decoy = bool(d_decoy["mean"] > max(d_decoy["sd"], 1e-12))
    out["deltas"] = {"e2e_minus_direct": d_direct, "e2e_minus_decoy_p95": d_decoy,
                     "beats_direct": beats_direct, "beats_decoy": beats_decoy}
    out["summaries"] = {
        "relay_e2e": _mstats([R[i]["e2e"]["r2"] for i in inter]),
        "direct": _mstats([D[i]["r2"] for i in inter]),
        "relay_hop2": _mstats([R[i]["stageB"]["r2"] for i in inter]),
    }

    # ---- verdict (instrument checks first; order pre-registered) ------------------------- #
    if not anchor_ok:
        side = "low-side (replication failure)" if anchor_dev < 0 else \
               ("high-side + elevated decoy (leakage)" if leak else "high-side")
        out["verdict"] = (f"NO-MEASUREMENT (anchor miss, {side}: mean={stA['mean']:.6f} "
                          f"vs {ANCHOR} +/- {window:.4f} -- STOP, fix, re-run)")
    elif leak:
        out["verdict"] = ("NO-MEASUREMENT (decoy elevated -- leakage: "
                          f"p95 means A={decA:.3f} B={decB:.3f} e2e={decE:.3f} > "
                          f"{DECOY_ELEVATED})")
    elif not collapsed:
        vs = (f"seeds {out['violation']['offenders']} have e2e r2 >= {VIOL_E2E_BAR} "
              f"(mean {viol_mean:.3f})" if viol_mean is not None
              else f"violation sub-intersection underpowered (n={len(vinter)})")
        out["verdict"] = (f"NO-MEASUREMENT (re-posed violation did NOT collapse: {vs} -- "
                          "repeater-filter/rescale bookkeeping wrong or check underpowered)")
    elif beats_direct and beats_decoy:
        out["verdict"] = ("PASS (horizon is architectural): relay e2e beats the fresh direct "
                          "span-3.0 baseline AND the e2e decoy floor by > seed sigma "
                          "on the ESP-honest paired intersection")
    else:
        out["verdict"] = ("FAIL (horizon binds active staging at this operating point): "
                          "instruments healthy, relay does not beat "
                          f"{'direct' if not beats_direct else 'decoy floor'} by > seed sigma")
    return out


def _secondary_summary(recs, seeds):
    """Optional K=0.16 bracket row (spec condition 6) -- reported, NEVER part of the verdict."""
    R16, D16 = recs["relay_k16"], recs["direct_k16"]
    esp16 = {i: {"relayA": R16[i]["stageA"]["esp"]["ok_slow"],
                 "relayB": R16[i]["stageB"]["esp"]["ok_slow"],
                 "direct": D16[i]["esp"]["ok_slow"]} for i in seeds}
    inter16 = paired_intersection(esp16, ["relayA", "relayB", "direct"])
    rec = {"intersection": inter16, "n": len(inter16),
           "underpowered": len(inter16) < MIN_PAIRS}
    if inter16:
        rec["relay_e2e"] = _mstats([R16[i]["e2e"]["r2"] for i in inter16])
        rec["direct"] = _mstats([D16[i]["r2"] for i in inter16])
        rec["stageA"] = _mstats([R16[i]["stageA"]["r2"] for i in inter16])
    return rec


def _replication_table(recs, seeds):
    """Per-seed measured-vs-committed table for the four Phase-1 rows (6dp match flags)."""
    rows = []
    for (span, K), refs in REF_TABLE.items():
        for i in seeds:
            if span == STAGE_SPAN:
                arm = "relay" if K == K_PRIMARY else "relay_k16"
                got = recs[arm][i]["stageA"]["r2"]
                esp = recs[arm][i]["stageA"]["esp"]["ok_slow"]
            else:
                arm = "direct" if K == K_PRIMARY else "direct_k16"
                got = recs[arm][i]["r2"]
                esp = recs[arm][i]["esp"]["ok_slow"]
            ref_r2, ref_esp = refs[i]
            rows.append({"span": span, "K": K, "seed": i, "measured": round(got, 6),
                         "committed": ref_r2, "match_6dp": bool(round(got, 6) == ref_r2),
                         "esp_measured": esp, "esp_committed": ref_esp,
                         "esp_match": bool(esp == ref_esp)})
    return rows


def _fmt(x, spec="+.4f"):
    """Format a possibly-None/underpowered value for the .md (None -> 'n/a')."""
    return format(x, spec) if isinstance(x, (int, float)) and x == x else "n/a"


def _write_md(path, verdict, secondary, repl_rows, recs, seeds, wall, hashes):
    v = verdict
    n_match = sum(r["match_6dp"] for r in repl_rows)
    lines = [
        "# Relay Gate-0 -- full-gate record (Stage 3)",
        "",
        f"Spec: relay_gate0_spec.md (sha256 {hashes['spec']}). Harness: "
        f"experiments/relay_gate0.py (sha256 {hashes['code']}, uncommitted at run time).",
        f"Provenance anchors: decade_drive b0f7664 (Phase-1). Seeds run: {seeds}. "
        f"K primary = {K_PRIMARY}. Wall-clock {wall/60:.0f} min.",
        "",
        "## Verdict",
        "",
        f"**{v['verdict']}**",
        "",
        "## Instrument checks (evaluated before the verdict, pre-registered order)",
        "",
    ]
    if "anchor" in v:            # absent only on the underpowered early return
        lines += [
            f"1. **Anchor**: stage-A intersection mean = {v['anchor']['stageA']['mean']:.6f} "
            f"(SE {v['anchor']['stageA']['se']:.6f}, n={v['anchor']['stageA']['n']}); target "
            f"{ANCHOR} +/- {v['anchor']['window']:.4f}; deviation "
            f"{v['anchor']['deviation']:+.6f} -> {'OK' if v['anchor']['ok'] else 'MISS'}.",
            f"2. **Decoy floors** (intersection means): stage-A p95 "
            f"{v['decoys']['stageA_p95_mean']:+.4f}, "
            f"stage-B p95 {v['decoys']['stageB_p95_mean']:+.4f}, e2e p95 "
            f"{v['decoys']['e2e_p95_mean']:+.4f}; elevated bar {DECOY_ELEVATED} -> "
            f"{'ELEVATED (leak)' if v['decoys']['elevated'] else 'clean'}.",
            f"3. **Re-posed violation control** (repeater-filter bookkeeping): message "
            f"[{VIOL_LO},{VIOL_HI}] rad/s (stage-A-tracked), repeater pass-band "
            f"[{MSG_LO},{MSG_HI}] (mismatched -> filters the message out); n="
            f"{v['violation']['n']}. e2e mean {_fmt(v['violation']['violation_e2e_mean'])} "
            f"(bar: e2e r2 < {v['violation']['bar_per_seed']} on EVERY seed) -> "
            f"{'COLLAPSED (instrument sound)' if v['violation']['collapsed'] else 'DID NOT COLLAPSE '+str(v['violation']['offenders'])}. "
            f"Signature: rms_in {_fmt(v['violation']['signature_rms_in'],'.4g')}, "
            f"scale {_fmt(v['violation']['signature_scale'],'.1f')}, "
            f"stage-B r2(m2,proc-m1) {_fmt(v['violation']['signature_stageB_r2'],'.4f')} "
            f"(small residual, large amplification, stage B reconstructs it while e2e ~ 0 = "
            f"the filter removed the message, not upstream/stage-B breakage).",
            f"4. **ESP-honest paired intersection**: {v['intersection']} "
            f"(n={v['n_intersection']}; a seed failing ANY of relay-A/relay-B/direct is "
            f"dropped everywhere).",
        ]
    else:
        lines += [
            f"- Battery ended on the pre-registered early exit before instrument checks: "
            f"intersection {v['intersection']} (n={v['n_intersection']}). Per-seed ESP "
            f"flags: {v['esp_table']}.",
        ]
    lines += [
        "",
        "## Verdict-row numbers (K=0.24, intersection means +/- SE)",
        "",
    ]
    if "summaries" not in v:
        lines += ["- not read (battery ended pre-verdict).", ""]
    else:
        s = v["summaries"]
        d = v["deltas"]
        lines += [
            f"- relay end-to-end r2(m2,m0): **{s['relay_e2e']['mean']:+.4f} +/- "
            f"{s['relay_e2e']['se']:.4f}** (per-hop: stage-B r2 "
            f"{s['relay_hop2']['mean']:+.4f})",
            f"- fresh direct span-3.0: **{s['direct']['mean']:+.4f} +/- "
            f"{s['direct']['se']:.4f}**",
            f"- paired delta e2e - direct: {d['e2e_minus_direct']['mean']:+.4f} "
            f"(sd {d['e2e_minus_direct']['sd']:.4f}) -> beats: {d['beats_direct']}",
            f"- paired delta e2e - e2e-decoy-p95: {d['e2e_minus_decoy_p95']['mean']:+.4f} "
            f"(sd {d['e2e_minus_decoy_p95']['sd']:.4f}) -> beats: {d['beats_decoy']}",
            "",
        ]
    lines += [
        "## Scramble robustness line (characterisation only, never verdict)",
        "",
        _scramble_line(recs, v.get("intersection", [])),
        "",
        "## Secondary K row (K=0.16 bracket -- optional, non-verdict)",
        "",
        _secondary_line(secondary),
        "",
        "## Replication table (measured vs committed b0f7664, 6dp)",
        "",
        f"{n_match}/{len(repl_rows)} rows match to 6dp. Non-matching rows (if any):",
    ]
    misses = [r for r in repl_rows if not r["match_6dp"]]
    if misses:
        for r in misses:
            lines.append(f"- span {r['span']} K {r['K']} seed {r['seed']}: measured "
                         f"{r['measured']:+.6f} vs committed {r['committed']:+.6f}")
    else:
        lines.append("- none -- every measured Phase-1 row reproduces the committed value.")
    lines += [
        "",
        "## Scope",
        "",
        "Offline two-stage square-law relay; compound span 3.0 = a claim about the",
        "INFORMATION PATH (two successive square-law demodulations end-to-end), not one",
        "physical spectrum (spec 'Honest framing'). STOP-and-report: no sweep, no multi-hop",
        "extension, no interpretation beyond the pre-registered outcome mapping.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _scramble_line(recs, inter):
    S, R = recs.get("scramble", {}), recs.get("relay", {})
    ok = [i for i in inter if i in S and S[i]["stageA"]["esp"]["ok_slow"]
          and S[i]["stageB"]["esp"]["ok_slow"]]
    if not ok:
        return "- no ESP-ok scramble seeds in the intersection (not read)."
    sm = _mstats([S[i]["e2e"]["r2"] for i in ok])
    rm = _mstats([R[i]["e2e"]["r2"] for i in ok])
    return (f"- scrambled-stage-A relay e2e = {sm['mean']:+.4f} +/- {sm['se']:.4f} vs "
            f"compliant {rm['mean']:+.4f} +/- {rm['se']:.4f} (n={sm['n']}) -- staging "
            f"{'topology-generic (consistent with Phase-1)' if abs(sm['mean'] - rm['mean']) < max(3 * rm['sd'], 0.1) else 'shows a topology dependence (flag)'} "
            f"(3-sigma/0.1 bar = code-level pre-run operationalization; the spec clause is "
            f"qualitative).")


def _secondary_line(sec):
    if sec.get("underpowered") or "relay_e2e" not in sec:
        return (f"- underpowered (intersection n={sec.get('n', 0)} < {MIN_PAIRS}; committed "
                "Phase-1 span-3.0 @ K=0.16 had ESP 3/10) -- reported, not read.")
    return (f"- relay e2e {sec['relay_e2e']['mean']:+.4f} +/- {sec['relay_e2e']['se']:.4f} vs "
            f"direct {sec['direct']['mean']:+.4f} +/- {sec['direct']['se']:.4f} "
            f"(n={sec['n']}; stage-A anchor row mean {sec['stageA']['mean']:+.4f}) -- "
            "bracket line only, never verdict.")


# ===================================================================================== #
#  VIOLATION-BAND RE-CALIBRATION SWEEP  (Stage-3 NM follow-up; fork 2+1: MEASURE the
#  slow-readout envelope-tracking cutoff, THEN set the violation band by the rule below)
# ===================================================================================== #
# PRE-REGISTERED (fixed in source BEFORE any sweep r2 is seen -- band-shopping guard):
#   * swept band width = the STANDARD message width (rho = hi/lo = 4.5, shared by the
#     compliant [0.2,0.9] and original violation [2,9] bands), so each swept band is a
#     genuine candidate violation band of the SAME shape;
#   * derived violation band = the LOWEST swept center whose stage-A r2 < SWEEP_R2_CUTOFF
#     on ALL sweep seeds.
# OUTCOME (2026-07-04): NO band qualified (readout tracks every representable band) -> the
# rate-limit check was RETIRED and the violation control RE-POSED to the repeater-filter
# check (VIOL_E2E_BAR, per-seed); see relay_gate0_spec_addendum.md.
SWEEP_WIDTH_RATIO = 4.5
SWEEP_R2_CUTOFF = 0.1
SWEEP_SEEDS = [0, 1, 2]          # 3 verdict-intersection seeds
SWEEP_CMAX_NYQ_FRAC = 0.95       # top band's UPPER edge <= this * Nyquist (representability)
SWEEP_NPTS = 8


def _band_from_center(c, rho=SWEEP_WIDTH_RATIO):
    r = rho ** 0.5
    return c / r, c * r


def _stage_a_r2(seed, wlo, whi, geom):
    """Stage-A-only trackability: inject an AM message in band [wlo,whi] into the fast
    tertile, reconstruct it from the slow band (K=0.24). No repeater, no stage B, no ESP
    replica -- just 'can the slow readout follow a message in this band'. batch-of-1
    (this is a fresh measurement, not a committed-shape replication)."""
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    sp = build_system(seed, p1.N, STAGE_SPAN)
    bands = band_indices(sp.omega)
    m_fast = masked_encoding(sp.omega, bands["fast"], np.random.default_rng(5000 + seed))
    outside = np.concatenate([bands["slow"], bands["guard"]])
    assert float(np.abs(m_fast[outside]).max()) == 0.0
    s_msg, u = am_input_band(L, dt_in, 1000 + seed, wlo, whi)
    X = p1.integrate_Ks(sp.omega, sp.L, m_fast, sp.z0, u, [K_PRIMARY], dt_in, n_sub)[0]
    _, r2, _ = demod_fit(X, bands["slow"], s_msg, sl, "full")
    del X
    return float(r2)


def _mechanism(band, flo, fhi):
    """Which mechanism a candidate violation band tests (Jason's caveat): bands overlapping
    the fast tertile [flo,fhi] confound the readout-envelope limit with injection
    ill-posedness (message colliding with the fast carriers)."""
    wlo, whi = band
    if whi <= flo:
        return ("readout-envelope-limit (clean)",
                f"band [{wlo:.2f},{whi:.2f}] lies entirely BELOW the fast tertile "
                f"[{flo:.2f},{fhi:.2f}]; collapse tests the slow readout's envelope limit only.")
    if wlo >= flo:
        return ("injection-ill-posedness (confounded)",
                f"band [{wlo:.2f},{whi:.2f}] lies entirely WITHIN the fast tertile "
                f"[{flo:.2f},{fhi:.2f}]; collapse may reflect the message colliding with the "
                f"fast carriers, NOT the readout-envelope limit.")
    return ("MIXED (partially confounded)",
            f"band [{wlo:.2f},{whi:.2f}] partially overlaps the fast tertile "
            f"[{flo:.2f},{fhi:.2f}]; collapse conflates the readout limit with injection "
            f"ill-posedness.")


def band_sweep(log, nseeds=None):
    import time
    seeds = SWEEP_SEEDS if nseeds is None else list(range(nseeds))
    geom = _geom(STAGE_SPAN)
    dt_in, eval_start, L, delays, sl, iw, ntr, n_sub = geom
    nyq = float(np.pi / dt_in)
    c_max = SWEEP_CMAX_NYQ_FRAC * nyq / (SWEEP_WIDTH_RATIO ** 0.5)
    centers = [float(c) for c in np.logspace(np.log10(1.5), np.log10(c_max), SWEEP_NPTS)]
    sp0 = build_system(0, p1.N, STAGE_SPAN)
    bi = band_indices(sp0.omega)
    slo, shi = float(sp0.omega[bi["slow"]].min()), float(sp0.omega[bi["slow"]].max())
    flo, fhi = float(sp0.omega[bi["fast"]].min()), float(sp0.omega[bi["fast"]].max())

    log("=== RELAY GATE-0 :: VIOLATION-BAND RE-CALIBRATION SWEEP (stage-A only; K=0.24) ===")
    log(f"    span {STAGE_SPAN}: Nyquist={nyq:.2f} rad/s | slow tertile [{slo:.2f},{shi:.2f}] "
        f"| guard ({shi:.2f},{flo:.2f}) | fast tertile [{flo:.2f},{fhi:.2f}]")
    log("    PRE-REGISTERED (before any r2 seen): "
        f"width rho={SWEEP_WIDTH_RATIO}; violation band = LOWEST center with stage-A r2 < "
        f"{SWEEP_R2_CUTOFF} on ALL seeds {seeds}.")
    log(f"    centers (rad/s): {[round(c,2) for c in centers]} "
        f"(c_max={c_max:.2f} -> upper edge {c_max*SWEEP_WIDTH_RATIO**0.5:.2f} "
        f"<= {SWEEP_CMAX_NYQ_FRAC}*Nyquist)")

    t0 = time.perf_counter()
    anchors = [("compliant", (MSG_LO, MSG_HI)), ("orig_violation", (VIOL_LO, VIOL_HI))]
    swept_specs = [(f"c={c:.2f}", _band_from_center(c)) for c in centers]
    rows = []
    for label, (wlo, whi) in anchors + swept_specs:
        r2s = [_stage_a_r2(s, wlo, whi, geom) for s in seeds]
        below = all(r < SWEEP_R2_CUTOFF for r in r2s)
        rows.append({"label": label, "band": [wlo, whi], "center": (wlo * whi) ** 0.5,
                     "r2_per_seed": r2s, "r2_mean": float(np.mean(r2s)),
                     "r2_max": float(np.max(r2s)), "all_below_cutoff": bool(below),
                     "overlaps_fast_tertile": bool(whi > flo)})
        log(f"    {label:16s} band [{wlo:5.2f},{whi:6.2f}] c={rows[-1]['center']:5.2f} "
            f"r2 {[round(r,3) for r in r2s]} mean {np.mean(r2s):+.3f}"
            f"{'  <cutoff ALL' if below else ''}{'  [overlaps fast]' if whi > flo else ''}")

    swept = [r for r in rows if r["label"].startswith("c=")]      # increasing center order
    chosen = next((r for r in swept if r["all_below_cutoff"]), None)
    if chosen is not None:
        mech, mech_detail = _mechanism(chosen["band"], flo, fhi)
    else:
        mech, mech_detail = ("NO-BAND-FOUND",
                             f"no wide (rho={SWEEP_WIDTH_RATIO}) band below {SWEEP_R2_CUTOFF} "
                             f"on all seeds within the Nyquist-representable range "
                             f"(top center {c_max:.2f}); the slow readout tracks wide messages "
                             f"across the whole representable band -- the 'untrackable wide band' "
                             f"premise fails and the control needs re-posing, not just re-centering.")

    log("\n    --- DERIVED (per the pre-registered rule) ---")
    if chosen is not None:
        log(f"    violation band = {chosen['label']} -> [{chosen['band'][0]:.3f},"
            f"{chosen['band'][1]:.3f}] rad/s (stage-A r2 {[round(r,3) for r in chosen['r2_per_seed']]})")
        log(f"    mechanism tested: {mech} -- {mech_detail}")
    else:
        log(f"    NO BAND SELECTED. {mech_detail}")

    hashes = {"code": _sha12(os.path.abspath(__file__)),
              "spec": _sha12(os.path.join(os.path.dirname(__file__), "..",
                                          "relay_gate0_spec.md"))}
    payload = {"stage": "violation-recalibration-sweep", "seeds": seeds, "K": K_PRIMARY,
               "prereg": {"width_rho": SWEEP_WIDTH_RATIO, "r2_cutoff": SWEEP_R2_CUTOFF,
                          "rule": "lowest swept center with stage-A r2 < cutoff on ALL seeds"},
               "geometry": {"nyquist": nyq, "slow_tertile": [slo, shi],
                            "guard": [shi, flo], "fast_tertile": [flo, fhi], "c_max": c_max},
               "rows": rows,
               "chosen_band": chosen["band"] if chosen else None,
               "chosen_center": chosen["center"] if chosen else None,
               "mechanism": mech, "mechanism_detail": mech_detail,
               "hashes": hashes, "env": _env_versions(),
               "wall_clock_s": time.perf_counter() - t0}
    _dump_json(os.path.join(RESDIR, "gate0_bandsweep.json"), payload)
    _write_addendum(os.path.join(os.path.dirname(__file__), "..",
                                 "relay_gate0_spec_addendum.md"), payload)
    log(f"\n    [written -> {os.path.relpath(os.path.join(RESDIR, 'gate0_bandsweep.json'))} "
        f"+ relay_gate0_spec_addendum.md]  (DRAFT; NOT committed)")
    log("    STOP -- addendum is a DRAFT pending Jason's ratification; no re-run until his word.")
    return payload


def _write_addendum(path, sw):
    g, pr = sw["geometry"], sw["prereg"]
    chosen, mech = sw["chosen_band"], sw["mechanism"]
    lines = [
        "# Relay Gate-0 -- Spec Addendum: violation-band re-calibration (DRAFT)",
        "",
        "**Status: DRAFT, pending Jason's ratification.** Not committed. Amends the "
        "bandwidth-violation control (spec condition 4) after the Stage-3 full gate returned "
        "**NO-MEASUREMENT** -- the original violation band [2,9] rad/s was *attenuated but "
        "tracked* (violation e2e 0.664 vs compliant 0.926; stage-A r2 ~0.79-0.92), so its "
        "premise ('a band the slow readout cannot envelope-track') was empirically false.",
        "",
        f"Provenance: sweep artifact results/R/gate0_bandsweep.json; harness sha256 "
        f"{sw['hashes']['code']}; spec sha256 {sw['hashes']['spec']}; "
        f"env {sw['env'].get('jax')}/{sw['env'].get('device')}.",
        "",
        "## Pre-registered selection rule (fixed in source BEFORE the sweep -- band-shopping guard)",
        "",
        f"- Swept band width **rho = {pr['width_rho']}** (= the standard message width; "
        "compliant [0.2,0.9] and original violation [2,9] both have hi/lo = 4.5), so every "
        "swept band is a candidate violation band of the same shape.",
        f"- **Derived violation band = the lowest swept center whose stage-A r2 < "
        f"{pr['r2_cutoff']} on ALL {len(sw['seeds'])} seeds {sw['seeds']}.**",
        f"- **Collapse bar for the re-run: unchanged** = {pr['collapse_bar']}.",
        f"- Representability: top band upper edge <= 0.95 x Nyquist ({g['nyquist']:.2f} rad/s); "
        f"c_max = {g['c_max']:.2f}. (A wide rho=4.5 band cannot be centered above ~14 without "
        "crossing Nyquist -- reaching center 28 would require narrowing the band, a different "
        "probe; flagged, not silently done.)",
        "",
        "## Band geometry (span 1.5)",
        "",
        f"slow tertile [{g['slow_tertile'][0]:.2f}, {g['slow_tertile'][1]:.2f}] | "
        f"guard ({g['guard'][0]:.2f}, {g['guard'][1]:.2f}) | "
        f"fast tertile [{g['fast_tertile'][0]:.2f}, {g['fast_tertile'][1]:.2f}] | "
        f"Nyquist {g['nyquist']:.2f} rad/s.",
        "",
        "## Measured curve (stage-A r2 of an AM message in each band; 3 seeds; K=0.24)",
        "",
        "| band | [w_lo, w_hi] rad/s | center | r2 mean | r2 per seed | < cutoff (all)? | overlaps fast? |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sw["rows"]:
        lines.append(
            f"| {r['label']} | [{r['band'][0]:.2f}, {r['band'][1]:.2f}] | {r['center']:.2f} | "
            f"{r['r2_mean']:+.3f} | {', '.join(f'{x:+.3f}' for x in r['r2_per_seed'])} | "
            f"{'YES' if r['all_below_cutoff'] else 'no'} | "
            f"{'yes' if r['overlaps_fast_tertile'] else 'no'} |")
    lines += ["", "## Derived violation band + mechanism", ""]
    if chosen is not None:
        lines += [
            f"- **Violation band (derived) = [{chosen[0]:.3f}, {chosen[1]:.3f}] rad/s** "
            f"(center {sw['chosen_center']:.2f}).",
            f"- **Mechanism this band tests: {mech}.** {sw['mechanism_detail']}",
            "",
            "## Re-run plan (on ratification)",
            "",
            "Re-run the violation arm ONLY (stage A -> repeater -> stage B, K=0.24, the "
            "verdict-intersection seeds) with MSG_BAND set to the derived band; leave every "
            "other arm's committed numbers untouched; re-read the gate on the UNCHANGED "
            "PASS/FAIL/NO-MEASUREMENT mapping. If the violation e2e now collapses below the "
            "pinned bar, the instrument is sound and the quarantined relay-vs-direct verdict "
            "is released; if it still does not collapse, the deeper negative stands.",
        ]
        if mech != "readout-envelope-limit (clean)":
            lines += [
                "",
                "**Caveat (carried per Jason):** the derived band overlaps the fast tertile, "
                "so a collapse here would test injection ill-posedness (message colliding with "
                "the fast carriers) at least as much as the readout-envelope limit. The "
                "addendum states this is the mechanism the chosen band tests; if a "
                "clean readout-limit test is wanted, the control likely needs re-posing "
                "(e.g. tie it to a measured per-stage envelope bandwidth) rather than "
                "re-centering.",
            ]
    else:
        lines += [
            f"- **NO band selected.** {sw['mechanism_detail']}",
            "",
            "This is itself a finding: within the Nyquist-representable range, a "
            "standard-width message stays trackable by the slow readout at every center, so "
            "the bandwidth-violation control as posed (a wide, untrackable band) cannot be "
            "realized by re-centering. Options for Jason: (i) re-pose the control to a "
            "narrow near-Nyquist probe; (ii) tie the violation to a measured per-stage "
            "envelope bandwidth; (iii) drop the control and justify the instrument another "
            "way. No re-run until the control is re-posed.",
        ]
    lines += ["", "## Scope", "",
              "Measurement + proposed instrument fix only. No relay verdict is read or moved "
              "here; the Stage-3 relay-vs-direct numbers remain QUARANTINED until the violation "
              "instrument is resolved and re-run clean (caught != fixed)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def violation_rerun(log):
    """Re-posed violation-arm-only re-run + gate re-read (addendum step, on Jason's word).
    Loads the Stage-3 NM record (gate0_relay.json -- UNTOUCHED) for every settled arm, re-runs
    ONLY the re-posed repeater-filter violation on GPU, re-decides on the unchanged
    PASS/FAIL/NM mapping (with the re-posed per-seed collapse bar), and writes a NEW artifact
    gate0_relay_reposed.{json,md}. Quarantine on relay-vs-direct lifts IFF the control collapses."""
    import time
    src = os.path.join(RESDIR, "gate0_relay.json")
    assert os.path.exists(src), f"missing Stage-3 NM record {src} -- run the full gate first"
    nm = json.load(open(src))
    recs = {k: {int(s): r for s, r in v.items()} for k, v in nm["recs"].items()}
    seeds = nm["seeds"]
    geom = _geom(STAGE_SPAN)
    log(f"=== RELAY GATE-0 :: RE-POSED VIOLATION RE-RUN (seeds {seeds}; NM record untouched) ===")
    t0 = time.perf_counter()
    for i in seeds:
        recs["violation"][i] = _reposed_violation(i, geom, log)
    # integrity guarantee (Jason's verification #2): every SETTLED arm re-serializes
    # byte-identically to the NM record -- only the violation arm changed.
    for arm in recs:
        if arm == "violation":
            continue
        out_arm = {str(s): r for s, r in recs[arm].items()}
        assert json.dumps(out_arm, sort_keys=True) == json.dumps(nm["recs"][arm], sort_keys=True), \
            f"settled arm '{arm}' drifted from the NM record -- re-run aborted"
    log(f"  [integrity] settled arms (relay/direct/relay_k16/direct_k16/scramble) "
        f"value-identical to NM record: OK")
    verdict = decide(recs, seeds)
    secondary = _secondary_summary(recs, seeds)
    repl_rows = _replication_table(recs, seeds)      # unchanged arms -> still expect 32/32
    wall = time.perf_counter() - t0
    hashes = {"code": _sha12(os.path.abspath(__file__)),
              "spec": _sha12(os.path.join(os.path.dirname(__file__), "..", "relay_gate0_spec.md"))}
    outp = os.path.join(RESDIR, "gate0_relay_reposed.json")
    _dump_json(outp, {"stage": "3-full-gate-reposed-violation", "seeds": seeds,
                      "framing": FRAMING, "reposed_from": "gate0_relay.json (Stage-3 NM; untouched)",
                      "K_primary": K_PRIMARY, "K_secondary": K_SECONDARY,
                      "m0_spec": nm.get("m0_spec"), "hashes": hashes, "env": _env_versions(),
                      "wall_clock_s": wall, "verdict": verdict, "secondary_k16": secondary,
                      "replication_table": repl_rows,
                      "recs": {k: {str(s): r for s, r in v.items()} for k, v in recs.items()}})
    _write_md(os.path.join(RESDIR, "gate0_relay_reposed.md"), verdict, secondary, repl_rows,
              recs, seeds, wall, hashes)
    log("\n=== RE-POSED GATE VERDICT ===")
    log(f"  {verdict['verdict']}")
    if "violation" in verdict:
        vv = verdict["violation"]
        log(f"  re-posed violation: e2e/seed {vv.get('e2e_per_seed')} (bar < {vv['bar_per_seed']}) "
            f"-> collapsed={vv['collapsed']}; signature rms_in {vv.get('signature_rms_in')} "
            f"scale {vv.get('signature_scale')}")
    n_match = sum(r["match_6dp"] for r in repl_rows)
    log(f"  replication table: {n_match}/{len(repl_rows)} (unchanged arms)")
    log(f"  [written -> {os.path.relpath(outp)} + gate0_relay_reposed.md]  (NM record UNTOUCHED; NOT committed)")
    log("  Quarantine lifts IFF collapsed AND verdict is PASS. STOP-and-report.")
    return verdict


def run(log, nseeds):
    import time
    seeds = list(range(nseeds))
    log(f"=== RELAY GATE-0 :: STAGE-3 FULL GATE (seeds {seeds}, K_primary={K_PRIMARY}) ===")
    t0 = time.perf_counter()
    geom = _geom(STAGE_SPAN)
    geom3 = _geom(DIRECT_SPAN)
    log(f"  span-{STAGE_SPAN} window: dt_in={geom[0]:.5g} L={geom[2]} eval_start={geom[1]} | "
        f"span-{DIRECT_SPAN}: dt_in={geom3[0]:.5g} L={geom3[2]} eval_start={geom3[1]}")
    recs = {"relay": {}, "relay_k16": {}, "violation": {}, "scramble": {},
            "direct": {}, "direct_k16": {}}
    outp = os.path.join(RESDIR, "gate0_relay.json")
    hashes = {"code": _sha12(os.path.abspath(__file__)),
              "spec": _sha12(os.path.join(os.path.dirname(__file__), "..",
                                          "relay_gate0_spec.md"))}
    m0_spec = {"generator": "p1.slow_bandlimited: 6 log-uniform sinusoids in msg band, "
                            "min-max mapped to [0.1,1.0]",
               "am": "u = 0.5*sqrt(s)*w, w = Rademacher(seed+777) -> u^2 = 0.25*s exact",
               "msg_band": [MSG_LO, MSG_HI], "seed_base": 1000,
               "N_MSG_PERIODS": p1.N_MSG_PERIODS,
               "e2e_decoy_base": E2E_DECOY_BASE, "stage_decoy_bases": {"A": 40000, "B": 60000}}

    for i in seeds:
        ts = time.perf_counter()
        arm_recs = relay_seed(i, geom, log)
        for tag in ("relay", "relay_k16", "violation", "scramble"):
            recs[tag][i] = arm_recs[tag]
        drecs = direct_seed(i, geom3, log)
        recs["direct"][i] = drecs["direct"]
        recs["direct_k16"][i] = drecs["direct_k16"]
        log(f"  seed {i}: done ({time.perf_counter() - ts:.0f}s; "
            f"{time.perf_counter() - t0:.0f}s elapsed)")
        _dump_json(outp, {"stage": "3-full-gate", "seeds_done": seeds[:i + 1],
                          "framing": FRAMING, "K_primary": K_PRIMARY,
                          "K_secondary": K_SECONDARY, "m0_spec": m0_spec,
                          "hashes": hashes, "env": _env_versions(),
                          "recs": {k: {str(s): r for s, r in v.items()}
                                   for k, v in recs.items()}})

    verdict = decide(recs, seeds)
    secondary = _secondary_summary(recs, seeds)
    repl_rows = _replication_table(recs, seeds)
    wall = time.perf_counter() - t0
    payload = {"stage": "3-full-gate", "seeds": seeds, "framing": FRAMING,
               "K_primary": K_PRIMARY, "K_secondary": K_SECONDARY, "m0_spec": m0_spec,
               "hashes": hashes, "env": _env_versions(), "wall_clock_s": wall,
               "verdict": verdict, "secondary_k16": secondary,
               "replication_table": repl_rows,
               "recs": {k: {str(s): r for s, r in v.items()} for k, v in recs.items()}}
    _dump_json(outp, payload)
    mdp = os.path.join(RESDIR, "gate0_relay.md")
    _write_md(mdp, verdict, secondary, repl_rows, recs, seeds, wall, hashes)

    log("\n=== STAGE-3 VERDICT ===")
    log(f"  {verdict['verdict']}")
    if "anchor" in verdict:
        a = verdict["anchor"]
        log(f"  anchor: mean={a['stageA']['mean']:.6f} target={ANCHOR} +/- {a['window']:.4f} "
            f"-> {'OK' if a['ok'] else 'MISS'}")
    if "violation" in verdict:
        vv = verdict["violation"]
        log(f"  violation: {vv['violation_e2e_mean']} vs compliant "
            f"{vv['compliant_e2e_mean']} -> collapsed={vv['collapsed']}")
    n_match = sum(r["match_6dp"] for r in repl_rows)
    log(f"  replication table: {n_match}/{len(repl_rows)} rows match committed to 6dp")
    log(f"  [written -> {os.path.relpath(outp)} + {os.path.relpath(mdp)}]  (NOT committed)")
    log("  STOP-and-report. No sweep, no multi-hop, no interpretation beyond the mapping.")
    return verdict


# ===================================================================================== #
#  Synthetic verdict-engine test (CPU; the Phase-2 lesson -- test decide() BEFORE the GPU)
# ===================================================================================== #
def _mk(seed, r2A=0.9861, espA=True, r2B=0.99, espB=True, e2e=0.96, e2e_dec=0.03,
        decA=-0.13, decB=-0.10, direct=0.0, espD=True, viol=0.02, espVA=True, espVB=True):
    relay = {"seed": seed, "stageA": {"r2": r2A, "esp": {"ok_slow": espA},
                                      "demod": {"decoy_p95": decA}},
             "stageB": {"r2": r2B, "esp": {"ok_slow": espB}, "demod": {"decoy_p95": decB}},
             "e2e": {"r2": e2e, "decoy_p95": e2e_dec}}
    direct_r = {"seed": seed, "r2": direct, "esp": {"ok_slow": espD}}
    viol_r = {"seed": seed, "stageA": {"esp": {"ok_slow": espVA}},
              "stageB": {"esp": {"ok_slow": espVB}, "r2": 0.9},   # stage B reconstructs residual
              "e2e": {"r2": viol},
              "repeater": {"rms_in": 1e-3, "scale": 180.0}}   # re-posed signature (small/large)
    return relay, direct_r, viol_r


def verdict_test(log):
    log("=== VERDICT-ENGINE SYNTHETIC TEST (11 cases; CPU only) ===")
    rng = np.random.default_rng(0)

    def build(mod):
        recs = {"relay": {}, "direct": {}, "violation": {}}
        for s in range(5):
            jit = 0.004 * rng.standard_normal()          # seed-level jitter -> finite sd
            r, d, v = _mk(s, r2A=0.9861 + jit, e2e=0.96 + jit, direct=0.0 + 0.01 * jit)
            recs["relay"][s], recs["direct"][s], recs["violation"][s] = r, d, v
        mod(recs)
        return decide(recs, list(range(5)))

    cases = []
    # 1 healthy PASS
    cases.append(("healthy -> PASS", build(lambda r: None), "PASS"))
    # 2 underpowered (all direct ESP fail)
    def m2(r):
        for s in r["direct"]:
            r["direct"][s]["esp"]["ok_slow"] = False
    cases.append(("all-direct-ESP-fail -> NM underpowered", build(m2), "add seeds"))
    # 3 anchor low miss
    def m3(r):
        for s in r["relay"]:
            r["relay"][s]["stageA"]["r2"] -= 0.05
    cases.append(("anchor low -> NM replication", build(m3), "low-side"))
    # 4 anchor high + elevated decoy -> leakage side
    def m4(r):
        for s in r["relay"]:
            r["relay"][s]["stageA"]["r2"] += 0.05
            r["relay"][s]["stageA"]["demod"]["decoy_p95"] = 0.5
    cases.append(("anchor high + decoy -> NM leakage", build(m4), "leakage"))
    # 5 decoy elevated (anchor fine)
    def m5(r):
        for s in r["relay"]:
            r["relay"][s]["e2e"]["decoy_p95"] = 0.5
    cases.append(("e2e decoy elevated -> NM leak", build(m5), "decoy elevated"))
    # 6 violation does not collapse
    def m6(r):
        for s in r["violation"]:
            r["violation"][s]["e2e"]["r2"] = 0.9
    cases.append(("violation no-collapse -> NM instrument", build(m6), "did NOT collapse"))
    # 7 relay at floors -> FAIL
    def m7(r):
        for s in r["relay"]:
            r["relay"][s]["e2e"]["r2"] = 0.01 + 0.004 * rng.standard_normal()
    cases.append(("relay at floors -> FAIL", build(m7), "FAIL"))
    # 8 beats-but-within-sigma -> FAIL (deltas positive, mean < sd)
    def m8(r):
        deltas = [0.001, 0.02, -0.015, 0.018, -0.012]
        for s in r["relay"]:
            r["relay"][s]["e2e"]["r2"] = r["direct"][s]["r2"] + deltas[s]
            r["relay"][s]["e2e"]["decoy_p95"] = r["relay"][s]["e2e"]["r2"] - deltas[s]
        for s in r["violation"]:
            r["violation"][s]["e2e"]["r2"] = 0.0
    cases.append(("within-sigma -> FAIL", build(m8), "FAIL"))
    # 9 attrition PASS (one seed fails stage B; rest healthy)
    def m9(r):
        r["relay"][3]["stageB"]["esp"]["ok_slow"] = False
    v9 = build(m9)
    cases.append(("attrition (seed 3 drops) -> PASS w/ n=4", v9, "PASS"))
    # 10 anchor high WITHOUT decoy elevation -> still NM (plain high-side label)
    def m10(r):
        for s in r["relay"]:
            r["relay"][s]["stageA"]["r2"] += 0.05
    cases.append(("anchor high, no leak -> NM high-side", build(m10), "high-side: mean"))
    # 11 beats decoy but NOT direct -> FAIL naming direct
    def m11(r):
        deltas = [0.001, 0.02, -0.015, 0.018, -0.012]
        for s in r["relay"]:
            r["relay"][s]["e2e"]["r2"] = r["direct"][s]["r2"] + deltas[s]
            r["relay"][s]["e2e"]["decoy_p95"] = -0.5
        for s in r["violation"]:
            r["violation"][s]["e2e"]["r2"] = 0.0
    cases.append(("beats decoy only -> FAIL (direct named)", build(m11), "beat direct"))

    allok = True
    for name, v, want in cases:
        got = v["verdict"]
        ok = want in got
        if name.startswith("attrition"):
            ok = ok and v["n_intersection"] == 4 and 3 not in v["intersection"]
        if name.startswith("all-direct"):
            ok = ok and v["n_intersection"] == 0
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {got[:90]}")

    # the .md writer must survive every verdict shape (Stage-3 review MAJOR: the
    # underpowered early-return once crashed it) -- exercise both shapes end-to-end.
    import tempfile
    for tag, vv in (("underpowered", cases[1][1]), ("healthy", cases[0][1])):
        p = os.path.join(tempfile.gettempdir(), f"_vt_md_{tag}.md")
        try:
            _write_md(p, vv, {"n": 0, "underpowered": True}, [], {}, list(range(5)), 0.0,
                      {"code": "selftest", "spec": "selftest"})
            os.remove(p)
            log(f"  [OK] _write_md survives the {tag} verdict shape")
        except Exception as e:
            allok = False
            log(f"  [WRONG] _write_md crashed on the {tag} verdict shape: {e!r}")
    log(f"  VERDICT-ENGINE: {'ALL BRANCHES CORRECT' if allok else 'BRANCH ERRORS -- FIX'}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--band-sweep", action="store_true",
                    help="Stage-3 NM follow-up: measure the slow-readout tracking cutoff")
    ap.add_argument("--violation-rerun", action="store_true",
                    help="re-posed violation-arm-only re-run + gate re-read (on Jason's word)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--nseeds", type=int, default=8,
                    help="spec n>=5; default 8 buys direct span-3 ESP-attrition margin")
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)

    def log(msg):
        print(msg)

    if args.sandbox:
        ok = sandbox(log)
        raise SystemExit(0 if ok else 1)
    if args.smoke:
        ok = smoke(log)
        raise SystemExit(0 if ok else 1)
    if args.verdict_test:
        ok = verdict_test(log)
        raise SystemExit(0 if ok else 1)
    if args.band_sweep:
        band_sweep(log)
        return
    if args.violation_rerun:
        violation_rerun(log)
        return
    if args.run:
        assert 1 <= args.nseeds <= REF_SEED_MAX + 1, \
            f"REF_TABLE covers committed seeds 0-{REF_SEED_MAX} (Phase-1 ran 10)"
        run(log, args.nseeds)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
