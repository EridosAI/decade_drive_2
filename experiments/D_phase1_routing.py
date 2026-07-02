"""
experiments/D_phase1_routing.py
===============================

Experiment D -- Phase 1: cross-band routing-efficiency LANDSCAPE.

Built on the VALIDATED Gate-0 AM instrument (which PASSED: the slow-band readout recovered
a designed slow message carried purely in the fast input's POWER, via the |z|^2 square-law
channel, R^2/|z|^2-only = 0.978 at span 1). Phase 1 turns that into a quantitative map.

PRIMARY (the routing headline) -- AM designed-message demodulation:
  Inject u = 0.5*sqrt(s)*w into the FAST band only (w = +-1 Rademacher carrier, so
  u^2 = 0.25*s EXACTLY -> the slow message s is carried purely in the fast POWER, a clean
  square-law channel with no linear path). s is a SLOW (sub-omega_min) band-limited message.
  Read the SLOW band only; reconstruct s(t-delay). Metric = R^2_det (calibration-honest).
  - decoy null: an INDEPENDENT message s' (same spectrum) NEVER injected -> chance floor
    (controls 'a rich slow basis fits any smooth target'); PASS = real >> decoy.
  - |z|^2-only vs Re/Im-only readout ablation -> the demodulation MECHANISM (square-law).
  - HEADLINE: demod R^2 vs SEPARATION (span) -- the attenuation-with-separation curve
    Gate-0 found (0.98 span1 -> 0.37 span2), now a full landscape vs (K, span).

(A white-i.i.d. Dambre-IPC secondary was DROPPED -- its positive control failed (the fast
band could not recover its own injected input's deg-1 at span >= 2, so the deg-1/deg-2
single-delay readings were uninformative). The AM measure above is the clean deliverable.)

WINDOW: matched-message-periods (N_MSG_PERIODS periods of the slowest message component) so
the demod estimate is fair across spans -- the same message occupies ~10x more samples per
decade as dt_in shrinks (the matched-samples-vs-matched-periods tension; this is the
matched-periods convention). STREAMING: span-3 trajectories are large; each (span,seed,K) is
integrated, reduced, and discarded.

Inherited unchanged: substrate/build_system/washout (core.reservoir); co-rotating input
integrator eps=0 (core.integrator_corotating); band partition + masked injection
(core.bands); Dambre targets + OOS capacity + Kubota threshold (core.ipc); ESP
(core.consistency). The AM demod fitter (inner-val lambda, bias-protected intercept, decoy
null) is the validated Gate-0 measure.

Run (ONE GPU process; float64; x64 ON):
  python experiments/D_phase1_routing.py --sandbox
  python experiments/D_phase1_routing.py --run
  python experiments/D_phase1_routing.py --run --spans 1.0 2.0 --nseeds 3   # verify run
STOP-and-report after --run. Nothing committed.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.reservoir import build_system, dt_in_for, washout_samples, build_graph
from core.magnus import graph_laplacian
from core.integrator_corotating import recommended_h, integrate_corotating_input_batch
from core.consistency import replica_spec, consistency_distance, ESP_EPS
from core.bands import band_indices, masked_encoding, band_summary

RESDIR = os.path.join(os.path.dirname(__file__), "..", "results", "D")

SPP = 2
N = 500
BETA = 1.0
K_OP = 0.16
SPANS = [1.0, 1.5, 2.0, 2.5, 3.0]
K_GRID = [0.0, 0.08, 0.12, 0.16, 0.24]
NSEEDS = 10
MSG_LO, MSG_HI = 0.2, 0.9          # slow message band (sub-omega_min; Gate-0's choice)
N_MSG_PERIODS = 12                 # eval window covers this many slowest-message periods
N_DELAYS = 4                       # message reconstruction at [0, stride, 2*stride, 3*stride]
N_DEC = 60                         # decoy-null draws (specificity / chance floor)
TRAIN_FRAC = 0.7
ENV_LAMS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def n_sub_for(span):
    om = build_system(0, N, span).omega
    dt_in = dt_in_for(10.0 ** span, SPP)
    h_rec = recommended_h(0.0, resolve_freqs=[om.max() - om.min()], spp_safe=16)
    return max(1, int(np.ceil(dt_in / h_rec)))


# --------------------------------------------------------------------------- #
# AM input: a slow message carried in the fast carrier's POWER (Gate-0, validated)
# --------------------------------------------------------------------------- #
def slow_bandlimited(L, dt_in, w_lo, w_hi, seed, n_sin=6, out_lo=0.1, out_hi=1.0):
    rng = np.random.default_rng(seed)
    t = (np.arange(L) + 1.0) * dt_in
    freqs = np.exp(rng.uniform(np.log(w_lo), np.log(w_hi), n_sin))
    phases = rng.uniform(0, 2 * np.pi, n_sin)
    s = np.zeros(L)
    for f, ph in zip(freqs, phases):
        s += np.sin(f * t + ph)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return out_lo + (out_hi - out_lo) * s


def am_input(L, dt_in, seed):
    """u = 0.5*sqrt(s)*w, w = Rademacher +-1 -> u^2 = 0.25*s exactly (pure square-law)."""
    s = slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=seed)
    w = np.random.default_rng(seed + 777).choice([-1.0, 1.0], size=L)
    return s, 0.5 * np.sqrt(s) * w


def msg_period_samples(dt_in):
    return (2.0 * np.pi / MSG_LO) / dt_in        # slowest message component, in samples


# --------------------------------------------------------------------------- #
# Demod fit: reconstruct the slow message s from a readout band (the validated Gate-0
# measure -- inner-val lambda, BIAS-PROTECTED intercept, decoy null)
# --------------------------------------------------------------------------- #
def r2_det(yh, y):
    return float(1.0 - np.sum((yh - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-300))


def band_features(X, idx, mode):
    Z = X[:, idx]
    Re, Im, P = Z.real, Z.imag, np.abs(Z) ** 2
    one = np.ones((X.shape[0], 1))
    if mode == "full":
        return np.concatenate([Re, Im, P, one], axis=1)
    if mode == "reim":
        return np.concatenate([Re, Im, one], axis=1)
    if mode == "pow":
        return np.concatenate([P, one], axis=1)
    raise ValueError(mode)


def demod_capacity(X, idx, s_real, decoys_s, delays, sl, mode="full"):
    """R^2_det of reconstructing the slow message s(t-delay) from readout band `idx`.
    Lambda chosen by inner validation on the delay-0 target (a fixed tiny lambda overfits
    the smooth message); bias column protected so the nonzero-mean message keeps its DC;
    decoy null = independent never-injected messages (specificity floor)."""
    iw = np.arange(sl.start, sl.stop)
    F = band_features(X, idx, mode)[iw]
    ntr = int(TRAIN_FRAC * F.shape[0])
    Ftr, Fte = F[:ntr], F[ntr:]
    mu, sd = Ftr.mean(0), Ftr.std(0)
    const = sd < 1e-7
    sd = sd + 1e-8; sd[const] = 1.0; mu[const] = 0.0     # protect bias -> keep intercept
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    Fp = Ftr.shape[1]
    y0 = s_real[iw][:ntr]
    nval = max(1, int(0.2 * ntr))
    inn, va = slice(0, ntr - nval), slice(ntr - nval, ntr)
    GTG, FTy = Ftr[inn].T @ Ftr[inn], Ftr[inn].T @ y0[inn]
    best_lam, best = ENV_LAMS[0], np.inf
    for lam in ENV_LAMS:
        W = np.linalg.solve(GTG + lam * np.eye(Fp), FTy)
        err = np.mean((Ftr[va] @ W - y0[va]) ** 2) / (np.var(y0[va]) + 1e-12)
        if err < best:
            best, best_lam = err, lam
    G = np.linalg.inv(Ftr.T @ Ftr + best_lam * np.eye(Fp))
    fitr2 = lambda y: r2_det(Fte @ (G @ (Ftr.T @ y[:ntr])), y[ntr:])
    real = [fitr2(np.roll(s_real, k)[iw]) for k in delays]
    dec = [fitr2(d[iw]) for d in decoys_s]
    return {"cap": float(np.sum(np.clip(real, 0.0, None))), "r2_d0": float(real[0]),
            "lam": float(best_lam), "r2_by_delay": [float(x) for x in real],
            "decoy_p95": float(np.percentile(dec, 95)), "decoy_mean": float(np.mean(dec))}


# --------------------------------------------------------------------------- #
def integrate_Ks(omega, Lmat, m, z0, u, Ks, dt_in, n_sub):
    """Batch the K-grid (throughput), chunked so the GPU output (chunk*L*N*16 bytes) stays
    bounded (~3 GB) -- at span 3 L~1.3e5 so B=5 would be ~5 GB; this auto-chunks to ~2."""
    Ln, Nn = len(u), len(omega)
    maxb = max(1, int(3e9 / (Ln * Nn * 16)))
    out = []
    for s in range(0, len(Ks), maxb):
        e = min(s + maxb, len(Ks))
        bb = e - s
        rep = lambda a: np.repeat(np.asarray(a)[None], bb, axis=0)
        _, zs = integrate_corotating_input_batch(
            rep(omega), rep(Lmat), rep(m), rep(z0), rep(u),
            K0s=np.asarray(Ks[s:e], float), epss=np.zeros(bb), Omegas=np.zeros(bb),
            betas=BETA * np.ones(bb), dt_in=dt_in, n_sub=n_sub)
        out.extend(np.asarray(zs[j]) for j in range(bb))
    return out


def scramble_laplacian(N_, seed):
    return graph_laplacian(build_graph("er", N_, 10.0, np.random.default_rng(70000 + seed)))


def am_window(span, n_msg=N_MSG_PERIODS, n_eval_extra=0):
    """dt_in, W0, eval_start, L for the AM demod: eval window covers n_msg slowest-message
    periods (matched across spans); eval_start past washout + max message delay."""
    om_max = 10.0 ** span
    dt_in = dt_in_for(om_max, SPP)
    W0 = washout_samples(1.0, om_max, dt_in) + 20
    stride = max(1, int(round((2.0 * np.pi / 1.0) / (2.0) / dt_in)))  # ~half slow period
    delays = [k * stride for k in range(N_DELAYS)]
    max_delay = delays[-1]
    n_eval = int(round(n_msg * msg_period_samples(dt_in)))
    eval_start = max(W0, max_delay)
    L = eval_start + n_eval
    return dt_in, W0, eval_start, L, delays, stride


# --------------------------------------------------------------------------- #
def run(spans, nseeds, log, outfile):
    allpts = []
    for span in spans:
        n_sub = n_sub_for(span)
        dt_in, W0, eval_start, L, delays, stride = am_window(span)
        sl = slice(eval_start, L)
        specs = [build_system(s, N, span) for s in range(nseeds)]
        bands = band_indices(specs[0].omega)
        rngs = [np.random.default_rng(5000 + s) for s in range(nseeds)]
        m_fast = [masked_encoding(sp.omega, bands["fast"], r) for sp, r in zip(specs, rngs)]
        outside = np.concatenate([bands["slow"], bands["guard"]])
        assert max(float(np.abs(mf[outside]).max()) for mf in m_fast) == 0.0
        log(f"\n===== SPAN {span} (N={N}, dt_in={dt_in:.4g}, n_sub={n_sub}, AM L={L}, "
            f"eval_start={eval_start}, delays={delays}) =====")
        log("  " + band_summary(specs[0].omega, bands))

        t0 = time.perf_counter()
        for i in range(nseeds):
            sp = specs[i]
            s_msg, u_am = am_input(L, dt_in, 1000 + i)
            decoys_s = [slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=40000 + i * 200 + d)
                        for d in range(N_DEC)]
            rep = replica_spec(sp, 9000 + i)
            Xmain = integrate_Ks(sp.omega, sp.L, m_fast[i], sp.z0, u_am, K_GRID, dt_in, n_sub)
            Xrep = integrate_Ks(sp.omega, sp.L, m_fast[i], rep.z0, u_am, K_GRID, dt_in, n_sub)
            for ki, K in enumerate(K_GRID):
                X, Xr = Xmain[ki], Xrep[ki]
                demod = demod_capacity(X, bands["slow"], s_msg, decoys_s, delays, sl, "full")
                d_full = consistency_distance(X, Xr, sl)
                d_slow = consistency_distance(X[:, bands["slow"]], Xr[:, bands["slow"]], sl)
                rec = {"span": span, "seed": i, "K": K, "demod": demod,
                       "esp": {"d_full": d_full, "d_slow": d_slow,
                               "ok_slow": bool(d_slow < ESP_EPS)},
                       "ablation": None, "fast_ctrl": None, "scramble": None}
                if K == K_OP:
                    rec["ablation"] = {
                        "pow": demod_capacity(X, bands["slow"], s_msg, decoys_s, delays, sl,
                                              "pow"),
                        "reim": demod_capacity(X, bands["slow"], s_msg, decoys_s, delays, sl,
                                               "reim")}
                    rec["fast_ctrl"] = demod_capacity(X, bands["fast"], s_msg, decoys_s,
                                                      delays, sl, "full")
                    Lscr = scramble_laplacian(N, i)
                    Xs = integrate_Ks(sp.omega, Lscr, m_fast[i], sp.z0, u_am, [K], dt_in,
                                      n_sub)[0]
                    rec["scramble"] = demod_capacity(Xs, bands["slow"], s_msg, decoys_s,
                                                     delays, sl, "full")
                    del Xs
                allpts.append(rec)
            del Xmain, Xrep
            log(f"  seed {i}: done ({time.perf_counter()-t0:.0f}s elapsed)")
        _report_span(allpts, span, log)
        with open(outfile, "w") as f:
            json.dump({"N": N, "SPANS": spans, "K_GRID": K_GRID, "NSEEDS": nseeds,
                       "MSG_BAND": [MSG_LO, MSG_HI], "N_MSG_PERIODS": N_MSG_PERIODS,
                       "points": allpts}, f, indent=1)
        log(f"  [written {len(allpts)} pts -> {os.path.basename(outfile)}]")
    return allpts


def _report_span(allpts, span, log):
    pts = [p for p in allpts if p["span"] == span]
    log(f"\n  -- span {span} AM-demod routing landscape (ESP-ok at driven K; NaN if none) --")
    log(f"    {'K':>5} {'demodR2_d0':>11} {'cap':>6} {'decoyp95':>9} {'ESP':>6}")
    for K in K_GRID:
        kp = [p for p in pts if p["K"] == K]
        use = kp if K == 0.0 else [p for p in kp if p["esp"]["ok_slow"]]
        espok = sum(1 for p in kp if p["esp"]["ok_slow"])
        if not use:
            log(f"    {K:5.2f} {'NaN (all ESP-fail)':>11} {espok:3d}/{len(kp)}")
            continue
        clip = lambda v: float(max(-9.99, min(1.5, v)))   # K=0 control ridge-extrapolates
        d0 = np.mean([clip(p["demod"]["r2_d0"]) for p in use])
        cap = np.mean([p["demod"]["cap"] for p in use])
        dp = np.mean([clip(p["demod"]["decoy_p95"]) for p in use])
        log(f"    {K:5.2f} {d0:11.3f} {cap:6.2f} {dp:9.3f} {espok:3d}/{len(kp)}")
    op = [p for p in pts if p["K"] == K_OP and p["ablation"]]
    if op:
        fu = np.mean([p["demod"]["r2_d0"] for p in op])
        pw = np.mean([p["ablation"]["pow"]["r2_d0"] for p in op])
        re = np.mean([p["ablation"]["reim"]["r2_d0"] for p in op])
        scr = np.mean([p["scramble"]["r2_d0"] for p in op])
        fc = np.mean([p["fast_ctrl"]["r2_d0"] for p in op])
        loc = "|z|^2-dominant (square-law demod)" if pw > re else "ReIm-dominant"
        log(f"    [K0 mechanism] demod r2_d0: full={fu:.3f} |z|^2-only={pw:.3f} "
            f"ReIm-only={re:.3f} -> {loc} | scramble={scr:.3f} | fast-ctrl={fc:.3f}")


# --------------------------------------------------------------------------- #
def sandbox(log):
    log("=== PHASE-1 v4 SANDBOX (AM scheme, tiny N) ===")
    span, N_s = 2.0, 90
    n_sub = n_sub_for(span)
    dt_in, W0, eval_start, L, delays, stride = am_window(span, n_msg=10)
    sp = build_system(0, N_s, span)
    bands = band_indices(sp.omega)
    sl = slice(eval_start, L)
    r = np.random.default_rng(3)
    mfast = masked_encoding(sp.omega, bands["fast"], r)
    assert np.all(mfast[np.concatenate([bands["slow"], bands["guard"]])] == 0.0)
    s_msg, u_am = am_input(L, dt_in, 7)
    decoys_s = [slow_bandlimited(L, dt_in, MSG_LO, MSG_HI, seed=50000 + d) for d in range(40)]
    X0, X16 = integrate_Ks(sp.omega, sp.L, mfast, sp.z0, u_am, [0.0, 0.16], dt_in, n_sub)
    d0 = demod_capacity(X0, bands["slow"], s_msg, decoys_s, delays, sl, "full")
    d16 = demod_capacity(X16, bands["slow"], s_msg, decoys_s, delays, sl, "full")
    pw = demod_capacity(X16, bands["slow"], s_msg, decoys_s, delays, sl, "pow")
    re = demod_capacity(X16, bands["slow"], s_msg, decoys_s, delays, sl, "reim")
    fc = demod_capacity(X16, bands["fast"], s_msg, decoys_s, delays, sl, "full")
    log(f"  {band_summary(sp.omega, bands)}")
    log(f"  AM L={L} eval_start={eval_start} delays={delays} stride={stride} "
        f"(msg_period={msg_period_samples(dt_in):.0f} samp)")
    log(f"  K=0   demod r2_d0={d0['r2_d0']:.3f} decoy_p95={d0['decoy_p95']:.3f}  (gap~0)")
    log(f"  K=.16 demod r2_d0={d16['r2_d0']:.3f} decoy_p95={d16['decoy_p95']:.3f} "
        f"cap={d16['cap']:.2f}  (gap>0 = routing)")
    log(f"  K=.16 mechanism: |z|^2-only={pw['r2_d0']:.3f} ReIm-only={re['r2_d0']:.3f} "
        f"({'|z|^2-dominant' if pw['r2_d0']>re['r2_d0'] else 'ReIm-dominant'})")
    log(f"  K=.16 FAST-band demod r2_d0={fc['r2_d0']:.3f} (positive control)")
    assert d0["r2_d0"] < d0["decoy_p95"] + 0.15, "K=0 not at decoy floor -> leak!"
    assert d16["r2_d0"] > d16["decoy_p95"] + 0.15, "no AM demod transfer at K=.16"
    log("  v4 sandbox checks passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--spans", type=float, nargs="+", default=SPANS)
    ap.add_argument("--nseeds", type=int, default=NSEEDS)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)
    logf = open(os.path.join(RESDIR, f"_phase1{args.tag}.log"), "a")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    if args.sandbox:
        sandbox(log); return
    if args.run:
        log(f"\n######## D Phase-1 AM routing landscape :: {time.strftime('%Y-%m-%d %H:%M')} "
            f"spans={args.spans} nseeds={args.nseeds} ########")
        out = os.path.join(RESDIR, f"phase1_routing{args.tag}.json")
        run(args.spans, args.nseeds, log, out)
        log(f"\n######## DONE -> {os.path.basename(out)} ########")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
