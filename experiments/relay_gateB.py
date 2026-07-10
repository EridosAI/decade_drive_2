"""
experiments/relay_gateB.py
==========================

Relay Gate-B: broadband trackability mechanism (ANALYSIS gate; retrodiction + locked predictions).
Per relay_gateB_broadband_spec.md. Stage-1 is CPU-ONLY on committed data (no new integrations):
fit the pre-registered models to the committed band-sweep + Gate-3 FULL r2 vs band center, decide
EXPLAINED / SHAPE-UNEXPLAINED / CLIFF-PREFERRED / NO-MEASUREMENT, and emit LOCKED held-out-band
predictions (P1) that make the explanation falsifiable. Stage-2 (6-run GPU probe) is a SEPARATE go
and is NOT implemented in this stage's modes.

Built STRICTLY by REUSE of committed machinery for helpers (imported g0/g1); modifies NOTHING in
relay_gate0/1/2/3.py, D_phase1_routing.py, core/, or any committed artifact. Pure numpy/scipy CPU.

Pre-registration pins (from the ratified spec; no post-hoc moving):
  * inputs: gate0_bandsweep.json (sha256 pinned below) + gate3_mechanism.json (recs sha16 pinned);
    NO-MEASUREMENT on any sha/digit re-verification failure.
  * band coordinate c = sqrt(lo*hi) (committed convention, re-verified against stored centers).
  * DEDUPE: same band + overlapping seeds -> bit-compare; identical -> merge (union of seeds);
    different -> both enter separately AND the discrepancy is reported verbatim.
  * strata: NON-OVERLAP (fit + acceptance; 6 band-points) / OVERLAP (reported, not gating) by the
    committed overlaps_fast_tertile flag (SUPRA assigned from the committed fast-tertile geometry).
  * models: M1 SNR=A*c^-p (2 par, p in [0.5,4]); M2 SNR=A/(4*lambda^2+c^2), lambda=0.1 pinned
    (1 par); M-CLIFF r2=r_top for c<nu_x else 0.05 (2 par). WLS on band means,
    weight_b = 1/max(SE_b, 0.02)^2.
  * acceptance: EXPLAINED iff best-smooth residuals within max(2*SE_b,0.03) on >=5/6 AND within
    max(2*SE_b,0.06) on 6/6 AND wRMSE_CLIFF >= 1.5*wRMSE_best. CLIFF-PREFERRED iff
    wRMSE_CLIFF < wRMSE_best_smooth. Else SHAPE-UNEXPLAINED.
  * P1 holdout bands pinned from (c, rho): H1 (3.30, 4.5), H2 (12.0, 4.5), H3 (sqrt(18), 1.5).
  * stage-2 predictive pass (pinned now): |mean_obs - pred| <= max(2*sigma_pred, 0.05) all three.

Operationalizations RESOLVED IN CODE (surfaced for ratification in the STOP report AND in the
artifact's operationalization block; cov/sigma_pred/H3-center set the LOCKED P1 intervals only,
while the wRMSE definition ALSO sets best-smooth selection and the cliff ratio, i.e. the branch):
  * wRMSE = sqrt( sum_b w_b*resid_b^2 / sum_b w_b )  (weighted, same pinned weights).
  * fit covariance: Cov = (J^T W J)^{-1} * max(1, chi2_red)  (Gauss-Newton at the WLS optimum,
    conservative inflation when underfit; J from the weighted residuals).
  * sigma_pred^2 = g^T Cov g + SE_FLOOR^2   (g = numeric gradient of the prediction wrt params;
    SE_FLOOR = 0.02, the same pinned weight floor).
  * H3 center = sqrt(2*9) = 4.24264... (the COMMITTED swept center, per "NARROW band at a swept
    center"; the spec's "4.24" is read as its 2-dp display).

Modes: --sandbox (synthetic recovery + conventions; NO real fit), --verdict-test (consequence
branches + md render), --stage1 (the real analysis; STOP-and-report), --reread (re-decide +
re-render from the committed inputs; asserts the measurement table is byte-identical).
"""
from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import argparse

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # ANALYSIS gate: every mode is CPU-only
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np                                     # noqa: E402
from scipy.optimize import least_squares               # noqa: E402
import relay_gate0 as g0                               # noqa: E402 (helpers: _check/_mstats/_fmt/_sha12/_dump_json)
import relay_gate1 as g1                               # noqa: E402 (_env_full)

RESDIR = g0.RESDIR

# ---- pinned inputs (NO-MEASUREMENT on mismatch) --------------------------------------- #
SWEEP_PATH = os.path.join(RESDIR, "gate0_bandsweep.json")
SWEEP_SHA256 = "881c44bee77620fc45218e1c4b34975b7cd8f60d4cd6925b8583e33c3eb6d10e"
GATE3_PATH = os.path.join(RESDIR, "gate3_mechanism.json")
GATE3_RECS_SHA16 = "44475823f7f72d4c"

# ---- pinned analysis constants --------------------------------------------------------- #
SE_FLOOR = 0.02              # weight floor: weight_b = 1/max(SE_b, SE_FLOOR)^2
RESID_TIGHT = 0.03           # acceptance: within max(2*SE_b, 0.03) on >= 5/6
RESID_LOOSE = 0.06           # acceptance: within max(2*SE_b, 0.06) on 6/6
CLIFF_FACTOR = 1.5           # acceptance: wRMSE_CLIFF >= 1.5 * wRMSE_best_smooth
CLIFF_FLOOR = 0.05           # M-CLIFF: r2 above the cut
LAMBDA_SL = 0.1              # M2 pinned (SL amplitude-mode one-pole; corner 2*lambda = 0.2)
P_LO, P_HI = 0.5, 4.0        # M1 prior slope range
PRED_PASS_ABS = 0.05         # stage-2: |mean_obs - pred| <= max(2*sigma_pred, 0.05)
MIN_TIGHT = 5                # of 6 non-overlap band-points

HOLDOUT = {                  # P1 bands pinned from (c, rho); edges lo=c/sqrt(rho), hi=c*sqrt(rho)
    "H1": {"c": 3.30, "rho": 4.5,
           "kind": ("non-overlap INTERPOLATION: in the fit domain (c bracketed by fitted centers "
                    "2.85 and 3.93, rho=4.5); the band itself was never swept = out-of-SAMPLE")},
    "H2": {"c": 12.0, "rho": 4.5,
           "kind": ("overlap-stratum test: EXTRAPOLATION beyond the fitted c-range [0.42,4.24] and "
                    "outside the fitted stratum (message overlaps the injection band)")},
    "H3": {"c": math.sqrt(18.0), "rho": 1.5,
           "kind": ("NARROW band at a swept center (width-dependence discriminant; the fit only saw "
                    "rho=4.5, and M1/M2 have NO width dependence -- this prediction equals the model "
                    "value at c=4.2426 regardless of rho; width-INdependence is the claim under test)")},
}

FRAMING = ("ANALYSIS gate; stage-1 is a RETRODICTION on committed data (no new integrations). "
           "Fit r2 vs band center c = sqrt(lo*hi) on NON-OVERLAP band means only; OVERLAP stratum "
           "reported, not gating. Falsification weight rests on the LOCKED P1 predictions "
           "(stage-2, separate go) and on Gate-L.")

HONESTY = ("Honesty clause (mandatory): stage-1 is a RETRODICTION -- the sweep means and Gate-3 "
           "points are committed, public, and known to the spec author before the spec was drafted. "
           "Pre-registration binds the functional forms, parameter counts, weights, strata, dedupe "
           "rule, and acceptance thresholds -- NOT data blindness. Falsification weight rests on "
           "the locked P1 predictions (stage-2) and on Gate-L. No fit was run before ratification; "
           "all numbers of record come from the ratified pipeline.")


def _mstats(v):
    return g0._mstats(v)


def _fmt(x, spec="+.4f"):
    return g0._fmt(x, spec)


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
#  Input loading + verification (instruments; NO-MEASUREMENT on failure)
# ===================================================================================== #
def load_inputs(sweep_path=SWEEP_PATH, gate3_path=GATE3_PATH,
                sweep_sha=SWEEP_SHA256, gate3_recs_sha=GATE3_RECS_SHA16):
    """Load + re-verify both committed inputs. Returns (sweep, gate3, checks) -- checks['ok'] False
    => NO-MEASUREMENT (reason in checks)."""
    checks = {"ok": True, "reasons": []}

    def fail(msg):
        checks["ok"] = False
        checks["reasons"].append(msg)

    sha = _sha256_file(sweep_path)
    checks["sweep_sha256"] = sha
    if sha != sweep_sha:
        fail(f"gate0_bandsweep.json sha256 {sha} != pinned {sweep_sha}")
    sweep = json.load(open(sweep_path))
    g3 = json.load(open(gate3_path))
    rsha = hashlib.sha256(json.dumps(g3["recs"], sort_keys=True).encode()).hexdigest()[:16]
    checks["gate3_recs_sha16"] = rsha
    checks["gate3_file_sha256"] = _sha256_file(gate3_path)
    if rsha != gate3_recs_sha:
        fail(f"gate3_mechanism.json recs sha16 {rsha} != pinned {gate3_recs_sha}")
    # digit re-verification: center convention + rho + K + seeds
    for i, r in enumerate(sweep.get("rows", [])):
        lo, hi = r["band"]
        if abs(math.sqrt(lo * hi) - r["center"]) > 1e-9:
            fail(f"sweep row {i}: stored center {r['center']} != sqrt(lo*hi) {math.sqrt(lo*hi)}")
        if abs(hi / lo - 4.5) > 1e-9:
            fail(f"sweep row {i}: rho {hi/lo} != 4.5")
    if sweep.get("K") != 0.24:
        fail(f"sweep K {sweep.get('K')} != 0.24")
    if sweep.get("seeds") != [0, 1, 2]:
        fail(f"sweep seeds {sweep.get('seeds')} != [0,1,2]")
    ft = sweep.get("geometry", {}).get("fast_tertile")
    if not ft:
        fail("sweep geometry missing fast_tertile (needed for SUPRA stratum assignment)")
    # gate3 'bands' lives OUTSIDE the pinned recs sha -> cross-verify against band_hz INSIDE the
    # pinned recs (same json parse, exact equality is safe)
    for b, edges in g3.get("bands", {}).items():
        rec0 = (g3.get("recs") or {}).get(f"{b}|0", {})
        if [float(x) for x in rec0.get("band_hz", [])] != [float(edges[0]), float(edges[1])]:
            fail(f"gate3 bands[{b}] {edges} != recs band_hz {rec0.get('band_hz')} "
                 "(cross-check vs the sha-pinned recs failed)")
    return sweep, g3, checks


def build_band_points(sweep, g3):
    """Measurements -> deduped band-points. Returns (points, dedupe_report). Each point:
    {key, band, center, rho, overlap, source, seeds, r2 (per seed), n, mean, sd, se, weight}."""
    ft = sweep["geometry"]["fast_tertile"]
    g3_seeds = sorted(int(k.split("|")[1]) for k in g3["recs"] if k.startswith("SUB|"))
    meas = []
    for i, r in enumerate(sweep["rows"]):
        meas.append({"key": f"sweep[{i}]", "band": [float(r["band"][0]), float(r["band"][1])],
                     "center": float(r["center"]), "overlap": bool(r["overlaps_fast_tertile"]),
                     "source": "gate0_bandsweep", "seeds": list(sweep["seeds"]),
                     "r2": {int(s): float(v) for s, v in zip(sweep["seeds"], r["r2_per_seed"])}})
    for b, (lo, hi) in g3["bands"].items():
        c = math.sqrt(lo * hi)
        overlap = (hi > ft[0]) and (lo < ft[1])          # committed fast-tertile geometry
        meas.append({"key": f"gate3[{b}]", "band": [float(lo), float(hi)], "center": float(c),
                     "overlap": bool(overlap), "source": "gate3_mechanism", "seeds": list(g3_seeds),
                     "r2": {int(i): float(g3["recs"][f"{b}|{i}"]["r2"]["FULL"]) for i in g3_seeds}})
    # DEDUPE (pre-registered): same band -> bit-compare overlapping seeds
    dedupe = []
    points = []
    used = set()
    for a in range(len(meas)):
        if a in used:
            continue
        ma = meas[a]
        partner = None
        for b in range(a + 1, len(meas)):
            if b in used:
                continue
            mb = meas[b]
            if ma["band"] == mb["band"]:
                partner = b
                break
        if partner is None:
            points.append(dict(ma))
            continue
        mb = meas[partner]
        shared = sorted(set(ma["r2"]) & set(mb["r2"]))
        bits = {s: (ma["r2"][s] == mb["r2"][s]) for s in shared}
        identical = bool(shared) and all(bits.values())
        rep = {"band": ma["band"], "a": ma["key"], "b": mb["key"], "shared_seeds": shared,
               "bit_identical": bits, "merged": identical,
               "unresolvable": not shared}                       # same band, NO shared seeds -> NM
        if identical:
            merged = dict(mb if len(mb["r2"]) >= len(ma["r2"]) else ma)
            merged["r2"] = {**ma["r2"], **mb["r2"]}                 # union of seeds (values bit-equal)
            merged["seeds"] = sorted(merged["r2"])
            merged["key"] = f"{ma['key']}+{mb['key']}"
            merged["source"] = "merged(bit-identical)"
            points.append(merged)
            used.add(partner)
        else:
            rep["discrepancy_verbatim"] = {
                ma["key"]: {s: repr(ma["r2"][s]) for s in shared},
                mb["key"]: {s: repr(mb["r2"][s]) for s in shared}}
            points.append(dict(ma))
            points.append(dict(mb))
            used.add(partner)
        dedupe.append(rep)
    for p in points:
        vals = [p["r2"][s] for s in sorted(p["r2"])]
        st = _mstats(vals)
        p["n"], p["mean"], p["sd"], p["se"] = st["n"], st["mean"], st["sd"], st["se"]
        p["weight"] = 1.0 / max(p["se"], SE_FLOOR) ** 2
    points.sort(key=lambda p: (p["center"], p["key"]))
    return points, dedupe


# ===================================================================================== #
#  Models + WLS fits (deterministic multi-start grid -> least_squares refine)
# ===================================================================================== #
def m1_f(c, logA, p):
    snr = np.exp(logA) * np.asarray(c, float) ** (-p)
    return snr / (1.0 + snr)


def m2_f(c, logA):
    snr = np.exp(logA) / (4.0 * LAMBDA_SL ** 2 + np.asarray(c, float) ** 2)
    return snr / (1.0 + snr)


def _wls_fit(fun, theta0_grid, bounds, c, y, w):
    """Deterministic WLS: evaluate the start grid, refine the best 3 with least_squares.
    Returns dict(theta, cov, wrmse, chi2, chi2_red, converged, resid)."""
    sw = np.sqrt(w)

    def res(theta):
        return (fun(c, *theta) - y) * sw

    scores = [(float(np.sum(res(t) ** 2)), tuple(t)) for t in theta0_grid]
    scores.sort(key=lambda x: (x[0], x[1]))
    best = None
    for _, t0 in scores[:3]:
        r = least_squares(res, x0=np.asarray(t0, float), bounds=bounds, method="trf",
                          xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    chi2 = float(2.0 * best.cost)
    n, k = len(y), len(best.x)
    dof = max(1, n - k)
    chi2_red = chi2 / dof
    J = best.jac                                          # already weighted (res includes sqrt(w))
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ)                          # UNSCALED (Jason-ratified; weights carry
    except np.linalg.LinAlgError:                         #  the variance model, floor included)
        cov = np.full((k, k), np.nan)
    cov_ok = bool(np.all(np.isfinite(cov)))
    lob, hib = bounds
    at_bound = [bool(abs(best.x[i] - lob[i]) < 1e-9 or abs(best.x[i] - hib[i]) < 1e-9)
                for i in range(k)]
    fit = fun(c, *best.x)
    wrmse = float(np.sqrt(np.sum(w * (y - fit) ** 2) / np.sum(w)))
    return {"theta": [float(x) for x in best.x], "cov": cov.tolist(),
            "theta_sd": [float(x) for x in (np.sqrt(np.diag(cov)) if cov_ok else [float("nan")] * k)],
            "at_bound": at_bound, "wrmse": wrmse,
            "chi2": chi2, "chi2_red": float(chi2_red),
            # converged REQUIRES a usable covariance: a singular/NaN cov must route to the
            # pre-registered NO-MEASUREMENT (fit non-convergence), never NaN into LOCKED P1
            "converged": bool(best.success) and cov_ok,
            "fit": [float(x) for x in fit]}


def fit_m1(c, y, w):
    grid = [(la, p) for la in np.log(np.logspace(-3, 6, 46)) for p in np.linspace(P_LO, P_HI, 36)]
    out = _wls_fit(m1_f, grid, ([-14.0, P_LO], [21.0, P_HI]), c, y, w)
    out["model"], out["n_params"] = "M1: SNR=A*c^-p", 2
    A, sdA, sdp = float(np.exp(out["theta"][0])), out["theta_sd"][0], out["theta_sd"][1]
    out["params"] = {"A": A, "p": float(out["theta"][1])}
    out["params_ci_2sd"] = {"A": [A * math.exp(-2 * sdA), A * math.exp(2 * sdA)],   # exp-transformed
                            "p": [out["theta"][1] - 2 * sdp, out["theta"][1] + 2 * sdp]}
    return out


def fit_m2(c, y, w):
    grid = [(la,) for la in np.log(np.logspace(-4, 8, 100))]
    out = _wls_fit(m2_f, grid, ([-16.0], [25.0]), c, y, w)
    out["model"], out["n_params"] = "M2: SNR=A/(4*lambda^2+c^2), lambda=0.1 pinned", 1
    A, sdA = float(np.exp(out["theta"][0])), out["theta_sd"][0]
    out["params"] = {"A": A, "lambda_pinned": LAMBDA_SL}
    out["params_ci_2sd"] = {"A": [A * math.exp(-2 * sdA), A * math.exp(2 * sdA)]}
    return out


def fit_cliff(c, y, w):
    """M-CLIFF: r2 = r_top for c < nu_x else CLIFF_FLOOR. nu_x scanned over segment cuts
    (loss is piecewise-constant between sorted centers); r_top closed-form weighted mean."""
    cs = np.asarray(c, float)
    order = np.argsort(cs)
    cuts = [float(cs[order[0]]) * 0.5]
    cuts += [float(0.5 * (cs[order[i]] + cs[order[i + 1]])) for i in range(len(cs) - 1)]
    cuts += [float(cs[order[-1]]) * 1.05]
    best = None
    for nu in cuts:
        below = cs < nu
        if below.any():
            r_top = float(np.sum(w[below] * y[below]) / np.sum(w[below]))
        else:
            r_top = None
        fit = np.where(below, (r_top if r_top is not None else CLIFF_FLOOR), CLIFF_FLOOR)
        wrmse = float(np.sqrt(np.sum(w * (y - fit) ** 2) / np.sum(w)))
        cand = {"nu_x": nu, "r_top": r_top, "wrmse": wrmse, "fit": [float(x) for x in fit]}
        if best is None or wrmse < best["wrmse"] - 1e-15:
            best = cand
    best["model"], best["n_params"] = f"M-CLIFF: r2=r_top for c<nu_x else {CLIFF_FLOOR}", 2
    best["converged"] = True
    # degeneracy flag (Jason's catch): if the best fit puts ALL points below the cut (no collapse in
    # range), the model reduces to a constant and nu_x is UNIDENTIFIED above the last center -- the
    # printed nu_x is an arbitrary point in the degenerate region and must carry a flag.
    best["nu_x_degenerate"] = bool(best["nu_x"] > float(np.max(cs)))
    best["nu_x_note"] = (f"nu_x UNIDENTIFIED above the last center {float(np.max(cs)):.4f} (no collapse "
                         "in range; the fit degenerates to a constant r_top and the printed nu_x is an "
                         "arbitrary point in the degenerate region)" if best["nu_x_degenerate"] else
                         "nu_x identified (collapse within the fitted range)")
    return best


def predict(fun, theta, cov, c):
    """Prediction + sigma_pred = sqrt(g^T Cov g + SE_FLOOR^2), g = numeric central gradient."""
    theta = np.asarray(theta, float)
    val = float(fun(np.array([c]), *theta)[0])
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        h = 1e-5 * max(1.0, abs(theta[i]))
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        g[i] = (float(fun(np.array([c]), *tp)[0]) - float(fun(np.array([c]), *tm)[0])) / (2 * h)
    var = float(g @ np.asarray(cov, float) @ g) + SE_FLOOR ** 2
    return val, float(math.sqrt(max(var, 0.0)))


def _dedupe_nm(dedupe, checks):
    """Spec NM trigger: 'dedupe bit-compare cannot be resolved' (same band, no shared seeds)."""
    for d in dedupe:
        if d.get("unresolvable"):
            checks["ok"] = False
            checks["reasons"].append(f"dedupe bit-compare cannot be resolved for band {d['band']} "
                                     f"({d['a']} vs {d['b']}: no shared seeds)")
    return checks


# ===================================================================================== #
#  Decision (pre-registered consequence map)
# ===================================================================================== #
def decide(points, checks):
    out = {"framing": FRAMING, "honesty": HONESTY, "input_checks": checks,
           "operationalization": {
               "se_floor": SE_FLOOR, "resid_tight": RESID_TIGHT, "resid_loose": RESID_LOOSE,
               "cliff_factor": CLIFF_FACTOR, "cliff_floor": CLIFF_FLOOR, "p_range": [P_LO, P_HI],
               "lambda_pinned": LAMBDA_SL, "min_tight": MIN_TIGHT,
               "wrmse": ("sqrt(sum_b w_b*resid_b^2 / sum_b w_b), w_b = 1/max(SE_b,0.02)^2 -- NOTE: "
                         "this definition also sets best-smooth selection and the cliff ratio (the branch)"),
               "cov": ("(J^T W J)^-1 UNSCALED, params (logA, p) -- Jason-ratified as independently "
                       "reproduced; param CIs reported as +/- 2sd (exp-transformed for A)"),
               "sigma_pred": "sqrt(g^T Cov g + SE_FLOOR^2) -- fit covariance + pinned floor in quadrature ONLY, no model-error term",
               "h3_center": ("sqrt(2*9) = 4.2426... = the COMMITTED swept center of band [2,9] (spec display "
                             "'4.24'); exact band [3.4641, 5.1962]"),
               "at_bound_policy": "a fitted param pinned at a prior bound is flagged; covariance is then conditional",
               "reread_lock": ("--reread asserts EVERY numeric substructure (fits/residuals/ratio/LOCKED P1) "
                               "byte-identical to the stage-1 record; loud failure on drift, never overwrite"),
               "stage2_pass": (f"|mean_obs - pred| <= max(2*sigma_pred, {PRED_PASS_ABS}) for ALL THREE "
                               "(conjunction), EVALUATED AT STAGE-2 from the formula + byte-locked "
                               "(pred, sigma) -- windows are NOT stored numbers (Jason's ruling: a "
                               "stored/display-rounded window is never rule-faithful); any other "
                               "pattern reported as-is; no threshold moves after data")}}
    if not checks["ok"]:
        out["verdict"] = ("NO-MEASUREMENT (input re-verification failed: "
                          + "; ".join(checks["reasons"]) + " -- STOP, fix inputs, re-run)")
        return out
    non = [p for p in points if not p["overlap"]]
    ove = [p for p in points if p["overlap"]]
    out["n_non_overlap"], out["n_overlap"] = len(non), len(ove)
    c = np.array([p["center"] for p in non])
    y = np.array([p["mean"] for p in non])
    w = np.array([p["weight"] for p in non])
    f1, f2, fc = fit_m1(c, y, w), fit_m2(c, y, w), fit_cliff(c, y, w)
    out["fits"] = {"M1": f1, "M2": f2, "M_CLIFF": fc}
    if not (f1["converged"] and f2["converged"]):
        out["verdict"] = "NO-MEASUREMENT (fit non-convergence: " \
                         f"M1={f1['converged']} M2={f2['converged']} -- STOP)"
        return out
    best_name = "M1" if f1["wrmse"] <= f2["wrmse"] else "M2"
    best = out["fits"][best_name]
    out["best_smooth"] = best_name
    # residual acceptance on the best smooth model (non-overlap only)
    resid_rows = []
    tight = loose = 0
    for p, fv in zip(non, best["fit"]):
        r = p["mean"] - fv
        tt = max(2 * p["se"], RESID_TIGHT)
        tl = max(2 * p["se"], RESID_LOOSE)
        ok_t, ok_l = abs(r) <= tt, abs(r) <= tl
        tight += ok_t
        loose += ok_l
        resid_rows.append({"center": p["center"], "n": p["n"], "mean": p["mean"], "se": p["se"],
                           "fit": fv, "resid": float(r), "tol_tight": float(tt),
                           "tol_loose": float(tl), "ok_tight": bool(ok_t), "ok_loose": bool(ok_l)})
    out["residuals"] = resid_rows
    out["tight_count"], out["loose_count"] = int(tight), int(loose)
    ratio = fc["wrmse"] / best["wrmse"] if best["wrmse"] > 0 else float("inf")
    out["cliff_ratio"] = float(ratio)
    # overlap stratum: extrapolation report only (never gates)
    fun = m1_f if best_name == "M1" else m2_f
    out["overlap_report"] = [
        {"key": p["key"], "center": p["center"], "n": p["n"], "mean": p["mean"], "se": p["se"],
         "extrapolation": float(fun(np.array([p["center"]]), *best["theta"])[0]),
         "resid": float(p["mean"] - fun(np.array([p["center"]]), *best["theta"])[0])} for p in ove]
    # LOCKED P1 (from the best smooth fit; written BEFORE any stage-2 go)
    p1 = {}
    for h, spec in HOLDOUT.items():
        val, sig = predict(fun, best["theta"], best["cov"], spec["c"])
        p1[h] = {"c": spec["c"], "rho": spec["rho"], "band": band_edges(spec["c"], spec["rho"]),
                 "kind": spec["kind"], "pred_r2": val, "sigma_pred": sig}
    out["locked_P1"] = {
        "model": best_name, "predictions": p1,
        # Jason's ruling: pass windows are NOT stored numbers (a stored/display-rounded window can
        # never be rule-faithful); stage-2 EVALUATES the ratified formula against the byte-locked
        # full-precision (pred_r2, sigma_pred) recorded here.
        "pass_rule": (f"PREDICTIVE PASS iff |mean_obs - pred_r2| <= max(2*sigma_pred, {PRED_PASS_ABS}) "
                      "for ALL THREE bands (conjunction), EVALUATED AT STAGE-2 from this formula and "
                      "the byte-locked full-precision (pred_r2, sigma_pred) -- windows are NOT stored. "
                      "Any other pattern: report as-is. No threshold moves after data."),
        "note": "locked at stage-1, before any stage-2 go; no threshold moves after data"}
    # consequence map (EXPLAINED / CLIFF-PREFERRED / SHAPE-UNEXPLAINED; disjoint by construction)
    explained = (tight >= MIN_TIGHT and loose == len(non) and ratio >= CLIFF_FACTOR)
    if explained:
        out["verdict"] = (
            f"EXPLAINED -- broadband trackability is quantitatively attributed to pointwise "
            f"square-law demodulation + linear-propagation attenuation: best smooth model "
            f"{best_name} ({best['model']}) fits the {len(non)} non-overlap band-points with "
            f"residuals within max(2*SE,{RESID_TIGHT}) on {tight}/{len(non)} and within "
            f"max(2*SE,{RESID_LOOSE}) on {loose}/{len(non)}; wRMSE_CLIFF/wRMSE_{best_name} = "
            f"{ratio:.2f} >= {CLIFF_FACTOR}. The Phase-3 line-67 erratum becomes DRAFTABLE "
            f"(drafting/filing in decade_drive is Jason's separate call and waits for Gate-L "
            f"consistency). LOCKED P1 predictions emitted (falsifiability payload; stage-2 is a "
            f"separate go).")
    elif ratio < 1.0:
        out["verdict"] = (
            f"CLIFF-PREFERRED -- the rate-limit picture partially survives: wRMSE_CLIFF "
            f"({fc['wrmse']:.4f}) < wRMSE_{best_name} ({best['wrmse']:.4f}). Report; escalate to "
            f"targeted sweep design.")
    else:
        out["verdict"] = (
            f"SHAPE-UNEXPLAINED -- acceptance fails (tight {tight}/{len(non)} need >={MIN_TIGHT}; "
            f"loose {loose}/{len(non)} need {len(non)}; cliff ratio {ratio:.2f} need >="
            f"{CLIFF_FACTOR}): residual structure reported verbatim; no erratum; Gate-L proceeds "
            f"regardless.")
    return out


# ===================================================================================== #
#  Markdown record
# ===================================================================================== #
def _write_md(path, v, points, dedupe, wall, hashes, note=""):
    ck = v["input_checks"]
    lines = [
        "# Relay Gate-B -- broadband trackability mechanism (analysis; retrodiction + locked predictions)",
        "",
        f"Spec: relay_gateB_broadband_spec.md (sha256 {hashes['spec']}). Harness: "
        f"experiments/relay_gateB.py (sha256 {hashes['code']}).",
        f"Inputs: gate0_bandsweep.json sha256 {ck.get('sweep_sha256','?')} (pinned match: "
        f"{ck.get('sweep_sha256')==SWEEP_SHA256}); gate3_mechanism.json recs sha16 "
        f"{ck.get('gate3_recs_sha16','?')} (pinned match: {ck.get('gate3_recs_sha16')==GATE3_RECS_SHA16}).",
        f"Wall-clock {wall:.1f} s (CPU).",
    ] + ([note] if note else []) + [
        "", f"Framing: {FRAMING}", "", f"{HONESTY}", "",
        "## Verdict", "", f"**{v['verdict']}**", "",
    ]
    if str(v.get("verdict", "")).startswith("EXPLAINED"):
        lines += [
            "## Interpretive note (ships with EXPLAINED; Jason-ratified text)", "",
            "NOTE (interpretive, ships with EXPLAINED): the winning model is the GENERIC smooth power "
            "law (p = 1.04, 2sd [0.34, 1.74]); the mechanism-derived one-pole form (asymptotic slope 2) "
            "fits 2.3x worse and p = 2 lies outside the 2sd interval. What this stage establishes is: "
            "no cliff and no rate limit -- the best cliff the data allows is a constant (nu_x "
            "unidentified above the last point), losing 5.27x -- NOT any specific attenuation law. A "
            "single pole is also too crude a derivation here: the slow tertile spans corner frequencies "
            "across [1.00, 3.13] (166 oscillators) and the fitted window c = 0.42-4.24 brackets them, "
            "where a distributed-corner system shows intermediate effective slope between 0 and 2. The "
            "attenuation law remains OPEN; stage-2 tests the M1 power-law interpolation, not a "
            "mechanism form. The overlap stratum sits systematically ABOVE the extrapolated fit (all 5 "
            "residuals positive, +0.024 to +0.052, broadly increasing with c) -- consistent with an "
            "additional direct transfer path when the message band lies inside the injection band, "
            "still power-borne per Gate-3 (SUPRA: SQ-carried). Recorded pre-data: H2 may land high in "
            "its window. Relevant to Gate-L.", ""]
    if "fits" in v:
        f1, f2, fc = v["fits"]["M1"], v["fits"]["M2"], v["fits"]["M_CLIFF"]
        best = v["best_smooth"]
        lines += [
            "## Band-points (dedupe outcome + strata)", "",
            f"- dedupe: {sum(1 for d in dedupe if d['merged'])} merged (bit-identical), "
            f"{sum(1 for d in dedupe if not d['merged'])} discrepant (reported verbatim below).",
        ] + (["- NO-PROTOCOL-DRIFT (free result of the dedupe rule): two DIFFERENT committed harnesses "
              "(the gate0-era band sweep and the gate3 mechanism battery) produced BIT-IDENTICAL cells "
              "on every shared (band, seed) -- the measurement protocol did not drift between them."]
             if dedupe and all(d["merged"] for d in dedupe) else [])
        for d in dedupe:
            if d["merged"]:
                lines.append(f"  - band {d['band']}: {d['a']} == {d['b']} bit-identical on seeds "
                             f"{d['shared_seeds']} -> MERGED (union of seeds).")
            else:
                lines.append(f"  - band {d['band']}: {d['a']} vs {d['b']} NOT identical -- both enter; "
                             f"verbatim: {d['discrepancy_verbatim']}")
        lines += ["", "| center | band | stratum | n | mean r2 | SE | weight | source |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in points:
            lines.append(f"| {p['center']:.4f} | [{p['band'][0]:.3f},{p['band'][1]:.3f}] | "
                         f"{'OVERLAP' if p['overlap'] else 'non-overlap'} | {p['n']} | "
                         f"{p['mean']:.4f} | {p['se']:.4f} | {p['weight']:.1f} | {p['source']} |")
        lines += [
            "", "## Fits (WLS on non-overlap band means; weights 1/max(SE,0.02)^2; param CIs +/- 2sd, "
            "A exp-transformed)", "",
            f"- M1 (2 par): A = {f1['params']['A']:.4g} [{f1['params_ci_2sd']['A'][0]:.4g}, "
            f"{f1['params_ci_2sd']['A'][1]:.4g}], p = {f1['params']['p']:.4f} "
            f"[{f1['params_ci_2sd']['p'][0]:.4f}, {f1['params_ci_2sd']['p'][1]:.4f}]; "
            f"wRMSE = {f1['wrmse']:.4f}; chi2_red = {f1['chi2_red']:.2f}"
            + (" **[WARNING: param at prior bound; covariance conditional]**" if any(f1["at_bound"]) else ""),
            f"- M2 (1 par, lambda=0.1 pinned): A = {f2['params']['A']:.4g} "
            f"[{f2['params_ci_2sd']['A'][0]:.4g}, {f2['params_ci_2sd']['A'][1]:.4g}]; "
            f"wRMSE = {f2['wrmse']:.4f}; chi2_red = {f2['chi2_red']:.2f}"
            + (" **[WARNING: param at prior bound; covariance conditional]**" if any(f2["at_bound"]) else ""),
            f"- M-CLIFF (2 par): r_top = {_fmt(fc['r_top'],'.4f')}, nu_x = {fc['nu_x']:.4f}"
            + (" **[DEGENERATE]**" if fc.get("nu_x_degenerate") else "") + f"; "
            f"wRMSE = {fc['wrmse']:.4f} (step model; no covariance -> no CI). {fc.get('nu_x_note','')}",
            f"- best smooth = **{best}**; wRMSE_CLIFF / wRMSE_{best} = **{v['cliff_ratio']:.2f}** "
            f"(EXPLAINED needs >= {CLIFF_FACTOR}; CLIFF-PREFERRED needs < 1).",
            "", f"## Residuals vs best smooth ({best}; acceptance: tight max(2*SE,{RESID_TIGHT}) on "
            f">={MIN_TIGHT}/{len(v['residuals'])}, loose max(2*SE,{RESID_LOOSE}) on "
            f"{len(v['residuals'])}/{len(v['residuals'])})", "",
            "| center | n | mean | fit | resid | tol_tight | tol_loose | tight | loose |",
            "|---|---|---|---|---|---|---|---|---|"]
        for r in v["residuals"]:
            lines.append(f"| {r['center']:.4f} | {r['n']} | {r['mean']:.4f} | {r['fit']:.4f} | "
                         f"{r['resid']:+.4f} | {r['tol_tight']:.4f} | {r['tol_loose']:.4f} | "
                         f"{'ok' if r['ok_tight'] else 'MISS'} | {'ok' if r['ok_loose'] else 'MISS'} |")
        nn = len(v["residuals"])
        lines += [f"", f"- tight {v['tight_count']}/{nn}, loose {v['loose_count']}/{nn}."
                  + ("" if nn == 6 else " **[NOTE: acceptance was pre-registered for 6 band-points; "
                     f"this run has {nn} -- surfaced, not silently normalized]**"),
                  "", "## OVERLAP stratum (reported, NOT gating; model extrapolated beyond its fit domain)", ""]
        for r in v["overlap_report"]:
            lines.append(f"- c = {r['center']:.4f} ({r['key']}, n={r['n']}): mean {r['mean']:.4f}, "
                         f"extrapolation {r['extrapolation']:.4f}, resid {r['resid']:+.4f}")
        lp = v["locked_P1"]
        lines += ["", f"## LOCKED P1 predictions (model {lp['model']}; {lp['note']})", "",
                  "Intervals: sigma_pred = sqrt(g^T Cov g + 0.02^2) -- fit covariance plus the pinned "
                  "SE floor ONLY; NO model-misspecification term. H2 is an extrapolation and H3 a "
                  "width-transfer claim: their intervals quantify fit uncertainty, not model error -- "
                  "that is exactly what stage-2 tests.", ""]
        for h in ("H1", "H2", "H3"):
            q = lp["predictions"][h]
            lines.append(f"- **{h}**: c = {q['c']:.4f}, rho = {q['rho']}, band = "
                         f"[{q['band'][0]:.4f}, {q['band'][1]:.4f}] ({q['kind']}) -> pred r2 = "
                         f"**{q['pred_r2']:.4f} +/- {q['sigma_pred']:.4f}** (displays; the json "
                         f"records full precision, byte-locked)")
        lines += ["", f"- Stage-2 scoring (pinned now): {lp['pass_rule']}",
                  "", "## Scope", "",
                  "Retrodiction on committed data; the fit only constrains the NON-OVERLAP stratum "
                  "(rho=4.5, c in [0.42, 4.24]). H1 is in-domain interpolation on a never-swept band "
                  "(out-of-sample); H2 extrapolates beyond the fitted c-range and stratum; H3 transfers "
                  "the fit to a narrower band (the models claim width-independence). The OVERLAP "
                  "extrapolations and P1 are what the stage-2 probe (separate go) and Gate-L can "
                  "falsify. No erratum files from this gate alone."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ===================================================================================== #
#  STAGE-1 (the real analysis) + reread
# ===================================================================================== #
def _load_and_build():
    """Shared stage1/reread input path: NM short-circuits BEFORE build (a structurally broken input
    must produce the pre-registered NO-MEASUREMENT record, not a traceback)."""
    try:
        sweep, g3, checks = load_inputs()
    except Exception as e:
        return [], [], {"ok": False, "reasons": [f"input load/parse failure: {e!r}"]}
    if not checks["ok"]:
        return [], [], checks
    points, dedupe = build_band_points(sweep, g3)
    _dedupe_nm(dedupe, checks)
    return points, dedupe, checks


def stage1(log):
    import time
    t0 = time.perf_counter()
    log("=== RELAY GATE-B :: STAGE-1 (CPU analysis on committed data; no integrations) ===")
    log(f"    framing: {FRAMING}")
    points, dedupe, checks = _load_and_build()
    v = decide(points, checks)
    wall = time.perf_counter() - t0
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..",
                                             "relay_gateB_broadband_spec.md"))}
    payload = {"gate": "relay-gateB", "stage": "1-analysis", "framing": FRAMING, "honesty": HONESTY,
               "hashes": hashes, "env": g1._env_full(), "wall_clock_s": wall,
               "points": [{k: p[k] for k in ("key", "band", "center", "overlap", "source", "n",
                                             "mean", "sd", "se", "weight")} |
                          {"r2": {str(s): p["r2"][s] for s in sorted(p["r2"])}} for p in points],
               "dedupe": dedupe, "verdict": v}
    outp = os.path.join(RESDIR, "gateB_broadband.json")
    g0._dump_json(outp, payload)
    _write_md(os.path.join(RESDIR, "gateB_broadband.md"), v, points, dedupe, wall, hashes)
    log(f"  dedupe: {[(d['band'], 'MERGED' if d['merged'] else 'DISCREPANT') for d in dedupe]}")
    if "fits" in v:
        log(f"  M1 wRMSE={v['fits']['M1']['wrmse']:.4f} (A={v['fits']['M1']['params']['A']:.3g}, "
            f"p={v['fits']['M1']['params']['p']:.3f}) | M2 wRMSE={v['fits']['M2']['wrmse']:.4f} "
            f"(A={v['fits']['M2']['params']['A']:.3g}) | CLIFF wRMSE={v['fits']['M_CLIFF']['wrmse']:.4f} "
            f"-> best={v['best_smooth']} ratio={v['cliff_ratio']:.2f}")
        log(f"  residuals tight {v['tight_count']}/{len(v['residuals'])} "
            f"loose {v['loose_count']}/{len(v['residuals'])}")
        for h, q in v["locked_P1"]["predictions"].items():
            log(f"  LOCKED {h}: c={q['c']:.4f} rho={q['rho']} -> pred r2 {q['pred_r2']:.4f} "
                f"+/- {q['sigma_pred']:.4f}")
    log(f"\n=== VERDICT ===\n  {v['verdict']}")
    log(f"  [written -> {os.path.relpath(outp)} + gateB_broadband.md]  (NOT committed)")
    return v


def reread(log):
    src = os.path.join(RESDIR, "gateB_broadband.json")
    assert os.path.exists(src), f"missing stage-1 record {src} -- run --stage1 first"
    nm = json.load(open(src))
    log("=== RELAY GATE-B :: REREAD (re-decide from committed inputs; byte-identical assert) ===")
    points, dedupe, checks = _load_and_build()
    fresh = [{k: p[k] for k in ("key", "band", "center", "overlap", "source", "n",
                                "mean", "sd", "se", "weight")} |
             {"r2": {str(s): p["r2"][s] for s in sorted(p["r2"])}} for p in points]
    assert json.dumps(fresh, sort_keys=True) == json.dumps(nm["points"], sort_keys=True), \
        "reread measurement table drifted from the stage-1 record"
    log("  [integrity] band-point table recomputed from committed inputs: byte-identical: OK")
    v = decide(points, checks)
    # LOCKED-NUMBERS CONTRACT (the P1 lock): every numeric substructure of the verdict must be
    # BYTE-IDENTICAL to the stage-1 record -- a LOUD AssertionError on any drift (code, scipy, or
    # BLAS change), never a silent overwrite. Only prose may re-render.
    for k in ("fits", "residuals", "tight_count", "loose_count", "cliff_ratio", "best_smooth",
              "overlap_report", "locked_P1", "n_non_overlap", "n_overlap"):
        if k in nm["verdict"] or k in v:
            assert json.dumps(v.get(k), sort_keys=True) == json.dumps(nm["verdict"].get(k),
                                                                      sort_keys=True), \
                f"reread NUMERIC DRIFT in verdict['{k}'] -- the LOCKED payload would change; STOP"
    log("  [integrity] ALL numeric substructures (fits/residuals/ratio/LOCKED P1) byte-identical: OK")
    hashes = {"code": g0._sha12(os.path.abspath(__file__)),
              "spec": g0._sha12(os.path.join(os.path.dirname(__file__), "..",
                                             "relay_gateB_broadband_spec.md"))}
    run_hashes = nm.get("run_hashes") or nm.get("hashes")     # first-write-wins (gate3 lesson)
    note = (f"Provenance: numbers from the stage-1 analysis (harness sha "
            f"{(run_hashes or {}).get('code','?')}); verdict RE-RENDERED by --reread "
            f"(sha {hashes['code']}), measurement table asserted byte-identical.")
    g0._dump_json(src, {**nm, "verdict": v, "hashes": hashes, "run_hashes": run_hashes,
                        "reread": "verdict re-rendered; measurement table byte-identical; no number changed"})
    _write_md(os.path.join(RESDIR, "gateB_broadband.md"), v, points, dedupe,
              nm.get("wall_clock_s", 0.0), hashes, note=note)
    log(f"  {v['verdict'][:100]}...")
    log(f"  [rewritten -> {os.path.relpath(src)} + gateB_broadband.md]  (NOT committed)")
    return v


# ===================================================================================== #
#  SANDBOX (synthetic only -- NO real fit is run here) + verdict-test
# ===================================================================================== #
SPEC_NON_OVERLAP_2DP = [0.42, 1.50, 2.07, 2.85, 3.93, 4.24]     # the spec's strata centers (display)
_CENTER_CACHE = {}


def _real_centers():
    """Non-overlap centers from the COMMITTED sweep file (geometry, not measurements)."""
    if "c" not in _CENTER_CACHE:
        d = json.load(open(SWEEP_PATH))
        _CENTER_CACHE["c"] = sorted(float(r["center"]) for r in d["rows"]
                                    if not r["overlaps_fast_tertile"])
    return _CENTER_CACHE["c"]


def _synth_points(y_means, ses=None, centers=None, overlap=False):
    centers = _real_centers() if centers is None else centers
    pts = []
    for i, (c, m) in enumerate(zip(centers, y_means)):
        se = 0.01 if ses is None else ses[i]
        pts.append({"key": f"syn[{i}]", "band": band_edges(c, 4.5), "center": float(c),
                    "overlap": overlap, "source": "synthetic", "seeds": [0, 1, 2],
                    "r2": {0: m, 1: m, 2: m}, "n": 3, "mean": float(m), "sd": se * math.sqrt(3),
                    "se": float(se), "weight": 1.0 / max(se, SE_FLOOR) ** 2})
    return pts


def sandbox(log):
    log("=== RELAY GATE-B :: SANDBOX (CPU; synthetic recovery + conventions; NO real fit) ===")
    log(f"    framing: {FRAMING}")
    R = {}
    rng = np.random.default_rng(11)

    # ---- CHECK 0: committed-input verification machinery (instruments; not a fit) -------- #
    log("\n(0) Input verification -- pinned shas, center convention, rho, K, seeds, geometry")
    sweep, g3, checks = load_inputs()
    cs = [r["center"] for r in sweep["rows"] if not r["overlaps_fast_tertile"]]
    conv_ok = [round(x, 2) for x in sorted(cs)] == SPEC_NON_OVERLAP_2DP
    c0 = all([
        g0._check(log, "sha + digit re-verification passes on the committed inputs", checks["ok"],
                  f"sweep sha {checks['sweep_sha256'][:12]}..., gate3 recs {checks['gate3_recs_sha16']}"),
        g0._check(log, "center convention c=sqrt(lo*hi) holds for all 10 stored rows", True,
                  "verified inside load_inputs (<=1e-9)"),
        g0._check(log, "non-overlap centers match the spec's 6 strata centers", conv_ok,
                  f"{[round(x,3) for x in sorted(cs)]}"),
        g0._check(log, "tampered sha -> NO-MEASUREMENT fires", not load_inputs(
            sweep_sha="0" * 64)[2]["ok"], "pinned-sha mismatch detected"),
    ])
    R["check0_inputs"] = {"pass": c0}

    # ---- CHECK 1: dedupe bit-compare on known-identical and known-different cells -------- #
    log("\n(1) Dedupe -- known-identical merges (union of seeds); known-different -> both + verbatim")
    swA = {"geometry": sweep["geometry"], "seeds": [0, 1, 2],
           "rows": [{"band": [0.2, 0.9], "center": math.sqrt(0.18), "r2_per_seed": [0.5, 0.6, 0.7],
                     "overlaps_fast_tertile": False}]}
    g3A = {"bands": {"SUB": [0.2, 0.9]}, "recs": {f"SUB|{i}": {"r2": {"FULL": v}} for i, v in
                                                  enumerate([0.5, 0.6, 0.7, 0.8, 0.9])}}
    ptsA, dedA = build_band_points(swA, g3A)
    g3B = {"bands": {"SUB": [0.2, 0.9]}, "recs": {f"SUB|{i}": {"r2": {"FULL": v}} for i, v in
                                                  enumerate([0.5, 0.6000000001, 0.7])}}
    ptsB, dedB = build_band_points(swA, g3B)
    c1 = all([
        g0._check(log, "identical cells -> ONE merged point, union of seeds (n=5)",
                  len(ptsA) == 1 and ptsA[0]["n"] == 5 and dedA[0]["merged"],
                  f"n={ptsA[0]['n']} seeds={ptsA[0]['seeds']}"),
        g0._check(log, "different cells -> BOTH points enter + verbatim discrepancy",
                  len(ptsB) == 2 and not dedB[0]["merged"] and "discrepancy_verbatim" in dedB[0],
                  f"{len(ptsB)} points; discrepancy seeds {list(dedB[0]['bit_identical'])}"),
    ])
    R["check1_dedupe"] = {"pass": c1}

    # ---- CHECK 2: WLS synthetic recovery (planted M1 {A,p}; planted M2 A) ---------------- #
    log("\n(2) WLS synthetic recovery -- planted M1 (A=2.5, p=1.8) and planted M2 (A=2.0)")
    c = np.array(_real_centers())
    yA = m1_f(c, math.log(2.5), 1.8) + rng.normal(0, 0.005, len(c))
    fA = fit_m1(c, yA, np.full(len(c), 1 / SE_FLOOR ** 2))
    yB = m2_f(c, math.log(2.0)) + rng.normal(0, 0.005, len(c))
    fB = fit_m2(c, yB, np.full(len(c), 1 / SE_FLOOR ** 2))
    c2 = all([
        g0._check(log, "M1 recovers planted p within 0.15", abs(fA["params"]["p"] - 1.8) < 0.15,
                  f"p_hat={fA['params']['p']:.3f}"),
        g0._check(log, "M1 recovers planted A within 25%", abs(fA["params"]["A"] / 2.5 - 1) < 0.25,
                  f"A_hat={fA['params']['A']:.3f}"),
        g0._check(log, "M2 recovers planted A within 25%", abs(fB["params"]["A"] / 2.0 - 1) < 0.25,
                  f"A_hat={fB['params']['A']:.3f}"),
    ])
    R["check2_recovery"] = {"pass": c2, "m1": fA["params"], "m2": fB["params"]}

    # ---- CHECK 3: selection rule fires BOTH ways on synthetic curves --------------------- #
    log("\n(3) Selection -- planted smooth -> EXPLAINED; planted cliff -> CLIFF-PREFERRED; "
        "planted structure -> SHAPE-UNEXPLAINED")
    okc = {"ok": True, "reasons": [], "sweep_sha256": SWEEP_SHA256, "gate3_recs_sha16": GATE3_RECS_SHA16}
    y_sm = m2_f(c, math.log(2.0)) + rng.normal(0, 0.004, len(c))
    v_sm = decide(_synth_points(list(y_sm)), okc)
    y_cl = np.where(c < 3.0, 0.92, CLIFF_FLOOR) + rng.normal(0, 0.004, len(c))
    v_cl = decide(_synth_points(list(y_cl)), okc)
    y_st = m1_f(c, math.log(2.5), 1.8) + np.array([0.09, -0.09, 0.09, -0.09, 0.09, -0.09])
    v_st = decide(_synth_points(list(y_st)), okc)
    fcl = v_cl["fits"]["M_CLIFF"]
    c3 = all([
        g0._check(log, "planted smooth (M2) -> EXPLAINED", v_sm["verdict"].startswith("EXPLAINED"),
                  f"ratio={v_sm['cliff_ratio']:.2f} tight={v_sm['tight_count']}/6"),
        g0._check(log, "planted cliff -> CLIFF-PREFERRED",
                  v_cl["verdict"].startswith("CLIFF-PREFERRED"),
                  f"cliff wRMSE={fcl['wrmse']:.4f} vs best smooth "
                  f"{v_cl['fits'][v_cl['best_smooth']]['wrmse']:.4f}"),
        g0._check(log, "planted cliff PARAMS recovered: nu_x brackets the planted 3.0; r_top ~ 0.92",
                  2.8489 < fcl["nu_x"] < 3.9261 and abs(fcl["r_top"] - 0.92) < 0.02,
                  f"nu_x={fcl['nu_x']:.4f} r_top={_fmt(fcl['r_top'],'.4f')}"),
        g0._check(log, "nu_x degeneracy flag: fires on a SHALLOW curve (no collapse in range; cliff -> "
                  "constant), NOT on the planted cliff",
                  fit_cliff(c, m1_f(c, math.log(30.0), 1.0), np.full(len(c), 1 / SE_FLOOR ** 2)
                            )["nu_x_degenerate"] is True and fcl.get("nu_x_degenerate") is False,
                  f"shallow={fit_cliff(c, m1_f(c, math.log(30.0), 1.0), np.full(len(c), 1 / SE_FLOOR ** 2))['nu_x_degenerate']} "
                  f"cliff={fcl.get('nu_x_degenerate')}"),
        g0._check(log, "planted +/-0.09 structure -> SHAPE-UNEXPLAINED",
                  v_st["verdict"].startswith("SHAPE-UNEXPLAINED"),
                  f"tight={v_st['tight_count']}/6 loose={v_st['loose_count']}/6 ratio={v_st['cliff_ratio']:.2f}"),
        g0._check(log, "decide() is DETERMINISTIC (double-run numeric identity; underpins the reread lock)",
                  json.dumps({k: v_sm[k] for k in ("fits", "cliff_ratio", "locked_P1")}, sort_keys=True)
                  == json.dumps({k: decide(_synth_points([float(x) for x in y_sm]), dict(okc))[k]
                                 for k in ("fits", "cliff_ratio", "locked_P1")}, sort_keys=True), ""),
    ])
    R["check3_selection"] = {"pass": c3}

    # ---- CHECK 4: weights formula + floor (high-n points cannot dominate) ---------------- #
    log("\n(4) Weights -- w = 1/max(SE,0.02)^2; floor binds tiny-SE (high-n) points")
    pts = _synth_points([0.9] * 6, ses=[0.001, 0.005, 0.02, 0.03, 0.05, 0.10])
    ws = [p["weight"] for p in pts]
    c4 = all([
        g0._check(log, "SE below floor -> weight capped at 1/0.02^2 = 2500",
                  ws[0] == 2500.0 and ws[1] == 2500.0 and ws[2] == 2500.0,
                  f"w={[round(x,1) for x in ws[:3]]}"),
        g0._check(log, "SE above floor -> 1/SE^2", abs(ws[4] - 1 / 0.05 ** 2) < 1e-9 and
                  abs(ws[5] - 1 / 0.10 ** 2) < 1e-9, f"w={[round(x,1) for x in ws[3:]]}"),
    ])
    R["check4_weights"] = {"pass": c4}

    # ---- CHECK 5: P1 machinery -- pinned bands match the spec display; sigma sane -------- #
    log("\n(5) P1 -- holdout band edges from (c,rho) match the spec's bracketed [lo,hi]; sigma finite")
    e1, e2, e3 = (band_edges(HOLDOUT[h]["c"], HOLDOUT[h]["rho"]) for h in ("H1", "H2", "H3"))
    disp = lambda e: [round(e[0], 2), round(e[1], 2)]
    val, sig = predict(m2_f, fB["theta"], fB["cov"], 3.30)
    c5 = all([
        g0._check(log, "H1 (3.30, 4.5) -> [1.56, 7.00]", disp(e1) == [1.56, 7.00], f"{disp(e1)}"),
        g0._check(log, "H2 (12.0, 4.5) -> [5.66, 25.46]", disp(e2) == [5.66, 25.46], f"{disp(e2)}"),
        g0._check(log, "H3 (sqrt(18), 1.5) -> exact [3.4641, 5.1962] (pins the committed swept center)",
                  abs(e3[0] - 3.4641) < 1e-3 and abs(e3[1] - 5.1962) < 1e-3,
                  f"exact [{e3[0]:.4f}, {e3[1]:.4f}]; 2-dp display [3.46, 5.20]"),
        g0._check(log, "sigma_pred finite and >= SE_FLOOR", math.isfinite(sig) and sig >= SE_FLOOR,
                  f"pred={val:.4f} sigma={sig:.4f}"),
    ])
    R["check5_p1"] = {"pass": c5, "H3_exact": e3}

    order = ["check0_inputs", "check1_dedupe", "check2_recovery", "check3_selection",
             "check4_weights", "check5_p1"]
    allpass = all(R[k]["pass"] for k in order)
    log("\n=== SANDBOX SUMMARY ===")
    for k in order:
        log(f"  {'PASS' if R[k]['pass'] else 'FAIL'}  {k}")
    log(f"  OVERALL: {'ALL PASS' if allpass else 'FAILURES PRESENT'}")
    os.makedirs(RESDIR, exist_ok=True)
    outp = os.path.join(RESDIR, "gateB_sandbox.json")
    g0._dump_json(outp, {"gate": "relay-gateB", "stage": "0-sandbox", "all_pass": allpass,
                         "framing": FRAMING, "checks": {k: R[k]["pass"] for k in order}})
    log(f"  [written -> {os.path.relpath(outp)}]  (sandbox artifact; NOT committed)")
    return allpass


def verdict_test(log):
    log("=== VERDICT-ENGINE SYNTHETIC TEST (EXPLAINED / SHAPE / CLIFF / NM; CPU only) ===")
    rng = np.random.default_rng(23)
    c = np.array(_real_centers())
    okc = {"ok": True, "reasons": [], "sweep_sha256": SWEEP_SHA256, "gate3_recs_sha16": GATE3_RECS_SHA16}
    allok = True

    def case(name, v, want):
        nonlocal allok
        ok = v["verdict"].startswith(want)
        allok &= ok
        log(f"  [{'OK' if ok else 'WRONG'}] {name}: {v['verdict'][:76]}")
        return v

    vE = case("smooth -> EXPLAINED",
              decide(_synth_points(list(m2_f(c, math.log(2.0)) + rng.normal(0, 0.004, 6))), okc),
              "EXPLAINED")
    vS = case("structure -> SHAPE-UNEXPLAINED",
              decide(_synth_points(list(m1_f(c, math.log(2.5), 1.8) +
                                        np.array([0.09, -0.09, 0.09, -0.09, 0.09, -0.09]))), okc),
              "SHAPE-UNEXPLAINED")
    case("cliff -> CLIFF-PREFERRED",
         decide(_synth_points(list(np.where(c < 3.0, 0.92, CLIFF_FLOOR) + rng.normal(0, 0.004, 6))), okc),
         "CLIFF-PREFERRED")
    bad = {"ok": False, "reasons": ["gate0_bandsweep.json sha256 deadbeef != pinned"],
           "sweep_sha256": "deadbeef", "gate3_recs_sha16": "deadbeef"}
    case("tampered inputs -> NO-MEASUREMENT", decide([], bad), "NO-MEASUREMENT")

    import tempfile
    for tag, vv in (("explained", vE), ("shape", vS)):
        p = os.path.join(tempfile.gettempdir(), f"_gB_md_{tag}.md")
        try:
            _write_md(p, vv, _synth_points([0.5] * 6), [], 0.0, {"code": "selftest", "spec": "selftest"})
            b = open(p, "rb").read()
            ok = (b"LOCKED P1" in b) and (b"Honesty clause" in b) and all(x < 128 for x in b)
            allok &= ok
            log(f"  [{'OK' if ok else 'WRONG'}] _write_md({tag}) renders LOCKED P1 + honesty clause, ASCII")
            os.remove(p)
        except Exception as e:
            allok = False
            log(f"  [WRONG] _write_md({tag}) crashed: {e!r}")
    log(f"  VERDICT-ENGINE: {'ALL BRANCHES CORRECT' if allok else 'BRANCH ERRORS -- FIX'}")
    return allok


# ===================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--reread", action="store_true")
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
    if args.stage1:
        stage1(log)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
