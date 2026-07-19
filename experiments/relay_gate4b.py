"""
experiments/relay_gate4b.py
===========================

Relay Program -- Gate 4b: K-extension (de-censor K*(s)).
Per the byte-frozen relay_gate4b_kextension_spec.md (ratified 2026-07-19, roadside paste W2).

MEASUREMENT gate. Single-hop Phase-1-protocol routing cells, extended above the committed
coupling ceiling K=0.24 to locate the true per-span argmax K_hat(s) -- or the constraint that
binds first. No chains. No contrast verdict: Gate-4's SHORT-WINS and D stand as committed and
are NOT re-litigated here.

Reuse-by-import (committed siblings NEVER edited):
  * p1 = D_phase1_routing : the physics (build_system, am_input, integrate_Ks, demod_capacity,
                            consistency ESP, band partition, am_window/n_sub_for) -- the exact
                            committed single-hop routing cell.
  * g4 = relay_gate4       : the committed K-rule (k_star_lookup), the committed landscape
                            loader (_load_landscape, PHASE1_JSON/SHA256), landscape_cells,
                            NOISE_FLOOR, LOOKUP_SPANS.
  * g0 = relay_gate0       : provenance/io/stats helpers (RESDIR, _sha12, _dump_json, _mstats,
                            _check).
  * gL = relay_gateL       : the committed decoy census (COMMITTED_DECOY_BASES + its own fresh
                            GATEL_DECOY_BASE -> 26 committed families) + footprint rule.

Modes (ONE GPU process; float64; x64 ON):
  --sandbox      : Stage 1. CPU-ONLY proven-to-fire checks (lookup-replay digit-exact vs the
                   committed k_star block, collision census over 26 families, verdict-engine
                   synthetic over every class x suffix, NM seal, locked-numbers, evaluate-at-use,
                   no-truncation). Prints a battery wall-clock ESTIMATE. Nothing on the GPU.
  --verdict-test : CPU. Exercises decide() across every class/suffix/NM branch. stdout only.
  --smoke        : Stage 2 (separate go). GPU-light seed-0: all four anchor cells digit-exact
                   6dp + one extension cell per span logged.
  --run          : Stage 3 (BATTERY -- ONLY on Jason's explicit word "G4B BATTERY GO").
  --reread       : CPU re-decide from unchanged committed recs; loud-fail on numeric drift.

HARD STOP after --smoke. Battery waits on Jason's explicit word. Nothing committed.
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import tempfile
import hashlib

# CPU-only modes: force the JAX CPU backend BEFORE jax loads (via p1), so no GPU is touched.
if any(a in sys.argv for a in ("--sandbox", "--verdict-test", "--reread")):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))                      # experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root

import numpy as np                                                 # noqa: E402
import D_phase1_routing as p1                                      # noqa: E402  (jax x64 on import)
import relay_gate0 as g0                                           # noqa: E402
import relay_gate4 as g4                                           # noqa: E402  (K-rule + landscape)
import relay_gateL as gL                                           # noqa: E402  (decoy census)

RESDIR = g0.RESDIR
SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "relay_gate4b_kextension_spec.md")
G4B_JSON = os.path.join(RESDIR, "gate4b_kextension.json")
G4B_MD = os.path.join(RESDIR, "gate4b_kextension.md")
G4B_SANDBOX = os.path.join(RESDIR, "gate4b_sandbox.json")
G4B_SMOKE = os.path.join(RESDIR, "gate4b_smoke.json")
GATE4_JSON = os.path.join(RESDIR, "gate4_hoptrade.json")           # committed k_star replay target

# --- ratified design (spec T1-T4 defaults) -------------------------------------------------- #
SPANS = [1.0, 1.5, 2.0, 3.0]                     # T2: all four (K-rule uniformity)
EXT_GRID = [0.28, 0.32, 0.40, 0.48]              # T1: half-step then 0.08 steps to 2x ceiling
COMMITTED_K = list(p1.K_GRID)                     # [0.0, 0.08, 0.12, 0.16, 0.24]
UNION_K = sorted(set(COMMITTED_K) | set(EXT_GRID))  # argmax basis
SEEDS = list(range(10))                          # T4: 0..9 (matches committed landscape statistic)
ANCHOR_K = 0.24
GRID_MAX = max(EXT_GRID)                          # 0.48 -- the NEW ceiling (STILL-CENSORED edge)
FULL_N = 10                                       # per-point n; below this needs a named cause
NOISE_FLOOR = g4.NOISE_FLOOR                      # 0.05 (ported unchanged, edit 5)
DECOY_BAR = 0.2                                   # fresh-decoy p95 elevation bar
ESP_DEGRADE_MIN = 5                               # >=5 of 10 seeds ESP-fail -> ESP-DEGRADED
DELTA_FLOOR = 0.02                                # coarse-tie proxy for the unpairable edge only

# fresh decoys: one family per span (spec-ratified bases); footprint rule inherited from Gate-L.
G4B_DECOY_BASE = {1.0: 700000, 1.5: 720000, 2.0: 740000, 3.0: 760000}
DECOY_FAMILY_SPAN = gL.DECOY_FAMILY_SPAN          # SEED_MAX*200 + (N_DEC-1) = 1859
# 26 committed families = Gate-L's 23 committed bases + Gate-L's own 3 fresh (now committed).
COMMITTED_FAMILIES_26 = list(gL.COMMITTED_DECOY_BASES) + list(gL.GATEL_DECOY_BASE.values())

PHASE1_JSON = g4.PHASE1_JSON
PHASE1_SHA256 = g4.PHASE1_SHA256

FRAMING = (
    "Gate-4b de-censors the K*(s) lookup: the committed Gate-4 found K*(s)=0.24=grid-max at every "
    "span (right-censored, AT-CEILING). This gate extends the grid to 2x the ceiling and reads the "
    "true per-span argmax K_hat(s) by the committed RAW-mean K-rule -- or names the constraint that "
    "binds first (STILL-CENSORED / ESP-BOUNDED-CEILING). MEASUREMENT gate; no mechanism claims; "
    "Gate-4's contrast verdict is not re-litigated."
)

PREREG = {
    "spec_sha256_full": "09f63de5249f73a12e9d0e9c88dd65db5ef44d5591d56097eb36591dde1b887f",
    "order_of_record": "spec byte-freeze PRECEDED the GPU-light smoke preview (seed-0); "
                       "pre-registration is on the record as intact under unblinded direction",
    "battery_go_ruling": "Jason, 2026-07-19: full grid as frozen, NO T-overrides -- span 3.0 "
                         "stays in. Dropping it minutes after a smoke preview reported it at "
                         "noise would narrow a frozen run set in a way that reads as "
                         "data-driven; span 3.0 emerging from noise above the old ceiling is "
                         "exactly the surprise this gate exists to catch. A preview at one "
                         "seed and the smoke's K points is not evidence against it.",
}


# ================================================================================= #
#  provenance / io helpers
# ================================================================================= #
def _sha256_full(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashes():
    return {"code": g0._sha12(__file__), "spec": g0._sha12(SPEC_PATH)}


def _env_full():
    import platform
    try:
        import jax
        jaxv = jax.__version__
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        jaxv, x64 = "n/a", None
    return {"python": sys.version.split()[0], "numpy": np.__version__, "jax": jaxv,
            "jax_enable_x64": x64, "platform": platform.platform(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<default>"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}


def _f(x, spec=".4f"):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        return format(x, spec)
    except Exception:
        return str(x)


def _argmax_lowest(means):
    """argmax over the ascending UNION_K; np.argmax returns the FIRST max -> lowest-K tie-break
    (ported verbatim from the committed k_star_lookup rule)."""
    return int(np.argmax(np.asarray(means, float)))


def _esp_honest_mean(r2, ok):
    vals = [r2[s] for s in range(len(r2)) if ok[s]]
    return float(np.mean(vals)) if vals else None


# ================================================================================= #
#  decoys + collision census (26 committed families + 4 fresh; proven at sandbox)
# ================================================================================= #
def g4b_decoys(span, seed_i, L, dt_in):
    """Fresh never-injected same-class decoys for the span's family (committed footprint rule:
    base + seed*200 + d, N_DEC draws)."""
    base = G4B_DECOY_BASE[span]
    return [p1.slow_bandlimited(L, dt_in, p1.MSG_LO, p1.MSG_HI, seed=base + seed_i * 200 + d)
            for d in range(p1.N_DEC)]


def collision_census(extra_fresh=None):
    """FULL collision matrix: the 4 fresh Gate-4b families vs all 26 committed families.
    Each family j occupies integer seeds [base_j, base_j + DECOY_FAMILY_SPAN]. Returns
    (report, ok); ok iff no fresh family overlaps any committed family (or any other fresh one).
    `extra_fresh` injects an extra colliding base to prove the census fires."""
    span_fp = DECOY_FAMILY_SPAN
    fresh = {f"G4B:{s}": G4B_DECOY_BASE[s] for s in SPANS}
    if extra_fresh is not None:
        fresh = dict(fresh)
        fresh["G4B:INJECT"] = extra_fresh
    committed = {f"COMMITTED[{i}]": b for i, b in enumerate(COMMITTED_FAMILIES_26)}
    all_bases = {**committed, **fresh}
    names = list(all_bases)
    overlaps = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a0, b0 = all_bases[names[x]], all_bases[names[y]]
            # skip committed-committed pairs (those are on the record already); test any pair
            # that includes a fresh family.
            if names[x] not in fresh and names[y] not in fresh:
                continue
            if not (a0 + span_fp < b0 or b0 + span_fp < a0):
                overlaps.append([names[x], names[y]])
    min_fresh = min(fresh.values())
    max_committed = max(committed.values()) + span_fp
    rep = {"fresh_bases": fresh, "family_span": span_fp,
           "n_committed_families": len(committed), "min_fresh_base": min_fresh,
           "max_committed_footprint": max_committed,
           "pairwise_overlaps_involving_fresh": overlaps,
           "clear_of_committed": bool(min_fresh > max_committed and not overlaps)}
    return rep, (len(overlaps) == 0)


# ================================================================================= #
#  lookup replay (sandbox instrument): committed K-rule replay, digit-exact vs k_star block
# ================================================================================= #
def _digit_exact(a, b):
    """Full-float equality for the committed replay (same blob, same code -> bit-exact)."""
    if a is None or b is None:
        return a is b
    return float(a) == float(b)


def lookup_replay(perturb=None):
    """Recompute the committed K*(s) table from the sha-verified committed landscape blob and
    match the committed gate4_hoptrade.json k_star block digit-exact. `perturb` (span,K)->value
    corrupts one committed table cell to prove the check fires. Returns (evidence, ok)."""
    land, sha, sha_ok = g4._load_landscape()
    committed = json.load(open(GATE4_JSON))["k_star"]["detail"]
    ks = g4.k_star_lookup(land)
    mism = []
    for span in g4.LOOKUP_SPANS:
        got = ks[span]
        ref = dict(committed[str(span)])
        if perturb is not None and perturb[0] == span:
            ref = json.loads(json.dumps(ref))
            ref["table"][str(perturb[1])] = ref["table"][str(perturb[1])] + 1e-6
        # K_star + flags
        for key in ("K_star", "noise_flag", "at_grid_max", "monotone_routing", "n_seeds"):
            if str(got.get(key)) != str(ref.get(key)):
                mism.append({"span": span, "field": key, "got": got.get(key), "ref": ref.get(key)})
        # argmax_margin
        if not _digit_exact(got.get("argmax_margin"), ref.get("argmax_margin")):
            mism.append({"span": span, "field": "argmax_margin",
                         "got": got.get("argmax_margin"), "ref": ref.get("argmax_margin")})
        # per-K table floats + esp_ok
        for K in COMMITTED_K:
            if not _digit_exact(got["table"].get(str(K)), ref["table"].get(str(K))):
                mism.append({"span": span, "field": f"table[{K}]",
                             "got": got["table"].get(str(K)), "ref": ref["table"].get(str(K))})
            if str(got["esp_ok"].get(str(K))) != str(ref["esp_ok"].get(str(K))):
                mism.append({"span": span, "field": f"esp_ok[{K}]",
                             "got": got["esp_ok"].get(str(K)), "ref": ref["esp_ok"].get(str(K))})
    ev = {"landscape_sha256": sha, "landscape_sha_ok": bool(sha_ok),
          "pinned_sha256": PHASE1_SHA256, "n_spans_replayed": len(g4.LOOKUP_SPANS),
          "mismatches": mism, "k_rule": "RAW mean r2_d0 over seeds 0-9; argmax lowest-K tie; "
          "NOISE_FLOOR=%.2f (ported verbatim from relay_gate4.k_star_lookup)" % NOISE_FLOOR}
    return ev, bool(sha_ok and not mism)


# ================================================================================= #
#  physics: the committed single-hop Phase-1 routing cell (reused verbatim)
# ================================================================================= #
def _span_ctx(span, nseeds):
    """Per-span setup exactly matching D_phase1_routing.run (build, window, bands, fast enc)."""
    n_sub = p1.n_sub_for(span)
    dt_in, W0, eval_start, L, delays, stride = p1.am_window(span)
    sl = slice(eval_start, L)
    specs = [p1.build_system(s, p1.N, span) for s in range(nseeds)]
    bands = p1.band_indices(specs[0].omega)
    rngs = [np.random.default_rng(5000 + s) for s in range(nseeds)]
    m_fast = [p1.masked_encoding(sp.omega, bands["fast"], r) for sp, r in zip(specs, rngs)]
    return {"n_sub": n_sub, "dt_in": dt_in, "eval_start": eval_start, "L": L, "delays": delays,
            "sl": sl, "specs": specs, "bands": bands, "m_fast": m_fast}


def anchor_r2d0(span, seed, ctx):
    """Re-run the committed (span, K=0.24) cell inside the committed K_GRID batch (batch shape
    matched -> bit-exact per the span-2.0 ULP precedent). r2_d0 is decoy-independent."""
    sp = ctx["specs"][seed]
    s_msg, u_am = p1.am_input(ctx["L"], ctx["dt_in"], 1000 + seed)
    decoys = g4b_decoys(span, seed, ctx["L"], ctx["dt_in"])   # (r2_d0 decoy-independent)
    Xmain = p1.integrate_Ks(sp.omega, sp.L, ctx["m_fast"][seed], sp.z0, u_am,
                            p1.K_GRID, ctx["dt_in"], ctx["n_sub"])
    ki = list(p1.K_GRID).index(ANCHOR_K)
    demod = p1.demod_capacity(Xmain[ki], ctx["bands"]["slow"], s_msg, decoys,
                              ctx["delays"], ctx["sl"], "full")
    return float(demod["r2_d0"])


def ext_cells_seed(span, seed, ctx, esp=True):
    """The extension-K cells for one seed: r2_d0 (fresh-decoy-scored), decoy_p95, ESP ok_slow.
    Extension K's batched together (no committed reference -> batch shape free)."""
    sp = ctx["specs"][seed]
    s_msg, u_am = p1.am_input(ctx["L"], ctx["dt_in"], 1000 + seed)
    decoys = g4b_decoys(span, seed, ctx["L"], ctx["dt_in"])
    Xmain = p1.integrate_Ks(sp.omega, sp.L, ctx["m_fast"][seed], sp.z0, u_am,
                            EXT_GRID, ctx["dt_in"], ctx["n_sub"])
    Xrep = None
    if esp:
        rep = p1.replica_spec(sp, 9000 + seed)
        Xrep = p1.integrate_Ks(sp.omega, sp.L, ctx["m_fast"][seed], rep.z0, u_am,
                               EXT_GRID, ctx["dt_in"], ctx["n_sub"])
    out = {}
    for ki, K in enumerate(EXT_GRID):
        X = Xmain[ki]
        demod = p1.demod_capacity(X, ctx["bands"]["slow"], s_msg, decoys,
                                  ctx["delays"], ctx["sl"], "full")
        ok_slow = None
        if esp:
            d_slow = p1.consistency_distance(X[:, ctx["bands"]["slow"]],
                                             Xrep[ki][:, ctx["bands"]["slow"]], ctx["sl"])
            ok_slow = bool(d_slow < p1.ESP_EPS)
        out[K] = {"r2_d0": float(demod["r2_d0"]), "decoy_p95": float(demod["decoy_p95"]),
                  "ok_slow": ok_slow}
    return out


# ================================================================================= #
#  union table + per-span classification
# ================================================================================= #
def union_span_data(span, cells_committed, ext_by_seed):
    """Per-span union mean-vs-K table with per-seed arrays (RAW; no ESP gating).
    committed K -> committed landscape per-seed; extension K -> this gate's re-run per-seed."""
    data = {}
    for K in UNION_K:
        if K in EXT_GRID:
            r2 = [ext_by_seed[s][K]["r2_d0"] for s in SEEDS]
            ok = [bool(ext_by_seed[s][K]["ok_slow"]) for s in SEEDS]
            dp = [ext_by_seed[s][K]["decoy_p95"] for s in SEEDS]
            source = "extension"
        else:
            r2 = [cells_committed[(span, K, s)]["r2_d0"] for s in SEEDS]
            ok = [bool(cells_committed[(span, K, s)]["ok_slow"]) for s in SEEDS]
            dp = [cells_committed[(span, K, s)]["decoy_p95"] for s in SEEDS]
            source = "committed"
        n = len([x for x in r2 if x is not None])
        data[K] = {"r2": r2, "ok": ok, "decoy_p95": dp, "n": n,
                   "mean": float(np.mean(r2)) if n else float("nan"), "source": source}
    return data


def classify_span(span, data):
    """Union-grid argmax K_hat(s) + class + suffixes (all evaluate-at-use; nothing stored)."""
    means = [data[K]["mean"] for K in UNION_K]
    ai = _argmax_lowest(means)
    K_hat = UNION_K[ai]
    mhat = data[K_hat]["mean"]

    if abs(mhat) < NOISE_FLOOR:
        base = "NOISE-ARGMAX"
    elif K_hat < GRID_MAX:
        base = "DE-CENSORED"
    else:
        base = "STILL-CENSORED"

    # margins over both union-grid neighbors
    margins = {}
    if ai > 0:
        margins["left_K"] = UNION_K[ai - 1]
        margins["left_margin"] = float(mhat - means[ai - 1])
    if ai < len(UNION_K) - 1:
        margins["right_K"] = UNION_K[ai + 1]
        margins["right_margin"] = float(mhat - means[ai + 1])

    suffixes = []
    notes = {}

    # -TIE-WITHIN-SE : top two means differ by < 1 SE of their per-seed paired difference
    order = sorted(range(len(UNION_K)), key=lambda j: -means[j])
    k1, k2 = UNION_K[order[0]], UNION_K[order[1]]
    gap = means[order[0]] - means[order[1]]
    pairable = (data[k1]["n"] == FULL_N and data[k2]["n"] == FULL_N)
    if pairable:
        D = [data[k1]["r2"][s] - data[k2]["r2"][s] for s in range(FULL_N)]
        se = g0._mstats(D)["se"] or 0.0
        if gap < se:
            suffixes.append("-TIE-WITHIN-SE")
            notes["tie"] = {"K_set": [k1, k2], "gap": float(gap), "se_paired": float(se),
                            "rule": "gap < 1*SE_paired (evaluate-at-use; not stored)"}
    elif gap < DELTA_FLOOR:
        suffixes.append("-UNPAIRABLE-TIE")
        notes["tie"] = {"K_set": [k1, k2], "gap": float(gap),
                        "why": "seed sets differ (n<%d) -> cannot pair" % FULL_N}

    # -ESP-DEGRADED : any point with >=5/10 ESP fails; K_hat degraded -> suffix
    esp_fail = {K: sum(1 for o in data[K]["ok"] if not o) for K in UNION_K}
    esp_degraded = {K: esp_fail[K] >= ESP_DEGRADE_MIN for K in UNION_K}
    notes["esp_degraded"] = {str(K): {"fail": esp_fail[K], "degraded": bool(esp_degraded[K])}
                             for K in UNION_K}
    if esp_degraded[K_hat]:
        suffixes.append("-ESP-DEGRADED")

    # ESP-BOUNDED-CEILING : all K above some non-degraded K_e are degraded, and K_hat <= K_e
    esp_bounded = False
    boundary = None
    for j, Ke in enumerate(UNION_K):
        above = UNION_K[j + 1:]
        if above and (not esp_degraded[Ke]) and all(esp_degraded[k] for k in above) \
                and K_hat <= Ke:
            esp_bounded = True
            boundary = Ke
            break
    notes["esp_bounded_ceiling"] = {"observed": bool(esp_bounded), "K_e": boundary}

    # -UNSTABLE-ARGMAX : leave-one-seed-out moves K_hat for any drop
    loso_moves = {}
    if all(data[K]["n"] == FULL_N for K in UNION_K):
        for drop in range(FULL_N):
            m2 = [float(np.mean([data[K]["r2"][s] for s in range(FULL_N) if s != drop]))
                  for K in UNION_K]
            khat2 = UNION_K[_argmax_lowest(m2)]
            if khat2 != K_hat:
                loso_moves[drop] = khat2
        if loso_moves:
            suffixes.append("-UNSTABLE-ARGMAX")
    notes["loso"] = {"moves": {str(k): v for k, v in loso_moves.items()},
                     "n_drops": FULL_N, "stable": not loso_moves}

    # DIVERGENT-ARGMAX (non-gating): ESP-honest-mean argmax vs RAW argmax
    esp_means = [_esp_honest_mean(data[K]["r2"], data[K]["ok"]) for K in UNION_K]
    valid = [(j, esp_means[j]) for j in range(len(UNION_K)) if esp_means[j] is not None]
    khat_esp = None
    if valid:
        best = max(v for _, v in valid)
        khat_esp = UNION_K[min(j for j, v in valid if v == best)]   # lowest-K tie
    divergent = bool(khat_esp is not None and khat_esp != K_hat)
    notes["esp_honest"] = {"argmax_K": khat_esp, "means": {str(UNION_K[j]): esp_means[j]
                           for j in range(len(UNION_K))},
                           "esp_ok_members": {str(K): [s for s in SEEDS if data[K]["ok"][s]]
                                              for K in UNION_K}}
    notes["divergent_argmax"] = bool(divergent)

    class_full = base + "".join(suffixes)
    return {"span": span, "K_hat": K_hat, "mean_at_Khat": float(mhat), "base": base,
            "suffixes": suffixes, "class": class_full, "margins": margins,
            "table": {str(K): float(data[K]["mean"]) for K in UNION_K},
            "n_per_K": {str(K): data[K]["n"] for K in UNION_K},
            "esp_bounded_ceiling": bool(esp_bounded), "divergent_argmax": bool(divergent),
            "notes": notes}


# ================================================================================= #
#  verdict engine (gate precedence -> per-span classification -> consequence map)
# ================================================================================= #
def decide(state):
    """Pre-registered verdict. Instruments-first gate-level seal: lookup-replay -> anchor ->
    fresh-decoy -> thin-n. Only if all pass does per-span classification + the consequence map
    run. Verdict string is NM-prefixed on any gate trip (pattern-matched by _write_md)."""
    gate = {}
    gate["lookup_ok"] = bool(state["lookup"]["ok"])
    gate["anchor_ok"] = bool(state["anchor"]["ok"])
    gate["decoy_ok"] = bool(state["decoy"]["ok"])
    gate["thin_ok"] = bool(state["thin"]["ok"])

    nm = None
    if not gate["lookup_ok"]:
        nm = "NO-MEASUREMENT (lookup-replay mismatch: committed K-rule port failed digit-exact)"
    elif not gate["anchor_ok"]:
        nm = "NO-MEASUREMENT (anchor-row 6dp miss vs committed landscape)"
    elif not gate["decoy_ok"]:
        nm = "NO-MEASUREMENT (fresh decoy elevated: p95 > %.2f)" % DECOY_BAR
    elif not gate["thin_ok"]:
        nm = "NO-MEASUREMENT (extension point with < 2 recorded seeds)"

    v = {"verdict": nm if nm else "GATE-4B MEASURED", "gate": gate,
         "instruments": {"lookup": state["lookup"].get("evidence"),
                         "anchor": state["anchor"], "decoy": state["decoy"],
                         "thin": state["thin"]}}
    if nm:
        return v

    per_span = {}
    conseq = []
    for span in SPANS:
        cs = classify_span(span, state["spans"][span])
        per_span[str(span)] = cs
        base = cs["base"]
        tag = "span %s: %s (K_hat=%s)" % (span, cs["class"], cs["K_hat"])
        if base == "DE-CENSORED":
            conseq.append(tag + " -- feeds the Gate-4 scope-note forecast (compression above the "
                          "old ceiling) + future S1 revision; NO retro-edit of Gate-4.")
        elif base == "STILL-CENSORED":
            conseq.append(tag + " -- banks the extend-or-stop call (right-censored at the NEW "
                          "ceiling %.2f); further extension is a future call." % GRID_MAX)
        elif base == "NOISE-ARGMAX":
            conseq.append(tag + " -- winning |mean| < %.2f; context-only, no de-censoring claim." % NOISE_FLOOR)
        if cs["esp_bounded_ceiling"]:
            conseq.append("span %s: ESP-BOUNDED-CEILING -- physical-ceiling observation "
                          "relevant to GC and GM design." % span)
        if cs["divergent_argmax"]:
            conseq.append("span %s: DIVERGENT-ARGMAX -- RAW vs ESP-honest argmax disagree; "
                          "interpretation is Jason's at read-through." % span)
    v["per_span"] = per_span
    v["consequence"] = conseq
    # band-level NM present? (any point n<FULL_N with a named cause is a report-only condition;
    # thin-n<2 already sealed above). Flag n<FULL_N for disclosure without sealing.
    underpop = [(str(span), str(K), state["spans"][span][K]["n"])
                for span in SPANS for K in UNION_K if state["spans"][span][K]["n"] < FULL_N]
    if underpop:
        v["verdict"] = v["verdict"] + " [under-populated points present -- see per-point n]"
        v["underpopulated"] = underpop
    return v


# ================================================================================= #
#  markdown render (NM seal enforced)
# ================================================================================= #
def _write_md(path, v, wall, hashes, colrep, env=None):
    is_nm = str(v.get("verdict", "")).startswith("NO-MEASUREMENT")
    L = []
    L.append("# Relay Gate-4b -- K-extension (de-censor K*(s))")
    L.append("")
    L.append("Spec: relay_gate4b_kextension_spec.md (sha12 %s). Harness sha12 %s."
             % (hashes.get("spec"), hashes.get("code")))
    L.append("Verdict: **%s**" % v.get("verdict"))
    L.append("")
    gate = v.get("gate", {})
    L.append("## Instrument checks (gate-level)")
    L.append("")
    L.append("- lookup-replay (committed K-rule, digit-exact vs gate4 k_star): %s"
             % ("OK" if gate.get("lookup_ok") else "FAIL"))
    an = v.get("instruments", {}).get("anchor", {})
    L.append("- anchor row (span,K=0.24 seeds 0-9, 6dp gate): %s"
             % ("OK" if gate.get("anchor_ok") else "FAIL"))
    dc = v.get("instruments", {}).get("decoy", {})
    L.append("- fresh decoys (26-family census clear; per-cell p95 <= %.2f): %s (max p95 %s)"
             % (DECOY_BAR, "OK" if gate.get("decoy_ok") else "FAIL", _f(dc.get("max_p95"))))
    L.append("- thin-intersection (every extension point n >= 2): %s"
             % ("OK" if gate.get("thin_ok") else "FAIL"))
    L.append("")

    if is_nm:
        L.append("## SEALED (NM-disclosure rule -- code-enforced)")
        L.append("")
        L.append("This gate is at a GATE-LEVEL NO-MEASUREMENT: an instrument tripped (see the "
                 "Verdict line + Instrument checks above). Per the NM-disclosure rule, the SEALED "
                 "sections -- the per-span mean-vs-K tables, the class table, the ESP boundary "
                 "map, and the consequence lines -- are WITHHELD until the NM resolution is "
                 "ratified. Resolution decisions are made blind. Re-render with --reread once "
                 "the resolution is ratified.")
        with open(path, "w") as f:
            f.write("\n".join(L) + "\n")
        return

    # anchor diagnostic
    if an:
        L.append("## Anchor (digit-exact 6dp gate; bit-exact DIAGNOSTIC)")
        L.append("")
        for span in SPANS:
            ps = an.get("per_span", {}).get(str(span))
            if not ps:
                continue
            L.append("- span %s: digit6 %s | bit-exact %d/%d (diagnostic)"
                     % (span, "OK" if ps.get("digit_ok") else "FAIL",
                        ps.get("bit_cells", 0), ps.get("total", 0)))
        L.append("")

    # per-span tables
    L.append("## Per-span mean-vs-K (union grid) + class")
    L.append("")
    ks = "  ".join("%.2f" % K for K in UNION_K)
    for span in SPANS:
        cs = v["per_span"][str(span)]
        L.append("### span %s -- %s (K_hat=%s)" % (span, cs["class"], cs["K_hat"]))
        L.append("```")
        L.append("K       " + ks)
        row = "mean  " + "  ".join("%5.2f" % cs["table"][str(K)] for K in UNION_K)
        L.append(row)
        nrow = "n     " + "  ".join("%5d" % cs["n_per_K"][str(K)] for K in UNION_K)
        L.append(nrow)
        # ESP-degraded map
        espd = cs["notes"]["esp_degraded"]
        drow = "ESPdeg" + "  ".join(("  deg" if espd[str(K)]["degraded"] else "   . ")
                                    for K in UNION_K)
        L.append(drow)
        L.append("```")
        m = cs.get("margins", {})
        if "left_K" in m:
            L.append("- margin over left neighbor K=%s: %s" % (m["left_K"], _f(m["left_margin"])))
        if "right_K" in m:
            L.append("- margin over right neighbor K=%s: %s" % (m["right_K"], _f(m["right_margin"])))
        if cs["notes"].get("tie", {}).get("K_set"):
            L.append("- tie: %s" % json.dumps(cs["notes"]["tie"]))
        if cs["divergent_argmax"]:
            eh = cs["notes"]["esp_honest"]
            L.append("- DIVERGENT-ARGMAX: RAW K_hat=%s vs ESP-honest K_hat=%s"
                     % (cs["K_hat"], eh.get("argmax_K")))
        if not cs["notes"]["loso"]["stable"]:
            L.append("- UNSTABLE-ARGMAX (LOSO moves): %s" % json.dumps(cs["notes"]["loso"]["moves"]))
        if cs["esp_bounded_ceiling"]:
            L.append("- ESP-BOUNDED-CEILING at K_e=%s" % cs["notes"]["esp_bounded_ceiling"]["K_e"])
        L.append("")

    # decoy max cell
    if dc.get("max_cell"):
        L.append("## Fresh decoy (max cell)")
        L.append("- %s (bar %.2f)" % (json.dumps(dc["max_cell"]), DECOY_BAR))
        L.append("")

    # consequence map
    L.append("## Consequence map (reported; no automatic actions)")
    L.append("")
    for line in v.get("consequence", []):
        L.append("- " + line)
    L.append("")
    L.append("## Provenance")
    L.append("- committed landscape: phase1_routing.json sha256 %s (decade_drive b0f7664)"
             % PHASE1_SHA256)
    L.append("- fresh decoy bases: %s | family span %d | 26 committed families censused"
             % (json.dumps(G4B_DECOY_BASE), DECOY_FAMILY_SPAN))
    L.append("- wall_clock_s: %s" % _f(wall, ".1f"))
    if env:
        L.append("- env: %s" % json.dumps(env))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ================================================================================= #
#  wall-clock estimate (advisory; from the committed Gate-4 per-stage timing)
# ================================================================================= #
def wall_estimate():
    """Rough battery wall-clock estimate. Model: per integrate_Ks call, time ~ nK * L(span),
    calibrated so a 5-K call at span 1.5 ~ 13 s (Gate-4 per-stage-integration figure). Battery
    per (span,seed): anchor(K_GRID,5) + ext-main(4) + ext-rep(4). ADVISORY, non-gating."""
    L_of = {}
    for span in SPANS:
        _, _, _, L, _, _ = p1.am_window(span)
        L_of[span] = L
    per_K_ref = 13.0 / (5.0 * L_of[1.5])          # calibrate: 5-K @ span1.5 -> 13 s
    total = 0.0
    per_span = {}
    for span in SPANS:
        calls_K = (5 + 4 + 4)                     # anchor 5K + ext-main 4K + ext-rep 4K
        t_seed = calls_K * L_of[span] * per_K_ref
        per_span[str(span)] = {"L": int(L_of[span]), "s_per_seed": round(t_seed, 1),
                               "s_all_seeds": round(t_seed * len(SEEDS), 1)}
        total += t_seed * len(SEEDS)
    return {"estimate_s": round(total, 1), "estimate_min": round(total / 60.0, 1),
            "per_span": per_span, "seeds": len(SEEDS),
            "model": "time ~ nK*L(span); calib 5K@1.5=13s (Gate-4 per-stage); ADVISORY only; "
                     "span-3 dominated; excludes compile/warmup"}


# ================================================================================= #
#  battery + smoke + reread
# ================================================================================= #
def _committed_cells():
    land, sha, ok = g4._load_landscape()
    if not ok:
        raise SystemExit("committed landscape sha mismatch (%s != pinned) -- NM" % sha)
    return g4.landscape_cells(land)


def _build_state(cells, anchor, spans_ext, decoy_max):
    """Assemble decide()'s state from committed cells + re-run anchor/ext."""
    spans_data = {}
    thin_off = []
    for span in SPANS:
        data = union_span_data(span, cells, spans_ext[span])
        spans_data[span] = data
        for K in EXT_GRID:
            if data[K]["n"] < 2:
                thin_off.append([str(span), str(K), data[K]["n"]])
    lookup_ev, lookup_ok = lookup_replay()
    decoy_ok = decoy_max["max_p95"] is None or decoy_max["max_p95"] <= DECOY_BAR
    return {"lookup": {"ok": lookup_ok, "evidence": lookup_ev},
            "anchor": anchor, "decoy": {**decoy_max, "ok": decoy_ok},
            "thin": {"ok": len(thin_off) == 0, "offenders": thin_off},
            "spans": spans_data}


def run(log, nseeds=10):
    log("=== RELAY GATE-4B :: BATTERY (K-extension) ===")
    t0 = time.perf_counter()
    cells = _committed_cells()
    anchor = {"per_span": {}, "ok": True}
    spans_ext = {span: {} for span in SPANS}
    decoy_max = {"max_p95": None, "max_cell": None}
    for span in SPANS:
        ctx = _span_ctx(span, nseeds)
        # anchor row (all seeds; 6dp gate vs committed)
        pssp = {"per_seed": {}, "digit_ok": True, "bit_cells": 0, "total": 0}
        for s in range(nseeds):
            got = anchor_r2d0(span, s, ctx)
            ref = cells[(span, ANCHOR_K, s)]["r2_d0"]
            diff = float(abs(got - ref))
            ulp = int(round(diff / float(np.spacing(ref)))) if diff > 0.0 else 0
            d6 = bool(round(got, 6) == round(ref, 6))
            bit = bool(diff == 0.0)
            pssp["per_seed"][str(s)] = {"got": got, "ref": ref, "diff": diff, "ulp": ulp,
                                        "digit6_ok": d6, "bit_exact": bit}
            pssp["digit_ok"] = pssp["digit_ok"] and d6
            pssp["bit_cells"] += int(bit)
            pssp["total"] += 1
        anchor["per_span"][str(span)] = pssp
        anchor["ok"] = anchor["ok"] and pssp["digit_ok"]
        # extension cells (all seeds)
        for s in range(nseeds):
            ec = ext_cells_seed(span, s, ctx, esp=True)
            spans_ext[span][s] = ec
            for K in EXT_GRID:
                dp = ec[K]["decoy_p95"]
                if decoy_max["max_p95"] is None or dp > decoy_max["max_p95"]:
                    decoy_max["max_p95"] = dp
                    decoy_max["max_cell"] = {"span": span, "K": K, "seed": s, "decoy_p95": dp}
        log("  span %s done (%.0fs)" % (span, time.perf_counter() - t0))
    wall = time.perf_counter() - t0
    state = _build_state(cells, anchor, spans_ext, decoy_max)
    v = decide(state)
    hashes = _hashes()
    # recs = per-cell extension record (byte-locked for reread)
    recs = {}
    for span in SPANS:
        for s in range(nseeds):
            for K in EXT_GRID:
                recs["%s|%s|%s" % (span, K, s)] = spans_ext[span][s][K]
    payload = {"verdict": v, "recs": recs, "anchor": anchor,
               "k_star_extended": {str(span): v["per_span"][str(span)] for span in SPANS}
               if "per_span" in v else None,
               "lookup_replay": state["lookup"]["evidence"], "decoy": state["decoy"],
               "wall_clock_s": wall, "hashes": hashes, "env": _env_full(),
               "collision": collision_census()[0], "framing": FRAMING,
               "preregistration": PREREG}
    g0._dump_json(G4B_JSON, payload)
    _write_md(G4B_MD, v, wall, hashes, payload["collision"], _env_full())
    log("  %s" % v["verdict"])
    log("  -> %s + %s" % (os.path.basename(G4B_JSON), os.path.basename(G4B_MD)))


def smoke(log):
    log("=== RELAY GATE-4B :: GPU-LIGHT SMOKE (seed 0) ===")
    cells = _committed_cells()
    rows = []
    all_ok = True
    for span in SPANS:
        ctx = _span_ctx(span, 1)
        got = anchor_r2d0(span, 0, ctx)
        ref = cells[(span, ANCHOR_K, 0)]["r2_d0"]
        d6 = bool(round(got, 6) == round(ref, 6))
        bit = bool(got == ref)
        all_ok = all_ok and d6
        # one extension cell (K=0.28), main only (light)
        ec = ext_cells_seed(span, 0, ctx, esp=False)
        k0 = EXT_GRID[0]
        rows.append({"span": span, "anchor_got": got, "anchor_ref": ref, "digit6_ok": d6,
                     "bit_exact": bit, "ext_K": k0, "ext_r2_d0": ec[k0]["r2_d0"],
                     "ext_decoy_p95": ec[k0]["decoy_p95"]})
        log("  span %s: anchor r2_d0=%.6f (ref %.6f) digit6=%s bit=%s | ext K=%.2f r2_d0=%.4f "
            "decoy_p95=%.4f" % (span, got, ref, d6, bit, k0, ec[k0]["r2_d0"], ec[k0]["decoy_p95"]))
    ev, lok = lookup_replay()
    crep, cok = collision_census()
    out = {"seed": 0, "anchor_all_digit6_ok": all_ok, "rows": rows,
           "lookup_replay_ok": lok, "collision_clear": cok, "env": _env_full(),
           "hashes": _hashes()}
    g0._dump_json(G4B_SMOKE, out)
    log("  anchor digit6 all-ok: %s | lookup-replay ok: %s | collision clear: %s" % (all_ok, lok, cok))
    return bool(all_ok and lok and cok)


def reread(log):
    log("=== RELAY GATE-4B :: REREAD (re-decide from unchanged recs; no GPU) ===")
    assert os.path.exists(G4B_JSON), "missing battery record %s -- run --run first" % G4B_JSON
    nm = json.load(open(G4B_JSON))
    flat = nm["recs"]
    spans_ext = {span: {} for span in SPANS}
    for key, r in flat.items():
        span, K, s = key.split("|")
        spans_ext[float(span)].setdefault(int(s), {})[float(K)] = r
    # LOCKED-NUMBERS: every cell re-serializes byte-identically to the record
    for key in flat:
        span, K, s = key.split("|")
        cur = spans_ext[float(span)][int(s)][float(K)]
        assert json.dumps(cur, sort_keys=True) == json.dumps(flat[key], sort_keys=True), \
            "cell '%s' drifted from the battery record" % key
    log("  [integrity] all extension cells byte-identical to the battery record: OK")
    cells = _committed_cells()
    decoy_max = {"max_p95": None, "max_cell": None}
    for span in SPANS:
        for s in spans_ext[span]:
            for K in EXT_GRID:
                dp = spans_ext[span][s][K]["decoy_p95"]
                if decoy_max["max_p95"] is None or dp > decoy_max["max_p95"]:
                    decoy_max["max_p95"] = dp
                    decoy_max["max_cell"] = {"span": span, "K": K, "seed": s, "decoy_p95": dp}
    state = _build_state(cells, nm["anchor"], spans_ext, decoy_max)
    v = decide(state)
    hashes = _hashes()
    payload = {**nm, "verdict": v, "hashes": hashes,
               "run_hashes": nm.get("run_hashes") or nm.get("hashes"),
               "reread": "re-decided from unchanged recs; no GPU; classification re-evaluated "
                         "(deltas/margins/tie-SE recomputed, not stored); no measured number changed"}
    g0._dump_json(G4B_JSON, payload)
    _write_md(G4B_MD, v, nm.get("wall_clock_s", 0.0), hashes, nm.get("collision", {}), _env_full())
    log("  %s" % v["verdict"])
    log("  run sha (immutable): %s; current code sha: %s"
        % ((payload["run_hashes"] or {}).get("code"), hashes["code"]))


# ================================================================================= #
#  verdict-engine synthetic test + CPU sandbox
# ================================================================================= #
def _synth_span(mean_by_K, ok_by_K=None, per_seed_spread=0.0):
    """Build a synthetic per-span `data` dict with per-seed arrays whose seed-means hit
    mean_by_K exactly (spread controls the paired SE)."""
    data = {}
    for K in UNION_K:
        m = mean_by_K[K]
        j = UNION_K.index(K)
        if per_seed_spread:
            # INDEPENDENT per-K zero-mean noise (distinct phase per K) so paired differences
            # between two K's carry real variance -> a non-degenerate paired SE.
            noise = [per_seed_spread * math.sin(1.7 * s + 0.9 * j + 0.3) for s in range(FULL_N)]
            mn = float(np.mean(noise))
            r2 = [m + noise[s] - mn for s in range(FULL_N)]
        else:
            r2 = [m] * FULL_N
        ok = [True] * FULL_N
        if ok_by_K is not None and K in ok_by_K:
            nfail = ok_by_K[K]
            ok = [s >= nfail for s in range(FULL_N)]
        data[K] = {"r2": r2, "ok": ok, "decoy_p95": [0.0] * FULL_N,
                   "n": FULL_N, "mean": float(np.mean(r2)), "source": "synthetic"}
    return data


def _clean_gates():
    return ({"ok": True, "evidence": {"mismatches": []}},
            {"ok": True, "per_span": {str(s): {"digit_ok": True, "bit_cells": FULL_N,
                                               "total": FULL_N} for s in SPANS}},
            {"ok": True, "max_p95": 0.03, "max_cell": {"decoy_p95": 0.03}},
            {"ok": True, "offenders": []})


def _state_from_spans(spans_data, lookup=None, anchor=None, decoy=None, thin=None):
    lk, an, dc, th = _clean_gates()
    return {"lookup": lookup or lk, "anchor": anchor or an, "decoy": decoy or dc,
            "thin": thin or th, "spans": spans_data}


def verdict_test(log):
    log("=== GATE-4B VERDICT-ENGINE SYNTHETIC (every class x suffix x NM branch) ===")
    ok = True

    def rising(top_K, topval=0.8):
        # monotone-ish rising to top_K, then flat below
        m = {}
        for K in UNION_K:
            m[K] = topval - 0.02 * max(0, UNION_K.index(top_K) - UNION_K.index(K)) \
                if K <= top_K else topval - 0.05 * (UNION_K.index(K) - UNION_K.index(top_K))
        m[top_K] = topval
        return m

    # DE-CENSORED: interior argmax at 0.32
    d = _synth_span(rising(0.32))
    cs = classify_span(1.0, d)
    ok &= g0._check(log, "DE-CENSORED (interior argmax)", cs["base"] == "DE-CENSORED",
                    "%s K_hat=%s" % (cs["class"], cs["K_hat"]))

    # STILL-CENSORED: argmax at 0.48 (new ceiling)
    d = _synth_span(rising(0.48))
    cs = classify_span(1.0, d)
    ok &= g0._check(log, "STILL-CENSORED (argmax at new ceiling)", cs["base"] == "STILL-CENSORED",
                    "%s K_hat=%s" % (cs["class"], cs["K_hat"]))

    # NOISE-ARGMAX: all means ~0
    d = _synth_span({K: 0.001 * (UNION_K.index(K) + 1) for K in UNION_K})
    cs = classify_span(3.0, d)
    ok &= g0._check(log, "NOISE-ARGMAX (|mean at K_hat| < 0.05)", cs["base"] == "NOISE-ARGMAX",
                    "%s mean=%.4f" % (cs["class"], cs["mean_at_Khat"]))

    # -TIE-WITHIN-SE: top two within 1 SE (large spread, tiny gap)
    m = rising(0.32)
    m[0.28] = m[0.32] - 0.0005
    d = _synth_span(m, per_seed_spread=0.5)
    cs = classify_span(1.0, d)
    ok &= g0._check(log, "-TIE-WITHIN-SE fires", "-TIE-WITHIN-SE" in cs["suffixes"],
                    "%s tie=%s" % (cs["class"], cs["notes"].get("tie")))
    # and does NOT fire with negligible spread + real gap
    d2 = _synth_span(rising(0.32), per_seed_spread=0.0)
    cs2 = classify_span(1.0, d2)
    ok &= g0._check(log, "-TIE-WITHIN-SE silent on clean separation",
                    "-TIE-WITHIN-SE" not in cs2["suffixes"], "no tie")

    # -ESP-DEGRADED: argmax point has >=5 ESP fails
    d = _synth_span(rising(0.32), ok_by_K={0.32: 6})
    cs = classify_span(2.0, d)
    ok &= g0._check(log, "-ESP-DEGRADED fires (K_hat 6/10 ESP-fail)", "-ESP-DEGRADED" in cs["suffixes"],
                    cs["class"])

    # ESP-BOUNDED-CEILING: everything above 0.24 degraded, argmax at 0.24
    d = _synth_span(rising(0.24), ok_by_K={K: 6 for K in UNION_K if K > 0.24})
    cs = classify_span(2.0, d)
    ok &= g0._check(log, "ESP-BOUNDED-CEILING fires", cs["esp_bounded_ceiling"],
                    "K_e=%s K_hat=%s" % (cs["notes"]["esp_bounded_ceiling"]["K_e"], cs["K_hat"]))

    # -UNSTABLE-ARGMAX: 0.28 and 0.32 near-tied at the top; seed 0 tips 0.32, so dropping it flips
    m = {K: 0.30 for K in UNION_K}
    m[0.28] = 0.80
    m[0.32] = 0.80
    d = _synth_span(m, per_seed_spread=0.0)
    d[0.32]["r2"][0] += 0.05                    # only seed 0 lifts 0.32 above 0.28
    d[0.32]["mean"] = float(np.mean(d[0.32]["r2"]))
    cs = classify_span(1.0, d)
    ok &= g0._check(log, "-UNSTABLE-ARGMAX fires (LOSO moves K_hat)",
                    "-UNSTABLE-ARGMAX" in cs["suffixes"], json.dumps(cs["notes"]["loso"]["moves"]))

    # DIVERGENT-ARGMAX: RAW argmax 0.32 (high raw, but its ONE esp-ok seed is low) vs
    # ESP-honest argmax 0.40 (all esp-ok at 0.70)
    d = _synth_span({K: 0.30 for K in UNION_K})
    d[0.32]["r2"] = [0.50] + [(8.0 - 0.50) / 9.0] * 9   # RAW mean 0.80; esp-ok only seed 0 (0.50)
    d[0.32]["ok"] = [s == 0 for s in range(FULL_N)]
    d[0.32]["mean"] = float(np.mean(d[0.32]["r2"]))
    d[0.40]["r2"] = [0.70] * FULL_N                     # RAW mean 0.70; all esp-ok
    d[0.40]["ok"] = [True] * FULL_N
    d[0.40]["mean"] = 0.70
    cs = classify_span(1.0, d)
    ok &= g0._check(log, "DIVERGENT-ARGMAX fires (RAW 0.32 vs ESP-honest 0.40)",
                    cs["divergent_argmax"] and cs["K_hat"] == 0.32
                    and cs["notes"]["esp_honest"].get("argmax_K") == 0.40,
                    "raw=%s esp=%s" % (cs["K_hat"], cs["notes"]["esp_honest"].get("argmax_K")))

    # NM gates
    good = {span: _synth_span(rising(0.32)) for span in SPANS}
    lk, an, dc, th = _clean_gates()
    v = decide(_state_from_spans(good))
    ok &= g0._check(log, "clean state -> MEASURED (no NM)", v["verdict"].startswith("GATE-4B MEASURED"),
                    v["verdict"])
    v = decide(_state_from_spans(good, lookup={"ok": False, "evidence": {"mismatches": [1]}}))
    ok &= g0._check(log, "lookup mismatch -> NM (sealed)", v["verdict"].startswith("NO-MEASUREMENT")
                    and "lookup" in v["verdict"], v["verdict"][:50])
    v = decide(_state_from_spans(good, anchor={"ok": False, "per_span": {}}))
    ok &= g0._check(log, "anchor miss -> NM", v["verdict"].startswith("NO-MEASUREMENT")
                    and "anchor" in v["verdict"], v["verdict"][:50])
    v = decide(_state_from_spans(good, decoy={"ok": False, "max_p95": 0.4, "max_cell": {}}))
    ok &= g0._check(log, "decoy elevated -> NM", v["verdict"].startswith("NO-MEASUREMENT")
                    and "decoy" in v["verdict"], v["verdict"][:50])
    v = decide(_state_from_spans(good, thin={"ok": False, "offenders": [["1.0", "0.28", 1]]}))
    ok &= g0._check(log, "thin-n -> NM", v["verdict"].startswith("NO-MEASUREMENT")
                    and "2 recorded seeds" in v["verdict"], v["verdict"][:50])

    # NM seal in _write_md
    v_nm = decide(_state_from_spans(good, lookup={"ok": False, "evidence": {"mismatches": [1]}}))
    p = os.path.join(tempfile.gettempdir(), "_g4b_nm.md")
    _write_md(p, v_nm, 0.0, _hashes(), {})
    t = open(p).read(); os.remove(p)
    sealed = ("SEALED (NM-disclosure" in t and "Per-span mean-vs-K" not in t
              and "Consequence map" not in t and "..." not in t)
    ok &= g0._check(log, "NM render SEALS tables/consequence + shows SEALED notice", sealed, "sealed")

    log("  verdict-engine synthetic: %s" % ("ALL PASS" if ok else "FAIL"))
    return bool(ok)


def sandbox(log):
    log("=== RELAY GATE-4B :: STAGE-1 CPU SANDBOX (no GPU; proven-to-fire) ===")
    log("    " + json.dumps(_env_full()))
    results = {}

    # (1) lookup-replay digit-exact vs committed k_star block; fires on a perturbed cell
    ev, lok = lookup_replay()
    ev_bad, lok_bad = lookup_replay(perturb=(1.5, 0.24))
    c1 = g0._check(log, "lookup-replay matches committed k_star digit-exact", lok,
                   "spans=%d mism=%d sha_ok=%s" % (ev["n_spans_replayed"], len(ev["mismatches"]),
                                                   ev["landscape_sha_ok"]))
    c1 &= g0._check(log, "  ...and FIRES on a 1e-6 perturbation of a committed cell", not lok_bad,
                    "mism=%d" % len(ev_bad["mismatches"]))
    results["check1_lookup_replay"] = {"pass": bool(c1), "landscape_sha_ok": ev["landscape_sha_ok"],
                                       "n_mismatch_clean": len(ev["mismatches"]),
                                       "n_mismatch_perturbed": len(ev_bad["mismatches"])}

    # (2) collision census: 4 fresh vs 26 committed clear; fires on an injected colliding base
    crep, cok = collision_census()
    crep_bad, cok_bad = collision_census(extra_fresh=COMMITTED_FAMILIES_26[0] + 10)
    c2 = g0._check(log, "collision census clear (4 fresh vs 26 committed)", cok,
                   "n_committed=%d clear=%s max_fp=%d min_fresh=%d" %
                   (crep["n_committed_families"], crep["clear_of_committed"],
                    crep["max_committed_footprint"], crep["min_fresh_base"]))
    c2 &= g0._check(log, "  ...and FIRES on an injected colliding base", not cok_bad,
                    "overlaps=%d" % len(crep_bad["pairwise_overlaps_involving_fresh"]))
    c2 &= g0._check(log, "  census covers exactly 26 committed families",
                    crep["n_committed_families"] == 26, "n=%d" % crep["n_committed_families"])
    results["check2_collision"] = {"pass": bool(c2), "n_committed": crep["n_committed_families"],
                                   "clear": crep["clear_of_committed"],
                                   "n_overlap_injected": len(crep_bad["pairwise_overlaps_involving_fresh"])}

    # (3) verdict-engine synthetic (every class x suffix x NM branch, each proven to fire)
    c3 = verdict_test(log)
    results["check3_verdict_engine"] = {"pass": bool(c3)}

    # (4) anchor gate fires on a 6dp miss (synthetic)
    good = {span: _synth_span({K: 0.5 for K in UNION_K}) for span in SPANS}
    an_bad = {"ok": False, "per_span": {"1.0": {"digit_ok": False, "bit_cells": 0, "total": 10}}}
    v_bad = decide(_state_from_spans(good, anchor=an_bad))
    v_ok = decide(_state_from_spans(good))
    c4 = g0._check(log, "anchor 6dp miss -> NM (fires)", v_bad["verdict"].startswith("NO-MEASUREMENT"),
                   v_bad["verdict"][:40])
    c4 &= g0._check(log, "  ...silent when anchor clean", not v_ok["verdict"].startswith("NO-MEASUREMENT"),
                    "measured")
    results["check4_anchor_gate"] = {"pass": bool(c4)}

    # (5) decoy elevation gate fires (fresh p95 > 0.2)
    v_bad = decide(_state_from_spans(good, decoy={"ok": False, "max_p95": 0.35, "max_cell": {"decoy_p95": 0.35}}))
    c5 = g0._check(log, "fresh decoy p95>0.2 -> NM (fires)", v_bad["verdict"].startswith("NO-MEASUREMENT")
                   and "decoy" in v_bad["verdict"], v_bad["verdict"][:40])
    results["check5_decoy_gate"] = {"pass": bool(c5), "bar": DECOY_BAR}

    # (6) thin-n<2 gate fires
    v_bad = decide(_state_from_spans(good, thin={"ok": False, "offenders": [["3.0", "0.48", 1]]}))
    c6 = g0._check(log, "extension point n<2 -> NM (fires)", v_bad["verdict"].startswith("NO-MEASUREMENT")
                   and "2 recorded seeds" in v_bad["verdict"], v_bad["verdict"][:40])
    results["check6_thin_gate"] = {"pass": bool(c6)}

    # (7) NM seal render (proven above in verdict_test; assert directly here too)
    v_nm = decide(_state_from_spans(good, lookup={"ok": False, "evidence": {"mismatches": [1]}}))
    p = os.path.join(tempfile.gettempdir(), "_g4b_sb_nm.md")
    _write_md(p, v_nm, 0.0, _hashes(), {})
    tnm = open(p).read(); os.remove(p)
    # healthy render for contrast (no seal, no truncation)
    p2 = os.path.join(tempfile.gettempdir(), "_g4b_sb_ok.md")
    _write_md(p2, v_ok, 1.0, _hashes(), collision_census()[0], _env_full())
    tok = open(p2).read(); os.remove(p2)
    c7 = g0._check(log, "NM render seals; healthy render has tables + no truncation",
                   ("SEALED" in tnm and "Per-span mean-vs-K" not in tnm
                    and "Per-span mean-vs-K" in tok and "..." not in tok), "seal+notrunc")
    results["check7_nm_seal_notrunc"] = {"pass": bool(c7)}

    # (8) locked-numbers: a serialized cell perturbation breaks the byte-identity assert
    cell = {"r2_d0": 0.5, "decoy_p95": 0.03, "ok_slow": True}
    base = json.dumps(cell, sort_keys=True)
    drifted = json.dumps({**cell, "r2_d0": 0.5 + 1e-9}, sort_keys=True)
    c8 = g0._check(log, "locked-numbers byte-identity fires on 1e-9 drift", base != drifted,
                   "byte-diff detected")
    results["check8_locked_numbers"] = {"pass": bool(c8)}

    # (9) evaluate-at-use: inflating per-seed spread changes the paired SE -> flips the tie
    m = {K: (0.8 if K == 0.32 else 0.79 if K == 0.28 else 0.5) for K in UNION_K}
    tight = classify_span(1.0, _synth_span(m, per_seed_spread=0.0))
    wide = classify_span(1.0, _synth_span(m, per_seed_spread=0.6))
    c9 = g0._check(log, "tie-SE recomputed from primitives (spread flips -TIE-WITHIN-SE)",
                   ("-TIE-WITHIN-SE" not in tight["suffixes"]) and ("-TIE-WITHIN-SE" in wide["suffixes"]),
                   "tight=%s wide=%s" % (tight["suffixes"], wide["suffixes"]))
    results["check9_evaluate_at_use"] = {"pass": bool(c9)}

    # (10) K-rule port equals committed union-table means for committed rows (self-consistency)
    cells = _committed_cells()
    land, _, _ = g4._load_landscape()
    ks = g4.k_star_lookup(land)
    ok10 = True
    for span in SPANS:
        for K in COMMITTED_K:
            mine = float(np.mean([cells[(span, K, s)]["r2_d0"] for s in SEEDS]))
            ref = ks[span]["table"][str(K)]
            ok10 = ok10 and (mine == ref)
    c10 = g0._check(log, "union committed-row means == committed k_rule table (bit-exact)", ok10,
                    "all spans x committed-K")
    results["check10_krule_consistency"] = {"pass": bool(c10)}

    order = ["check1_lookup_replay", "check2_collision", "check3_verdict_engine",
             "check4_anchor_gate", "check5_decoy_gate", "check6_thin_gate",
             "check7_nm_seal_notrunc", "check8_locked_numbers", "check9_evaluate_at_use",
             "check10_krule_consistency"]
    allpass = all(results[k]["pass"] for k in order)

    # wall-clock estimate (advisory)
    est = wall_estimate()
    log("")
    log("  battery wall-clock ESTIMATE: ~%.1f min (%.0fs), seeds=%d [%s]"
        % (est["estimate_min"], est["estimate_s"], est["seeds"], est["model"]))
    for span in SPANS:
        ps = est["per_span"][str(span)]
        log("    span %s: L=%d ~%.0fs/seed ~%.0fs all-seeds" %
            (span, ps["L"], ps["s_per_seed"], ps["s_all_seeds"]))
    log("  fresh decoy bases: %s (family span %d); 26 committed families censused"
        % (json.dumps(G4B_DECOY_BASE), DECOY_FAMILY_SPAN))
    log("")
    log("  SANDBOX: %s (%d/%d checks)" % ("ALL PASS" if allpass else "FAIL",
                                          sum(results[k]["pass"] for k in order), len(order)))
    out = {"pass": bool(allpass), "checks": results, "order": order,
           "wall_estimate": est, "collision": collision_census()[0],
           "fresh_decoy_bases": G4B_DECOY_BASE, "env": _env_full(), "hashes": _hashes(),
           "note": "Stage-1 CPU sandbox; each check stores its comparison payload (fire-evidence)."}
    g0._dump_json(G4B_SANDBOX, out)
    return bool(allpass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--verdict-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reread", action="store_true")
    ap.add_argument("--nseeds", type=int, default=10)
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
        assert 1 <= args.nseeds <= 10, "committed landscape covers seeds 0..9"
        run(log, args.nseeds)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
