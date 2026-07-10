# Relay Gate-3 -- mechanism decomposition (readout-channel ablation)

Spec: relay_gate3_mechanism_spec.md (sha256 20b797cb5aa6). Harness: experiments/relay_gate3.py (sha256 1302eadd7ddc).
Seeds run: [0, 1, 2, 3, 4, 5, 6, 7]. K = 0.24, span 1.5. Wall-clock 5 min. Seed scheme collision-free: True.
Provenance: numbers from the battery run (harness sha 908a5811760b, 5 min GPU); verdict RE-FRAMED by --reread (sha 1302eadd7ddc), recs unchanged.

Framing: Stage-A Phase-1 replica (N=500, span 1.5, K=0.24), message fast-tertile injected, slow-tertile readout. Ablation is FIT-TIME ONLY (subset the SAME trajectory's readout features; SQ/LIN are column-subsets of FULL). Retention R_abl=r2_abl/r2_FULL per (band,seed); a band classifies only where FULL r2>0.2. SUPRA is carrier-comparable (injection ill-posedness may contribute). No chains; drift-attribution is out of scope.

## Verdict

**MECHANISM MATRIX -- SUB=SQ-carried, RES=SQ-carried, SUPRA=SQ-carried. C3 STANDS unmodified (SQ-carried across all classified bands, RES included); the broadband side-finding needs another explanation (report, no new claim). [SCOPED -- the injection supplies no linear message content; the LIN-carried branch tested only reservoir-mediated re-encoding into first-order coordinates, which did not occur. NOT an independent refutation of a linear resonant channel. See "Degeneracy and the scope of this verdict".]**

## Instrument checks (pre-registered order)

1. **Anchor (FULL x SUB == committed Phase-1 b0f7664)**: mean 0.986454 (SE 0.001116, n=8); target 0.986 +/- 0.0200 -> mean_ok=True; per-seed digit-exact vs REF_TABLE -> True (8/8 seeds). Anchor provenance: verified against decade_drive b0f7664 results/D/phase1_routing.json (sha256 2e739315141e88c3c5c698f88ed6f84efaae46f7257397c756f58ee4c3965590; working tree == blob at b0f7664). All 40 REF_TABLE cells (4 (span,K) x 10 seeds) faithful: 40/40 r2_d0 + 40/40 ESP flags, 0 discrepancies, at REF_TABLE's stored 6-dp precision. The full-precision phase1_routing mean(seeds 0-7) at (1.5, 0.24) = 0.9864540115271048 is BIT-EXACT to the committed gate1 record anchor mean (REF_TABLE's own 6-dp mean = 0.986454).
2. **Per-band decoys** -- the gate statistic is the PER-CELL decoy p95: the 95th percentile of r2 over 60 never-injected same-class decoy messages, computed independently for each (band, seed, mode) cell. The gate takes the MAX over ALL such cells. This battery: max 0.103 at SUB/FULL (margin 0.097 below the 0.2 bar) -> clean. Runtime-logged readout widths at span 1.5 (slow tertile n=166; FULL 499 / SQ 167 / LIN 333 columns). Observed IN THIS BATTERY ONLY (n=8; no general mode-ordering law is claimed): LIN's nulls sit just above zero and tightly clustered, while FULL's are mostly negative but far more dispersed and contain the single largest null of the run -- which is why the max cell above is a FULL cell. A plausible reading, NOT a hypothesis this gate tests: each mode's ridge penalty is inner-validated on the REAL message, so a mode that cannot fit it is regularized toward the mean and its decoy prediction collapses to the train mean (null just above 0), while a mode that fits well takes a small penalty and its wider feature set overfits the decoy's train split (null negative, high variance). Overfitting drives a null DOWN, not up; leakage would require features that genuinely track the decoy. The gate is sound because it maxes over ALL cells -- it assumes no mode bounds the others.
3. **ESP-honest intersection** (ESP-ok across ALL 3 bands): [0, 1, 2, 3, 4, 5, 6, 7] (n=8).

## Readout observable-order matrix -- WHICH OBSERVABLE ORDER SUFFICES for readout (linear Re,Im vs quadratic |z|^2); demod r2_d0 (intersection means), 3 bands x 3 ablations

Reading: this is an observable-ORDER sufficiency test -- which readout order the ridge demod needs to reconstruct the message -- NOT a claim of a distinct physical transport channel. 'SQ-carried' = the quadratic |z|^2 observable ALONE suffices (linear insufficient); 'LIN-carried' = the linear (Re,Im) observable ALONE suffices; 'MIXED' = neither order alone suffices.

- **SUB** (0.2, 0.9): FULL 0.9865 | SQ 0.9870 | LIN -0.0558  (decoy p95 seed-means: FULL -0.028/SQ 0.008/LIN 0.036)
- **RES** (2.0, 9.0): FULL 0.8601 | SQ 0.8675 | LIN -0.0426  (decoy p95 seed-means: FULL -0.078/SQ -0.043/LIN 0.015)
- **SUPRA** (10.0, 28.0): FULL 0.6650 | SQ 0.6800 | LIN -0.0210  (decoy p95 seed-means: FULL -0.037/SQ -0.022/LIN 0.009)

## Retention + per-band observable-order sufficiency (R_abl = r2_abl / r2_FULL; guard FULL r2 > 0.2)

Ratios are reported UNTOUCHED -- never clipped at 1. R_abl > 1 is expected and means the ablated feature set matches or EXCEEDS FULL out-of-sample, because FULL's extra features mildly overfit (pure out-of-sample variance cost when the dropped order carries nothing). 'Retention' is floor-language: the classifier's 0.9 is a FLOOR, not a cap.

- **SUB**: R_SQ = +1.0005 +/- 0.0002, R_LIN = -0.0564 +/- 0.0555 (n_valid=8, FULL r2 0.986) -> **SQ-carried**
- **RES**: R_SQ = +1.0089 +/- 0.0027, R_LIN = -0.0505 +/- 0.0187 (n_valid=8, FULL r2 0.860) -> **SQ-carried**
- **SUPRA**: R_SQ = +1.0244 +/- 0.0067, R_LIN = -0.0303 +/- 0.0073 (n_valid=8, FULL r2 0.665) -> **SQ-carried**

## Consequence

C3 STANDS unmodified (SQ-carried across all classified bands, RES included); the broadband side-finding needs another explanation (report, no new claim).

## Degeneracy and the scope of this verdict (read with the consequence above)

1. **Injection degeneracy.** Message injected as square-law AM, u = 0.5*sqrt(s)*w (Rademacher w) => u^2 = 0.25*s exactly (max|u^2 - 0.25*s| = 2.8e-17) and, because w is white and independent of s, the input carries no coherent linear message content at any frequency (eval-window |corr(u,s)| <= 0.032 in SUB, RES and SUPRA alike; lagged cross-correlations at the noise floor). LIN-carried therefore had no input-level pathway: it remained reachable only if the reservoir's nonlinearity re-encoded the power envelope into first-order (Re/Im) slow-tertile coordinates -- slow drifts and AM sidebands in Re/Im are linearly readable, so this door was physically live. B1 -- the only C3-modifying branch -- was reachable only through it. This gate tested that door and found it shut (LIN at or below its own never-injected decoy null, 24/24 cells); it did not arbitrate between two live input channels. The AM scheme is the ratified Gate-0 instrument and is not re-litigated here.
2. **What this gate does establish (real, narrow).** A power-encoded message survives cross-band routing to the slow tertile at all three bands (FULL r2 0.9865 / 0.8601 / 0.6650), and the reservoir does NOT re-encode it into a linearly-readable slow-tertile form -- not even at RES, where [2,9] overlaps the slow tertile's natural range [1.00, 3.13].
3. **What it does not establish.** That a live linear resonant transmission channel is impossible. A coherent (non-power) linear injection was never applied; testing it requires a different injection and is a separate gate.
4. **Where the discrimination actually lives.** R_SQ >= 0.9 is near-automatic: SQ is a strict column-subset of FULL, so dropping Re/Im columns that carry nothing generically improves out-of-sample fit (R_SQ >= 1 in 23/24; lone exception SUB seed 5 at 0.9997 -- noise-level, far above the floor). All falsifiable content is in R_LIN < 0.5: LIN scores at or below its own never-injected decoy null in 24/24 cells (its real r2 is negative in 20/24). Read 'SQ-carried' as 'the linear observable order is insufficient AND the message is retained on |z|^2', not as an independent selection of a quadratic channel.
5. **SUPRA is corroborative, not necessary.** B2 fires on SUB + RES alone (a below-guard SUPRA still routes to C3 STANDS). On SUPRA's carrier-comparable ill-posedness: we identify no pathway by which it could produce the LIN/MIXED that alone would break B2 -- any carrier-mediated reconstruction under this power-encoding is intrinsically a |z|^2 path (argued, not categorical).

## Scope

Stage-A only, offline, one operating point (K=0.24), span 1.5. Ablation is fit-time feature-subsetting on ONE trajectory per (band, seed). Drift-attribution (WHERE the Gate-2 m0-referenced loss accrues) is OUT of scope. Gate-4 (hop-length trade) consumes this gate's channel answer.
