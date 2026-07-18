# Relay Gate-L -- Coherent Linear Injection (the other door)

Spec: relay_gateL_coherent_spec.md (sha256 8526ab9d2b25). Harness: experiments/relay_gateL.py (sha256 378e4e02ceae).
Wall-clock 10 min. Fresh-decoy families collision-free: True.

Framing: Gate-L: coherent zero-mean linear injection (u = a*(s - mean), corr(u,s) = 1) vs the committed Gate-3 AM subtrahend -- do first-order slow-tertile coordinates carry the message when linear content is supplied directly? Stage-A, K=0.24, span 1.5, per band. STOP-and-report.

## Verdict

**COHERENT-LIN-LIVE (SUB, RES, SUPRA LIN-LIVE): SUB=LIN-LIVE; RES=LIN-LIVE; SUPRA=LIN-LIVE**

## Pre-data fence (restated)

Why the Gate-3 classifier does NOT port: with z = Z0 + dz, |z|^2 = |Z0|^2 + 2*Re(conj(Z0)*dz) + |dz|^2; under coherent injection dz is first-order in the drive, so the beat term makes |z|^2 carry LINEAR-in-drive content. A high SQ r2 here is NOT a quadratic-mechanism signature. FENCED PRE-DATA: no mechanism claim / retention ratio / channel classification is derived from the SQ or FULL columns (computed + recorded for continuity only). All falsifiable content lives in the LIN column (Re z, Im z are first-order observables under any injection).

## Instrument checks (pre-registered order: anchor, reachability, ladder, decoys, ESP)

1. **Anchor** (AM x SUB per-seed digit-exact 6dp vs REF_TABLE; mean rule): ALL PASS (mean 0.9865 +/- 0.0011). Diagnostic (not gated): bit-exact 8/8 vs the committed gate3 full-precision cells.
2. **Reachability** (linear-arm tripwire, gate |corr(u,s)| >= 0.99): min |corr(u,s)| 1.0000 -> OK. Quadratic arm corr(u^2,s) recorded, no gate.
3. **Linearity ladder** (P_track increasing AND P_track(a) >= 10.0x chance [P3]): SUB: pass=True, ratio=5781538.3185; RES: pass=True, ratio=12746576.1432; SUPRA: pass=True, ratio=13693915.5523.
4. **Decoy floors** (max fresh p95 over all coherent cells): 0.176 at cell SUB|2|LIN; bar 0.2 -> clean.
5. **ESP** nested ok_slow per seed (values below); symmetric intersection per band; memberships in the matrix.

GAP (accepted 2026-07-18, no backfill): the P_slow ladder analog (slow-tertile drive-tracking projection; spec report-only context for the verdict prose) was NOT recorded at the battery -- trajectories are not stored, so it cannot be derived post hoc. Accepted gap; does not affect the verdict or any gated instrument.

## LIN matrix (coherent r2_LIN vs committed AM baseline, per band)

- **SUB**: coherent mean r2_LIN = 1.0000 (n=8) vs committed AM baseline mean -0.0558 / median -0.0121; D_LIN mean = 1.0558 +/- 0.0548, median 1.0121; delta = 0.1096; max fresh LIN decoy p95 = 0.1764; seeds [0, 1, 2, 3, 4, 5, 6, 7] -> **LIN-LIVE**
- **RES**: coherent mean r2_LIN = 1.0000 (n=8) vs committed AM baseline mean -0.0426 / median -0.0438; D_LIN mean = 1.0426 +/- 0.0155, median 1.0438; delta = 0.0310; max fresh LIN decoy p95 = 0.1448; seeds [0, 1, 2, 3, 4, 5, 6, 7] -> **LIN-LIVE**
- **SUPRA**: coherent mean r2_LIN = 1.0000 (n=8) vs committed AM baseline mean -0.0210 / median -0.0124; D_LIN mean = 1.0210 +/- 0.0056, median 1.0124; delta = 0.0200; max fresh LIN decoy p95 = 0.0943; seeds [0, 1, 2, 3, 4, 5, 6, 7] -> **LIN-LIVE**

## Robustness -- leave-one-seed-out (report-only, LIVE bands)

- SUB: LIN-LIVE survives 8/8 single-seed drops (ALL); tightest drop seed 0 (margin +0.9348); deltas 2*SE-governed, both legs re-checked per drop.
- RES: LIN-LIVE survives 8/8 single-seed drops (ALL); tightest drop seed 4 (margin +1.0048); deltas 2*SE-governed, both legs re-checked per drop.
- SUPRA: LIN-LIVE survives 8/8 single-seed drops (ALL); tightest drop seed 2 (margin +0.9980); deltas 2*SE-governed, both legs re-checked per drop.

## ESP (nested ok_slow; d_slow per seed, mean +/- SE; eps = 1e-2)

- SUB: mean 1.39e-04 +/- 1.38e-04; per-seed [0:2.07e-06 1:3.42e-09 2:2.47e-12 3:2.26e-10 4:1.30e-08 5:7.70e-15 6:4.89e-09 7:1.11e-03]
- RES: mean 1.48e-04 +/- 1.48e-04; per-seed [0:2.25e-06 1:3.05e-09 2:2.20e-12 3:2.45e-10 4:1.34e-08 5:8.02e-15 6:4.54e-09 7:1.18e-03]
- SUPRA: mean 2.84e-04 +/- 2.84e-04; per-seed [0:3.59e-06 1:5.20e-09 2:2.96e-12 3:4.14e-10 4:2.11e-08 5:1.57e-14 6:9.03e-09 7:2.27e-03]

## Per-seed D_LIN signs (pattern-fold from Gate-B; observed, not claimed)

- SUB: 0:+ 1:+ 2:+ 3:+ 4:+ 5:+ 6:+ 7:+
- RES: 0:+ 1:+ 2:+ 3:+ 4:+ 5:+ 6:+ 7:+
- SUPRA: 0:+ 1:+ 2:+ 3:+ 4:+ 5:+ 6:+ 7:+

## DC fractions (realized E[s]^2/E[s^2] per band; report-only [P1])

- SUB: DC fraction mean = 0.9116
- RES: DC fraction mean = 0.9369
- SUPRA: DC fraction mean = 0.9348

## Consequence map

COHERENT-LIN-LIVE (band-resolved: SUB, RES, SUPRA): a coherent linear channel exists where injected coherently; C3 gains the scope note Gate-3's consequence map anticipated (|z|^2 is the power-envelope channel; coherent first-order transmission also exists at SUB, RES, SUPRA).

The Phase-3 line-67 erratum WAIT lifts on this landing (class fired: COHERENT-LIN-LIVE (SUB, RES, SUPRA LIN-LIVE)); the landing class selects the replacement wording. Drafting/filing in decade_drive remains Jason's separate call.

## Scope

Readout-level observable-order question (symmetric to Gate-3): does the FIRST-ORDER slow-tertile coordinate carry the message under coherent injection -- NOT the internal transfer path (the Stuart-Landau nonlinearity mixes orders en route). One operating point (K=0.24, span 1.5), stage-A, offline, no chains. Windows/thresholds NOT stored (delta evaluated at verdict from byte-locked (D, SE, n)). STOP-and-report.
