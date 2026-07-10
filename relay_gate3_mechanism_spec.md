# Relay Program — Gate 3 Spec: Mechanism Decomposition (Readout-Channel Ablation)

**Status**: Drafted (Claude). Ready for ratification + CC handoff.
**Date**: 2026-07-05
**Builds on**: Gate-0 band-sweep side-finding (no representable band is untrackable by
the slow readout) + Gate-2 CLOSED (b49834f). **Supersedes** prior ordering: mechanism
decomposition is Gate 3; the hop-length trade moves to Gate 4 (its design consumes
this gate's answer).
**Gate type**: cheap measurement gate, stage-A only, no chains, no new core machinery.
STOP-and-report.

## Objective (one sentence)
Determine WHICH readout channel carries the cross-band message as a function of
message band — the |z|^2 square-law (C3's claim) vs the linear (Re z, Im z) resonant
channel — by readout-feature ablation, settling the broadband side-finding's mechanism
and whether C3 needs a scope note (and giving Gate-4 the channel to optimize for).

## Design
Stage-A Phase-1 replica only (N=500, span 1.5, K=0.24, lambda=0.1, ER-10, beta=1.0,
SPP=2), message injected fast-tertile only, slow-tertile readout. Two axes:
- **Message band** (3): SUB [0.2,0.9] (below slow-band resonance; the standard band),
  RES [2,9] (overlapping slow-tertile natural range [1.00, 3.13]), SUPRA [10,28]
  (above slow resonance, inside fast-tertile range [10.1,31.6], below Nyquist 31.6 —
  carry the recorded caveat: SUPRA messages are carrier-comparable, so injection
  ill-posedness may contribute there).
- **Readout ablation** (3): FULL [Re z, Im z, |z|^2]+bias; SQ [|z|^2]+bias;
  LIN [Re z, Im z]+bias. Ablation is FIT-TIME ONLY: one integration per (band, seed),
  features subset for the three ridge fits — same trajectories, same CV protocol.

## Pre-registered readout (no post-hoc moving)
Retention ratio R_abl = r2_abl / r2_FULL per (band, seed), intersection means.
Validity guard: a band classifies only where FULL r2 > 0.2. Thresholds (pinned now):
channel = **SQ-carried** if R_SQ >= 0.9 and R_LIN < 0.5; **LIN-carried** if R_LIN >=
0.9 and R_SQ < 0.5; **MIXED** otherwise. Per-band classification, no pooling.
- Pre-registered consequence map: SUB=SQ and RES=LIN -> C3 gains a scope note ("|z|^2
  is the sub-resonance envelope channel; resonant linear transmission carries in-band
  and above") and the Phase-3 "envelope-of-envelope rate cost" erratum becomes
  data-backed (drafting/filing it in decade_drive is Jason's separate call).
  SUB=SQ everywhere -> C3 stands unmodified; broadband finding needs another
  explanation (report, no new claim). Any other pattern: report the matrix as-is.
- **NO-MEASUREMENT**: anchor miss; any per-band decoy elevated; intersection
  underpowered (<2; target n>=5, seeds 0-7, extend 8-9 on attrition).

## Instruments
- **Anchor**: FULL x SUB is an exact Phase-1 replica — per-seed digit-exact expectation
  vs committed b0f7664; rule |mean - 0.986| <= max(2*SE, 0.02).
- **Per-band decoys**: never-injected same-class decoy per band, Phase-1 protocol,
  scored under all three ablations (bar 0.2). Distinct decoy bases per band; CC
  proves the collision matrix against all committed seed families before running.
- ESP nested ok_slow per seed (single stage); ESP-honest intersection across all
  (band x seed) runs; per-seed values, mean +/- SE (labeled), intersection in json.

## Deliverables
results/R/gate3_mechanism.json (per band x ablation x seed r2, retention ratios,
decoys, ESP flags, intersection, m-band specs, env provenance) + short .md verdict
(anchor, decoys, 3x3 matrix, per-band classification, consequence-map line).
STOP-and-report. Drift-attribution (WHERE the Gate-2 m0-referenced loss accrues) is
explicitly OUT of scope — it needs instrumented chain runs and is a separate decision.

## Compute
One integration per (band, seed): 3 x 8 = 24 stage-A runs + per-band decoys;
fits are CPU-cheap on stored features. ~25-35 min total on the 4080. One GPU
process; float64; x64; the Oscillator_Reservoir_Program venv.

## Standing rules
Gate-first: CPU sandbox (ablation subsetting correctness — SQ/LIN feature slices
proven against FULL's layout; decoy protocol match; collision matrix; validity-guard
and classifier logic on synthetic matrices), verdict-engine test, then 1-seed smoke
(FULL x SUB must reproduce committed seed-0 0.981470 digit-exact; one RES and one
SUPRA run complete with logging), then battery on Jason's go. Committed artifacts and
core/ untouched. Verify load-bearing claims in sandbox before accepting numbers.
Commit only on Jason's exact word, author Jason Dury <jason@eridos.ai>, no co-author.

---
**End of spec.** Outcome feeds Gate-4 directly: if RES/SUPRA are LIN-carried, the
hop-length trade must price hops against the channel actually in use, and C3's scope
note ships with this gate's commit rather than by argument.
