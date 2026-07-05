# Relay Program — Gate 1 Spec: Multi-Hop Loss-Law Probe (Offline, H=5)

**Status**: Drafted (Claude). Ready for ratification + CC handoff.
**Date**: 2026-07-04
**Builds on**: Gate-0 PASS (commit 8361553): offline two-stage relay crosses the horizon
(e2e 0.9264 +/- 0.0233 SE vs direct span-3.0 -0.0017 +/- 0.0064 SE, n=7).
**Gate type**: measurement gate (loss-law shape), offline, no new core machinery.
STOP-and-report; no sweep beyond H=5; hop-length variation is Gate 2, not this.

## Objective (one sentence)
Measure how relay fidelity decays with hop count — extend the Gate-0 chain to H=5
identical stages and determine whether per-hop loss is constant (a routing budget,
extrapolable) or compounding (repeater error amplification, a depth limit).

## Honest framing
Each repeater re-injects into a fresh span-1.5 network; "compound span 1.5*H" is an
information-path claim (H successive square-law demodulations), never a claim about one
physical spectrum. State this in all outputs. No new beat-the-floor claim is made here:
the chain-vs-direct question was settled at Gate-0; this gate's claim is the SHAPE of
the loss curve. Cite the committed Gate-0/b0f7664 floors; do not re-run direct arms.

## Design
**Chain.** m0 -> stage 1 -> repeater -> stage 2 -> ... -> stage 5. Every stage a fresh
independent-seed network, Gate-0 construction exactly: N=500, single continuous
log-uniform comb, span 1.5, lambda=0.1, ER mean-degree 10, beta=1.0, SPP=2, K=0.24.
Repeater identical to Gate-0 (zero-phase brick-wall to MSG_BAND [0.2,0.9] + affine
rescale to m0 stats), params logged PER HOP. Seed scheme: chain i, stage s uses a
derived seed (CC defines the derivation, logs it, and verifies no collision with any
decoy base: 40000/60000/80000 family).

**Replication bridge.** Stage 1 IS Gate-0's stage A (same seeds, same m0). Anchor rule
unchanged: |mean - 0.986| <= max(2*SE, 0.02) on the intersection; per-seed values are
expected digit-exact against gate0_relay_reposed.json — verify, log, STOP on miss.

**Metrics per chain.** Cumulative r2(m_k, m0) for k=1..5; per-hop fidelity
r2(m_k, processed-m_{k-1}); repeater rms_in/scale per hop.

## Pre-registered loss-law readout (no post-hoc moving)
- Cumulative ratios rho_k = r2_k / r2_{k-1}, k=2..5, computed per seed on the
  intersection. Validity guard (probe bounded by the reservoir): rho_k enters the trend
  test only where r2_{k-1} > 0.2; ratios of near-floor numbers are no measurement.
- **(A) BUDGET-LIKE**: no downward trend in rho_k with k beyond seed sigma (slope of
  rho_k vs k consistent with 0 within 2*SE) -> per-hop cost is constant; report the
  budget (mean rho) and extrapolated H_half (r2 crossing 0.5).
- **(B) COMPOUNDING**: rho_k decreases with k beyond seed sigma -> staging has a depth
  limit; report the fitted decay and measured/extrapolated H_half.
- **(C) NO-MEASUREMENT**: anchor miss; or violation check fails to collapse; or decoy
  elevated; or intersection underpowered (<2; target n>=5, extend seeds to 8,9 if
  attrition bites). Fix instrument, re-run. A null from a broken instrument is not a
  finding.
- Either A or B is first-class: A prices the ladder; B bounds it.

## Conditions (K=0.24 throughout; n>=5 paired seeds, seeds 0-7 first)
1. **Chain** (H=5), full per-hop logging.
2. **End-to-end decoy at depth**: never-injected m0' scored at H=5, full Phase-1 decoy
   protocol per stage (bases as in Gate-0; bar: not elevated, 0.2 convention).
3. **Filter-violation at depth** (the re-posed Gate-0 check, run on the full chain):
   message [2,9] rad/s; first repeater pass-band [0.2,0.9] deletes it; per-seed e2e
   r2 < 0.1 at H=5, with the small-rms_in / large-scale signature logged. Must collapse.
4. **Scramble on stage 3** (one condition): degree-matched random coupling mid-chain;
   loss-law topology-genericity check.

## Statistics preconditions (named)
ESP-honest paired intersection across ALL five stages and all arms (nested esp->ok_slow,
per stage per seed); finite-sd (>=2) or add seeds; report per-seed values, mean +/- SE
(labeled SE), and the intersection list in the json.

## Deliverables
results/R/gate1_multihop.json (per chain/seed/stage r2s, ratios, ESP flags,
intersection, repeater params per hop, seed-derivation record) + short .md verdict
(anchor, violation, decoy, A/B/C classification). STOP-and-report. No interpretation
beyond the outcome mapping; Gate-2 (hop-length) and the mechanism-decomposition gate
are separate decisions.

## Compute
~5 stages x 8 seeds x ~50 s + violation chain + decoys: well under 2 h on the 4080.
One GPU process; float64; x64 on.

## Standing rules
Gate-first; CPU sandbox before GPU (chain wiring, per-hop logging, H-stage intersection
logic, violation plumbing), then 1-seed smoke (full H=5 pass; stage-1 must reproduce the
committed seed-0 0.981470), then battery on Jason's go. Single-variable: hop count only.
core/integrator_corotating.py stays untouched. Verify load-bearing claims in sandbox
before accepting numbers. Commit only on Jason's exact word, author
Jason Dury <jason@eridos.ai>, no co-author line.

---
**End of spec.** If A: Gate-2 (hop-length sweep) prices hop size against the measured
budget. If B: the depth limit itself becomes the headline and Gate-2 tests whether
shorter hops relax it. Mechanism-decomposition gate (broadband side-finding) can run
in parallel after this gate's sandbox — separate ratification.
