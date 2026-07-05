# Relay Program — Gate 2 Spec: Depth Extension (Offline, H=10)

**Status**: Drafted (Claude). Ready for ratification + CC handoff.
**Date**: 2026-07-05
**Builds on**: Gate-1 CLOSED (commit 7d6f3f2): A/budget-like, two-regime law — one-time
insertion loss rho_2=0.941, near-flat steady-state rho_3..5=0.960 with a mild late
drift (slope -0.0148, t=-4.06) UNRESOLVED at n=8/H=5.
**Supersedes** the Gate-1 end-note ordering: depth extension is Gate 2; the hop-length
sweep moves to Gate 3.
**Gate type**: measurement gate, offline, no new core machinery. STOP-and-report.

## Objective (one sentence)
Extend the identical-stage chain from H=5 to H=10 and determine whether the Gate-1
steady-state late drift continues (steady decline -> bounded ladder) or asymptotes
(clean priced budget), replacing the H_half=16 extrapolation with measurement.

## Design leverage (why this is cheap and sharp)
Same seed-derivation scheme, extended: stages 6-10 use build 500+i..900+i, enc
5500+i..5900+i, rep 9500+i..9900+i, carrier 2400+i..2800+i, decoy bases
160000..240000 step 20000 (CC re-proves the full collision matrix over i=0..9,
all families, before anything runs). Because stages 1-5 are seed-identical to
Gate-1, m_1..m_5 and rho_2..5 are DETERMINISTIC replays: **hops 1-5 must reproduce
gate1_multihop.json digit-exact per seed** — a free 5-deep replication bridge, and
it makes the Gate-1 "last-hop" ambiguity structurally decidable: nothing downstream
affects upstream, so if the drift was a noise fluctuation concentrated at hop 5,
rho_6..10 revert to the ~0.96 steady band; if it is genuine depth decline, the
downward trend continues monotonically through the new ratios.

## Honest framing
Compound span 1.5*H is an information-path claim (H successive square-law
demodulations), never one physical spectrum. No new chain-vs-direct claim; committed
floors cited. The Gate-1 two-regime law is the committed prior: the insertion loss
rho_2 is EXCLUDED from this gate's trend statistic by pre-registration (it is a known
one-time cost, not steady-state data).

## Pre-registered readout (no post-hoc moving)
Steady-state ratios rho_k, k=3..10 (8 ratios; validity guard carried: rho_k enters
only where parent r2 > 0.2). Per-seed linear slope of rho_k vs k on the intersection;
ratified thresholds carried: SLOPE_EPS=0.01, threshold max(2*SE, 0.01).
- **(A2) ASYMPTOTIC / PRICED**: |slope_mean| <= threshold -> steady-state is flat at
  depth; report rho_ss (mean rho_3..10), the measured r2_cum(H=10), and the
  budget-model residual (predicted vs measured endpoint). H_half re-derived and still
  labeled extrapolation unless r2 crosses 0.5 in-measurement.
- **(B2) STEADY DECLINE**: slope_mean < -threshold -> the drift is real; fit the decay
  law, report measured/extrapolated H_half with the fit, and the headline becomes the
  bounded ladder.
- Rising slope beyond +threshold -> INSTRUMENT-SUSPICION (flagged NO-MEASUREMENT),
  never A2 (Pin B carried).
- **(C) NO-MEASUREMENT**: anchor miss; prefix-bridge miss (any hop 1-5, any seed, vs
  gate1_multihop.json); violation-at-depth fails to collapse; decoy elevated;
  intersection underpowered (<2; target n>=5, extend seeds 8,9 on attrition).
- Verdict .md always reports slope_mean +/- SE and margin regardless of class (Pin A).

## Conditions (K=0.24; n>=5 paired seeds, seeds 0-7 first)
1. **Chain** H=10, full per-hop logging (r2_cum, r2_hop, rms_in/rms_target/scale).
2. **End-to-end decoy at depth** (H=10), full per-stage Phase-1 protocol, bar 0.2.
3. **Filter-violation at depth** (H=10): [2,9] message, first repeater pass-band
   [0.2,0.9]; per-seed e2e r2 < 0.1 with the small-rms_in / large-scale signature.
No scramble arm: topology-genericity of the loss law was settled at Gate-1;
single-variable discipline — depth only.

## Statistics preconditions (named)
ESP-honest paired intersection across ALL ten stages and all arms (nested
esp->ok_slow); attrition risk doubles at H=10 — finite-sd (>=2) hard floor, target
n>=5, add seeds 8-9 before reading anything if the intersection thins. Per-seed
values, mean +/- SE (labeled), intersection list in the json.

## Deliverables
results/R/gate2_depth.json (per seed/stage r2s, ratios, ESP flags, intersection,
repeater params per hop, seed-derivation record, env provenance incl. interpreter
path) + short .md verdict (anchor, prefix-bridge, violation, decoy, A2/B2/C with
slope +/- SE). STOP-and-report. Gate-3 (hop-length) and the mechanism-decomposition
gate remain separate decisions.

## Compute
~10 stages x 8 seeds x ~50 s + violation chain + decoys: ~1.5-2 h on the 4080.
One GPU process; float64; x64 on; the Oscillator_Reservoir_Program venv.

## Standing rules
Gate-first; CPU sandbox before GPU (extended seed collision matrix, 10-stage chain
wiring + intersection, prefix-bridge loader, rho window k=3..10, violation plumbing),
then 1-seed smoke (full H=10; hops 1-5 must replay gate1 seed-0 digit-exact:
r2_cum = 0.981470 / 0.962334 / 0.931043 / 0.896005 / 0.842353), then battery on
Jason's go. core/integrator_corotating.py and all committed artifacts untouched.
Verify load-bearing claims in sandbox before accepting numbers. Commit only on
Jason's exact word, author Jason Dury <jason@eridos.ai>, no co-author line.

---
**End of spec.** A2 prices the ladder on measurement; B2 bounds it and its decay fit
becomes the design number for Gate-3's hop-length trade. Either is first-class.
