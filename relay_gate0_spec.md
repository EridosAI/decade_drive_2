# Relay Program — Gate 0 Spec: Offline Two-Stage Square-Law Relay Probe

**Status**: Drafted (Claude, full-context instance). Ready for ratification + CC handoff.
**Date**: 2026-07-02
**Program**: NEW program (Relay), successor to Decade-Drive per its Terminus. Repo: fresh, in Oscillator_Reservoir_Program_2. This is a new decision from a clean baseline, not Decade-Drive reopened.
**Tests**: the shipped prediction of the Decade-Drive Phase-3 document (Section 6, Branch B): engineered staging crosses the ~2-decade passive routing horizon.
**Gate type**: cheap, offline, zero new core machinery. STOP-and-report; no sweep.

---

## Objective (one sentence)
Determine whether an offline two-stage square-law relay — stage 1's slow-band |z|^2 reconstruction, band-limited and re-injected as stage 2's message — transfers usable information across a compound 3-decade information path where direct passive transfer is dead (committed Phase-1 direct span-3.0: r2 = -0.003, ESP-robust @K=0.24).

## Honest framing (pre-empt the obvious objection)
The substrate is in natural units, so two chained span-1.5 networks are spectrally identical systems. "Compound span 3.0" is a claim about the INFORMATION PATH — the message survives two successive square-law demodulations end-to-end — not about one physical spectrum. The comparison target is therefore the DIRECT span-3.0 floor, re-run fresh in the same batch: the question is precisely whether staging beats what one passive 3-decade hop cannot do. State this framing in all outputs; do not let a pass be written as "we routed across 3 decades of one substrate."

## Design

**Stage A (replication anchor + hop 1).** Phase-1 replica: N=500, single continuous log-uniform comb, span 1.5, lambda=0.1, ER mean-degree 10, beta=1.0, SPP=2, K=0.24 (ESP-robust). Designed AM message m0 in MSG_BAND=[0.2,0.9] rad/s injected into the fast tertile only; slow-tertile-only readout x=[Re z, Im z, |z|^2]+bias, ridge+CV. Output: reconstruction m1. **Anchor requirement (numeric, pre-registered)**: anchor PASSES iff |stage-A intersection mean - 0.986| <= max(2*SE, 0.02). Rationale: a pure 2*SE rule tightens with n and would punish precision; the 0.02 floor is small against the span ladder (next rung 0.751) and keeps the window clear of 0.945 (the K=0.16 value at this span), so reproducing the wrong K row still fails. Diagnostic asymmetry: low-side miss = replication failure; high-side miss with elevated decoy = leakage. Either is NO-MEASUREMENT — STOP, fix, re-run before interpreting anything downstream.

**Repeater step (the architecture's detect-filter-remodulate, made explicit).** Band-limit m1 to MSG_BAND (this filter IS the repeater's F), then affine-rescale to match the m0 message-class statistics (zero-mean, matched RMS). Document the exact transform in the json. The band-limited, rescaled m1 is stage B's injected message.

**Stage B (hop 2).** Fresh network, independent seed, identical construction to stage A. Inject processed-m1 into ITS fast tertile; slow-tertile readout; output m2.

**End-to-end metric.** r2(m2, m0) — against the ORIGINAL message. Also report r2(m1,m0) and r2(m2, processed-m1) so the per-hop decomposition is legible.

**Direct baseline (fresh, paired).** Direct span-3.0 run: same construction, span 3.0, same m0 realizations, same seed set, K=0.24 (its ESP-robust floor per committed record), in the same batch. This — not the committed Phase-1 number — is the paired comparison target.

## Conditions (gate battery; K=0.24 primary throughout)
1. **Relay** (stage A -> repeater -> stage B), n>=5 seeds (paired: stage-A seed i feeds stage-B seed i; the two networks use independent seeds within pair i).
2. **Direct span-3.0** (fresh baseline), same seeds/m0.
3. **End-to-end decoy**: a same-spectrum never-injected decoy message m0' scored against m2 (and a full decoy pass: m0' fed through both stages' readouts trained on it — match Phase-1's decoy protocol exactly).
4. **Bandwidth-violation instrument check**: one relay condition with violation MSG_BAND = [2, 9] rad/s (10x standard), placing message content at/above the slow tertile's natural-frequency floor (omega_min = 1 at span 1.5) — a band the slow readout cannot envelope-track. Transfer MUST collapse relative to the compliant relay condition; if it does not, the envelope-of-envelope bookkeeping is wrong and the gate is no measurement. Log in the json: violation band, standard band, and the slow-tertile omega range — same auditable treatment as the repeater transform.
5. **Scramble on stage A** (one condition): degree-matched random coupling. Staging should be topology-generic if Phase-1's finding extends.
6. **Secondary K row (optional, same structure)**: relay + direct at K=0.16 to bracket operating K. Not required for the verdict.

## Statistics preconditions (named; carried from the Phase-2 lesson)
- **ESP-honest paired intersection**: a seed counts only if ESP-ok across every condition compared (both relay stages AND the direct baseline). All deltas computed on that intersection.
- **Finite-sd requirement**: >=2 seeds in the intersection or the gate is underpowered — add seeds, do not read a verdict.
- ESP gate (consistency.py, EPS=1e-2) per stage per seed; per-stage ESP flags recorded.
- Report per-seed values, mean +/- SE, and the intersection list in the json.

## Pre-registered outcomes (no post-hoc moving)
- **PASS (horizon is architectural)**: r2(m2,m0) beats BOTH the fresh direct span-3.0 value AND the end-to-end decoy floor by > seed sigma on the paired intersection. Naive expectation ~0.97 (0.986^2) at K=0.24; anything materially above the floors passes — do not quietly redefine success as hitting 0.97.
- **FAIL (horizon binds active staging)**: relay at or below floors with all instrument checks healthy. Report as the sharper negative: the loss in the lossy exchange is intrinsic at this operating point, not topological.
- **NO MEASUREMENT**: stage-A anchor misses 0.986; or violation-control fails to collapse; or decoy elevated (leakage); or intersection underpowered. Fix instrument, re-run. A null from a broken instrument is not a FAIL.

## Deliverables
- results/R/gate0_relay.json: per (condition, seed, stage) r2 values, ESP flags, the paired intersection, the repeater transform parameters, m0 spec.
- Short log: anchor check result, violation-control result, verdict mapping.
- STOP-and-report. No sweep, no multi-hop extension, no interpretation beyond the outcome mapping.

## Compute
Small: a handful of (condition x seed) Phase-1-scale points, not a sweep. Well under Phase-1's overnight on the 4080. One GPU process; float64; x64 on.

## Standing rules
Gate-first; STOP at every checkpoint; single-variable discipline (K fixed at 0.24 for the verdict row); probe bounded by reservoir (N=500, not reduced); verify load-bearing claims in sandbox before accepting numbers (specifically: repeater transform correctness, decoy protocol match to Phase-1, paired-intersection application); commit only on Jason's exact command, author Jason Dury <jason@eridos.ai>, no co-author line.

---
**End of spec.** If the gate PASSES, the natural next decisions (separately gated): multi-hop extension (3+ stages), online in-network repeater, and hop-length sweep. If it FAILS, the negative closes the staging question at this operating point and the interference-computation gate becomes the front of the queue.
