# Relay Gate-0 -- Spec Addendum: retire the rate-limit violation, re-pose to a repeater-filter check (PINNED, pending ratification)

**Status: PINNED for ratification -- fixed here BEFORE the re-run; nothing re-run or committed
until Jason's word.** Amends the bandwidth-violation control (spec condition 4) after the Stage-3
full gate returned **NO-MEASUREMENT**. Structure: **retire -> replace -> re-read.**

Provenance: sweep artifact `results/R/gate0_bandsweep.json`; Stage-3 NM record
`results/R/gate0_relay.{json,md}` (untouched); harness `experiments/relay_gate0.py` (uncommitted).
All numbers below are sourced from those artifacts + the CPU pre-check in this file.

---

## 1. The sweep rule, pinned BEFORE any result (band-shopping guard)

Fixed in source (constants + printed banner) before a single r2 was seen:

- swept band width **rho = 4.5** = the standard message width (compliant [0.2,0.9] and original
  violation [2,9] both have hi/lo = 4.5), so every swept band is a genuine candidate violation band
  of the same shape;
- **derived violation band = the LOWEST swept center whose stage-A r2 < 0.1 on ALL 3 seeds [0,1,2]**;
- representability: top band upper edge <= 0.95 x Nyquist (31.62 rad/s) -> c_max = 14.16 (a wide
  rho=4.5 band cannot be centered above ~14 without crossing Nyquist; reaching center 28 would
  require narrowing the band -- a different probe -- flagged, not silently done).

Measured curve (stage-A r2 of an AM message in each band; 3 seeds; K=0.24; self-validating anchors
first):

| band | [w_lo, w_hi] rad/s | center | r2 mean | overlaps fast? |
|---|---|---|---|---|
| compliant | [0.20, 0.90] | 0.42 | +0.986 | no |
| orig_violation | [2.00, 9.00] | 4.24 | +0.884 | no |
| c=2.85 | [1.34, 6.04] | 2.85 | +0.907 | no |
| c=5.41 | [2.55, 11.48] | 5.41 | +0.861 | yes |
| c=10.28 | [4.84, 21.80] | 10.28 | +0.758 | yes |
| c=14.16 | [6.68, 30.04] | 14.16 | +0.698 | yes |

Validated: the compliant anchor reproduced the committed span-1.5/K=0.24 seeds-0/1/2 mean (0.986),
and orig_violation reproduced the battery's violation stage-A exactly (0.884). **NO band reached the
0.1 cutoff on any seed** -- even the top band ([6.68, 30.04], against Nyquist, deep in the fast
tertile [10.1, 31.6]) tracks at r2 ~ 0.70.

## 2. Retirement rationale (rate-limit violation check = empirically void)

The Stage-3 NM was: original violation band [2,9] was *attenuated but tracked* (e2e 0.664 vs compliant
0.926; stage-A r2 ~0.79-0.92), so its premise -- "a band the slow readout cannot envelope-track" --
was false. The sweep shows this is not fixable by re-centering: **no representable standard-width band
is untrackable** (the slow |z|^2 ridge readout reconstructs the fast-band power envelope at any
sub-Nyquist message rate). Moving the band cannot fix a premise the substrate falsifies, so the
rate-limit violation check is **RETIRED**, with the sweep curve as the recorded reason. (Mechanism
caveat, honored: the modest droop only appears at centers >= 5.4, which already overlap the fast
tertile -- so even that droop is confounded with injection ill-posedness and still never collapses.)

## 3. Re-posed check (adopted): repeater-filter bookkeeping

The retired check tried to test the READOUT's envelope limit (no such limit exists in range). The
re-posed check tests the one part of the envelope-of-envelope machinery that CAN fail: the
**repeater's band-limit + rescale bookkeeping**.

**Definition.** Message band = **[2,9] rad/s** (stage A provably tracks it, r2 ~0.88, per the sweep --
so upstream physics is NOT the cause of any collapse). Repeater pass-band = the **standard [0.2,0.9]**,
deliberately MISMATCHED to the message.

**Mechanism (pinned).** m1 carries the [2,9] message; the brick-wall filter keeps only its [0.2,0.9]
content = the reconstruction RESIDUAL (~0); the affine rescale amplifies that residual to full message
RMS; stage B therefore receives an m0-UNCORRELATED signal -> **e2e r2(m2, m0) ~ 0**. The isolation is
the point: a collapse here can only be the repeater filter/rescale (the bookkeeping under test), not
upstream physics.

**Signature the check worked for the stated reason (logged per seed):** small **rms_in**, large
**scale** (a big amplification of a tiny residual). CPU pre-check (pure-[2,9]-message proxy, seeds
0/1/2): rms_in 1.2e-3 / 1.8e-4 / 1.5e-3 vs rms_target ~0.14 -> **scale 120 / 794 / 93**. The real m1
reconstruction residual (measured at the GPU re-run) is somewhat larger but the signature holds.

## 4. Numeric collapse bar (pinned)

**Collapse iff e2e r2 < 0.1 on EVERY intersection seed** (per-seed absolute; no ratio ambiguity). If
any intersection seed has e2e r2 >= 0.1, the re-posed control did NOT collapse -> the repeater-filter/
rescale bookkeeping is wrong -> NO-MEASUREMENT. (Supersedes the retired ratio bar entirely.)

## 5. Re-run + re-read plan (on ratification)

- Re-run the RE-POSED violation arm ONLY (`--violation-rerun`, ~10 min GPU): loads the Stage-3 NM
  record for every settled arm, re-runs the re-posed violation for all seeds, re-decides.
- **New artifact `results/R/gate0_relay_reposed.{json,md}`; the Stage-3 NM record
  `gate0_relay.{json,md}` stays UNTOUCHED.**
- Re-read the gate on the UNCHANGED PASS/FAIL/NO-MEASUREMENT mapping. Anchor (0.986421) and decoys
  (clean) are already green and the intersection is powered (n=7), so **the quarantine on the
  relay-vs-direct verdict lifts IFF the re-posed control collapses AND the verdict is PASS.**
- **All commits together at resolution** -- no commit until the re-posed control resolves.

## 6. Kept, not dropped: the decoy control

Retiring the rate-limit check does NOT remove a guarantee. The decoy control already guards
specificity/leakage on its own NM clause (Stage-3 e2e decoy p95 -0.424 vs relay 0.926 -- a
never-injected message cannot be reconstructed); the re-posed check guards the filter/scoring
bookkeeping. Different failure modes -- both retained.

## 7. Side-finding (noted; not acted on)

The sweep incidentally shows the slow |z|^2 ridge readout is NOT envelope-rate-limited within the
representable range -- it reconstructs power-envelope structure at any sub-Nyquist message rate. This
gently weakens the Phase-3 doc's "envelope-of-envelope rate cost" assumption. Phase-3 is committed and
closed; this is recorded only. Whether it ever merits a decade_drive erratum is Jason's call, later.

## 8. Scope

Instrument retirement + replacement + a proposed re-run. No relay verdict is read or moved here; the
Stage-3 relay-vs-direct numbers remain QUARANTINED until the re-posed violation resolves (caught !=
fixed).
