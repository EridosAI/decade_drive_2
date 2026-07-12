# Relay Program -- Gate 4 Spec: Hop-Length Trade (matched total span)

Status: Drafted (Claude). Ready for ratification + CC handoff.
Date: 2026-07-10
Builds on: Gate-3 CLOSED d1a7116 (channel answer: hops price against the
|z|^2 observable-order channel, all bands) + Gate-1 (one-time insertion
loss, near-flat) + Gate-2 (steady per-hop decline at depth) + the
committed Phase-1 (span x K) landscape at decade_drive b0f7664.
Independent of Gate-B stage-2 (which stays on its own go).
Gate type: measurement gate, offline decode-and-forward chains
(committed Gate-1/2 mechanics). STOP-and-report.

## Objective (one sentence)
Price the hop-length trade: at matched total span, does end-to-end
fidelity favor many short hops or few long hops -- quantifying
r2_e2e(hop span s | total S) for the |z|^2 channel Gate-3 pinned.

## Design
Message: standard SUB [0.2, 0.9] m0, Phase-1 protocol; stage n+1's
message is stage n's decoded shat (committed offline relay mechanics).
Configs (H hops x span s, matched H*s = S_total):
  PRIMARY   S=3.0: (1 x 3.0), (2 x 1.5), (3 x 1.0)
  SECONDARY S=2.0: (1 x 2.0), (2 x 1.0)   [consistency set, not gating]
K per span by pre-registered lookup, NO tuning: K*(s) = argmax_K of the
committed Phase-1 mean r2_d0 over seeds 0-9 at that span (b0f7664
landscape). CC prints the resolved K*(s) table at sandbox, before any
GPU. At s=3.0 the argmax is over noise (all K rows ~ 0); it resolves
mechanically and the config is a floor endpoint regardless -- flagged.
Anchor-arm contingency: if K*(1.5) != 0.24, BOTH (2 x 1.5) arms run --
K=0.24 (anchor arm) and K*(1.5) (trade arm); resolved at sandbox.
Seeds 0-7, extend 8-9 on attrition. One GPU process, float64/x64,
Oscillator_Reservoir_Program venv.

## Pre-registered readout (no post-hoc moving)
Primary statistic: e2e r2_d0 (final-stage decode vs m0) per
(config, seed).
ESP-honest pairing (Phase-2 lesson, applied): the PAIRED CONTRAST uses
only seeds ESP-ok in every stage of BOTH contrast configs; the per-config
pricing table uses per-config intersections; both memberships printed.
Primary contrast: D = r2(3x1.0) - r2(2x1.5), per-seed paired,
mean +/- SE_paired (labeled).
Classification (pinned): SHORT-WINS iff D >= delta; LONG-WINS iff
D <= -delta; FLAT otherwise; delta = max(2*SE_paired, 0.02) --
EVALUATED AT VERDICT from byte-locked primitives, never stored
(standing rule).
(1 x 3.0) = committed floor endpoint (Phase-1 (3.0, K*) row): context,
not part of the contrast. SECONDARY S=2.0 contrast reported under the
same rule, labeled consistency-only.
Decomposition (reported, NOT gating): per-hop insertion vs slope via
the committed Gate-2 m0-referenced machinery.
NO-MEASUREMENT: any anchor miss; any per-stage decoy elevated (bar
0.2); paired intersection < 2; Gate-1 replay mismatch.

## Instruments
- Anchors (digit-exact): hop-1 of the K=0.24 (2 x 1.5) arm vs committed
  Phase-1 REF (seed-0 0.981470 etc.); hop-1 of every other (s, K*(s))
  vs the committed Phase-1 per-seed cells; the full K=0.24 (2 x 1.5)
  chain replays Gate-1's committed first-two-hop cells digit-exact
  where construction permits (chain-mechanics anchor; CC verifies
  feasibility at sandbox).
- Per-stage decoys: never-injected same-class decoy per stage, Phase-1
  protocol, fresh bases; collision matrix proven against ALL committed
  families (Phase-1, Gates 0-3, Gate-B) before running.
- ESP nested ok_slow per stage.

## Deliverables
results/R/gate4_hoptrade.json (per config x seed x stage r2, e2e,
decoys, ESP, K* table + lookup provenance, pairing memberships, paired
contrasts, classification, env, sha chain) + short .md (anchors, decoy
max cell, pricing table, contrast, classification, decomposition).
STOP-and-report.

## Compute
Stage integrations: 8 seeds x (6 + 3) = 72 (+16 if the dual (2 x 1.5)
arm resolves in): ~17-22 min on the 4080. Fits CPU-cheap.

## Standing rules
Gate-first: CPU sandbox (chain mechanics vs Gate-1 digit-exact where
construction permits; K* lookup proven against the committed landscape;
collision matrix; classifier + NM branches on synthetic matrices; the
INSTITUTIONALIZED amend-path test -- reread fails loudly on any numeric
drift; derived thresholds never stored, evaluate-at-use), verdict-engine
test, 1-seed smoke (all hop-1 anchors digit-exact + one full chain per
config logged), battery on Jason's go, adversarial panel, pre-commit
read-through, commit only on Jason's exact word, author Jason Dury
<jason@eridos.ai>, no co-author. Committed artifacts and core/
untouched. ASCII throughout.

## Ratification edits (2026-07-10; merged verbatim pre-ratification)

1. CONTRAST-ARM PIN: the PRIMARY contrast always uses the K*(s) trade
   arms. If K*(1.5) = 0.24 the (2 x 1.5) trade and anchor arms are the
   same runs. If K*(1.5) != 0.24, the K=0.24 arm runs INSTRUMENT-ONLY
   (Phase-1 anchor + Gate-1 replay); its e2e prints in the .md labeled
   INSTRUMENT-ARM, outside the contrast.

2. REPLAY-ANCHOR DISPOSITION: "where construction permits" resolves at
   sandbox, pre-GPU, one of two ways. (i) FEASIBLE -- certified at
   sandbox; the battery replay binds; feasible-but-different = NM.
   (ii) INFEASIBLE-BY-CONSTRUCTION -- declared at sandbox with the
   specific structural reason; the anchor narrows; CC substitutes the
   strongest construction-permitted chain-mechanics check against
   Gate-1's committed recs; the smoke waits on Jason's explicit
   acknowledgment of the narrowed instrument. Infeasibility can never
   be declared after any Gate-4 number exists. Structural enforcement
   includes the time axis: the sandbox infeasibility-declaration path
   hard-fails if any Gate-4 smoke or battery record exists on disk
   ("declaration window closed").

3. UNDERPOWERED SUFFIX (symmetric): if the paired intersection n is
   2-4, ANY classification prints with the -UNDERPOWERED suffix
   (SHORT-WINS-UNDERPOWERED, LONG-WINS-UNDERPOWERED,
   FLAT-UNDERPOWERED). Basis, computed exactly: at 1-3 dof the 2*SE
   bar's null exceedance is 0.295 / 0.184 / 0.139, so tiny-n wins are
   not self-certifying and no direction gets a full-strength verdict
   below the house power target. Full-strength verdicts require
   paired n >= 5 (pre-registered primitive, storable). Paired n < 2
   remains NM. delta itself is unchanged: max(2*SE_paired, 0.02),
   evaluate-at-use. Verdict space: {SHORT-WINS, LONG-WINS, FLAT,
   each with optional -UNDERPOWERED, NM}; the verdict-engine test
   covers all three suffixed branches synthetically.

4. ACCOUNTING PINNED: S_total = H * s -- codifies Gate-1's committed
   compound-span framing (1.5 * H); no committed alternative exists.

5. K-RULE UNIFORMITY: the argmax lookup runs identically at every
   span, K=0.0 included; if K*(3.0) resolves into the noise floor
   (possibly null coupling), the .md flags it as a noise-argmax and
   the (1 x 3.0) endpoint remains context-only. No special-casing.

## Verdict-scope addendum (ratified 2026-07-11, pre-smoke; from the resolved K* table)

Verdict-scope addendum (ratify pre-smoke): K*(s) resolved to the grid
maximum at every span (AT-GRID-MAX; right-censored -- spans 1.0-2.0
monotone in K to the edge). The hop-length classification is scoped
to couplings within the committed landscape (K <= 0.24); behavior
above that ceiling is untested and out of scope for this gate.

## (1x3.0) sourcing -- Option B (ratified 2026-07-11, pre-battery)

(1x3.0) SOURCING (Option B): the context row's per-seed values are
SOURCED from the committed phase1_routing.json cells (3.0, 0.24,
seed), decade_drive b0f7664, sha 2e739315...; bridge = the smoke's
seed-0 bit-exact reproduction (diff 0.0e+00). The row inherits the
committed ESP ok_slow flags (note: seed 7 is ESP-fail in the
committed record) and cites the committed decoy_p95 (base-40000
protocol); Gate-4's fresh-base decoy requirement applies to GPU-run
configs only. The row is labeled SOURCED-FROM-COMMITTED in json and
.md and is never presented as Gate-4 GPU output. Context-only
status, NOISE-ARGMAX flag, and exclusion from all contrasts
unchanged.
