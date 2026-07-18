# Relay Program -- Gate L Spec: Coherent Linear Injection (the other door)

Status: v2. Drafted 2026-07-12 (Claude); four taste calls RATIFIED by
Jason 2026-07-17 (see Ratification record); adversarial panel run
2026-07-17, edits P1-P4 merged below, PENDING Jason's confirm word.
CC handoff releases on that word.
Builds on: Gate-3 CLOSED d1a7116 -- its SCOPED verdict is this gate's
reason to exist: the AM injection supplied zero linear message content
(u^2 = 0.25*s exact; |corr(u,s)| <= 0.032), so Gate-3 tested only
reservoir-mediated re-encoding into first-order coordinates and found
that door shut. It explicitly did NOT refute a linear resonant channel.
Gate-L opens the untested door: supply the linear content directly.
Also consumes: committed Gate-3 per-seed LIN baselines (the paired
subtrahend); Gate-B pattern-fold (all-positive holdout residuals,
observed-not-claimed). The Phase-3 line-67 erratum WAITS on this gate
by pre-registration.
Gate type: cheap measurement gate, stage-A only, no chains.
STOP-and-report.

## Objective (one sentence)
Determine whether a coherent linear transmission channel exists -- do
slow-tertile first-order coordinates (Re z, Im z) carry the message when
the injection supplies linear message content by construction
(corr(u,s) = 1) -- closing the branch Gate-3's scope note left open, per
message band.

## Design
Stage-A Phase-1 replica (N=500, span 1.5, K=0.24, lambda=0.1, ER-10,
beta=1.0, SPP=2), message injected fast-tertile only, slow-tertile
readout -- byte-matched to the committed Gate-3 protocol in EVERYTHING
except the injection map. Same bands (SUB [0.2,0.9], RES [2,9], SUPRA
[10,28]), same seeds 0-7 (extend 8-9 on attrition), same build/enc/rep/
msg seed families -- so each (band, seed) cell reuses the IDENTICAL
reservoir and the IDENTICAL message realization s as the committed AM
cell. The contrast is then a controlled experiment on the injection map
alone.

Injection map (the one change): COHERENT, ZERO-MEAN [P1]:
  u = a * (s - mean_inj(s)),   a = 0.5 * sqrt(E_inj[s] / Var_inj(s))
(subscript inj = statistics over the injection window).
- No Rademacher carrier (the committed AM map is u = 0.5*sqrt(s)*w; the
  carrier is what destroyed input-level linear content). The carrier
  seed family (base 1777) is UNUSED in coherent cells; the sandbox
  seed-consumption audit proves it is never drawn.
- Zero-mean rationale [P1]: the committed AM drive is zero-mean
  (Rademacher); an uncentered coherent map u = a*s would spend the
  DC fraction E[s]^2/E[s^2] of the matched power budget on a static
  offset carrying no message content, making a DEAD verdict
  attackable on message-band power grounds. Centering removes the
  confound categorically; Pearson corr(u, s) = 1 exactly under
  centering, so the reachability gate is unchanged. All injected
  power now lives in the message band -- the maximally fair coherent
  test at matched power.
- Power matched to the committed AM scheme BY EXPRESSION:
  E[u^2] = a^2 * Var_inj(s) = 0.25 * E_inj[s], the same injected
  power the AM map delivers by construction. Never stored as a
  literal; evaluated at run; arithmetic identity asserted per cell:
  |E[u^2] - 0.25*E_inj[s]| <= 1e-9 * 0.25*E_inj[s].
  The realized DC fraction E[s]^2/E[s^2] is recorded per cell
  (report-only context) [P1].
- Band caveats (recorded, not gated): under coherent injection the
  drive spectrum IS the message band. SUPRA [10,28] then sits inside
  the fast-tertile natural range [10.1,31.6] (injection-RESONANT --
  maximally favorable for exciting first-order response; the fair test
  of the linear-channel hypothesis, noted so a live SUPRA is read
  correctly). RES [2,9] partially overlaps the slow-tertile natural
  range [1.00,3.13] (readout-resonant edge). SUB is far from both.

Arms and run set:
  COHERENT   3 bands x 8 seeds = 24 stage-A runs (the gate).
  ANCHOR     AM x SUB, 8 seeds = 8 runs: exact Phase-1/Gate-3 replica
             through the same import spine; per-seed digit-exact vs
             REF_TABLE (instrument continuity, handoff item (d)).
  LADDER     seed 0 x 3 bands x 3 rungs {0.5a, a, 2a} = 9 runs
             (null-interrogation instrument; see Instruments) [R3:
             seed-0 only, ratified].
  NO-INJ     u = 0, seed 0, 1 run (band-independent chance baseline
             for the ladder's tracking statistic [P3]).
Total 42 stage-A integrations; all fits CPU-cheap on stored features.

## Why the Gate-3 classifier does NOT port (pre-data fence)
With z = Z0 + dz (Z0 the drive-free trajectory, dz the drive-induced
response), |z|^2 = |Z0|^2 + 2*Re(conj(Z0)*dz) + |dz|^2. Under coherent
injection dz is first-order in the drive, hence in s -- the beat term
2*Re(conj(Z0)*dz) makes |z|^2 carry LINEAR-in-drive content. So a high
SQ r2 here does NOT indicate a quadratic mechanism, and R_SQ/R_LIN
retention semantics are ambiguous by construction. FENCED PRE-DATA:
no mechanism claim, retention ratio, or channel classification is
derived from the SQ or FULL columns in this gate. Both are computed
and recorded (continuity + follow-on design only). All falsifiable
content lives in the LIN column, which stays unambiguous: Re z, Im z
are first-order observables under any injection.

## Pre-registered readout (no post-hoc moving)
Primary statistic: r2_LIN_coh per (band, seed) -- the LIN [Re z, Im z]
+bias ridge fit, committed Gate-3 fit protocol, fit-time ablation.

Primary contrast (handoff item (b)), per band, per-seed paired:
  D_LIN(band, seed) = r2_LIN_coh(band, seed) - r2_LIN_AM(band, seed)
where the subtrahend is BY-COMMITTED-EXPRESSION from
results/R/gate3_mechanism.json (sha256[:12] 878c154850c7, at d1a7116):
recs["BAND|seed"].r2.LIN. Read at use from the committed file with a
sha assert; never copied into the harness. ESP-honest SYMMETRIC
pairing: seeds ESP-ok in this battery AND in the committed Gate-3
record (committed: all of 0-7 ok). Both memberships printed.
delta = max(2*SE_paired, 0.02), EVALUATED AT VERDICT, never stored.
Median-D per band reported alongside mean-D, REPORT-ONLY [P2]: the
committed subtrahend contains one outlier (SUB seed 1 = -0.4367 vs
|others| <= ~0.04) that distorts a mean without threatening the
classification; the median column keeps the .md honest.

Classification per band (pinned; precedence order as listed):
1. LIN-DEAD  iff intersection-mean r2_LIN_coh <= the band's max
   per-cell fresh LIN decoy p95 (indistinguishable from never-
   injected, the Gate-3 death criterion applied symmetrically)
   [R2: conservatism direction ratified -- harder to declare DEAD
   than Gate-3's 24/24 per-cell rule; biases toward INTERMEDIATE
   over a false death certificate].
2. LIN-LIVE  iff D_LIN > delta AND intersection-mean r2_LIN_coh > 0.2.
   BOTH legs required. The D leg alone is a known trap: the committed
   AM baselines are dead-negative (band means -0.056/-0.043/-0.021),
   so a dead-vs-dead comparison can yield D ~ +0.05 > delta from two
   nulls. The 0.2 absolute floor (reuse of the committed full_r2_min /
   decoy-bar primitive, committed lineage) is what makes LIVE mean
   alive; with the global decoy gate clean it also sits above every
   fresh null by construction.
3. INTERMEDIATE otherwise -- report as-is with both quantities;
   no consequence-map branch fires beyond report.
-UNDERPOWERED suffix symmetric at paired n in [2,4]; full strength
needs n >= 5; n < 2 is NM (house rule, unchanged). Per-seed D signs
reported per band (pattern-fold inheritance from Gate-B: observed,
not claimed). LOSO reported (report-only) for any LIVE band.

Pre-registered consequence map:
- ALL THREE LIN-DEAD -> COHERENT-LIN-SHUT: even with corr(u,s)=1 at
  the input, first-order slow-tertile coordinates do not carry the
  message at any tested band; at this operating point the square-law
  observable is the only demonstrated carrier, now tested from BOTH
  injection sides. Gate-3's scope note RESOLVES shut-side; C3
  unmodified.
- ANY band LIN-LIVE -> COHERENT-LIN-LIVE (band-resolved): a coherent
  linear channel exists where injected coherently; C3 gains the scope
  note Gate-3's consequence map anticipated ("|z|^2 is the power-
  envelope channel; coherent first-order transmission also exists at
  [bands]").
- Any other pattern: report the matrix as-is, no new claim.
ERRATUM UNBLOCK (pre-registered): the Phase-3 line-67 erratum's WAIT
lifts when this gate lands, WHICHEVER class fires; the landing class
selects the replacement wording (exclusive-square-law vs dual-channel
vs pattern-specific). Drafting/filing in decade_drive remains Jason's
separate call, unchanged from the Gate-3 consequence map.

NO-MEASUREMENT branches: anchor miss (any per-seed digit mismatch or
mean-window fail); any fresh decoy cell elevated (bar 0.2); any
reachability-floor failure (see Instruments -- tripwire semantics,
gate-level NM) [R1: tripwire ratified]; ladder gate fail (band-level
NM); rung-a ladder run fails to complete (band-level NM); paired
intersection < 2 (band-level NM). NM SEAL applies (code-enforced,
inherited _write_md NM-shape self-test).

## Instruments (pre-registered order: anchor, reachability, ladder, decoys, ESP)
- ANCHOR (item (d)): AM x SUB per-seed digit-exact vs REF_TABLE (6dp
  house standard); mean rule |mean - 0.986| <= max(2*SE, 0.02).
  Certifies the shared import spine end-to-end. Excluded from the
  fresh-decoy gate (its cells are committed replicas carrying
  committed decoy_p95 -- Gate-4 (1x3.0) precedent).
- REACHABILITY AUDIT (item (a)), per coherent cell, both arms:
  (i) linear arm: |corr(u, s)| recorded; GATE >= 0.99 [R4: 0.99
  literal ratified]. Convention pinned [P4]: the gate statistic is
  the MAX over the committed Gate-3 audit's lag set, computed on the
  eval window with the imported audit machinery (import, not
  reimplement). u = a*(s - mean) cannot legitimately miss this floor,
  so ANY failure implies a harness fault, not physics: tripwire
  semantics, gate-level NM [R1].
  (ii) quadratic arm: corr(u^2, s) recorded per cell, no gate.
  NOTE [P1]: under the zero-mean map u^2 = a^2*(s - mean)^2 is a
  folded encoding of s, so this value is expected REDUCED relative
  to an uncentered map; it is recorded as measured. Quadratic-pathway
  liveness is already certified by the committed Gate-3 AM arm and is
  not this gate's question; the record exists for the audit trail.
- LINEARITY LADDER (item (c)) -- the blind-instrument interrogation
  for an expected null. Per band, seed 0 [R3], rungs {0.5a, a, 2a}.
  Statistic [P3 -- replaces raw in-band power, which is band-
  inequitable: at SUPRA the fast tertile's own limit-cycle lines lie
  in-band (natural range [10.1,31.6]), so a raw-power floor compares
  drive response against carrier power and near-certainly fails
  spuriously]:
    For each fast-tertile oscillator i, over the eval window, with
    x_i = Re z_i band-passed to the message band (committed band-pass
    machinery, imported) and mean-removed, and u the injected drive:
      beta_i = <x_i, u> / <u, u>
      P_track(rung) = mean_i beta_i^2 * <u, u>
    CHANCE level = the identical computation applied to the NO-INJ
    (u = 0) trajectory against the same rung-a u waveform, per band.
  GATE per band: P_track strictly increasing over completed rungs AND
  P_track(a) >= 10 * P_track_chance [R4: 10x literal ratified, now
  applied to the chance-normalized tracking statistic]. Passing
  certifies the drive demonstrably enters the system's first-order
  coordinates at the injection tertile -- so a LIN-DEAD verdict reads
  as "no cross-tertile first-order transfer" (the finding), not
  "drive never arrived" (instrument failure). Fail -> band-level NM.
  Rung 2a integration failure is tolerated if 0.5a and a complete and
  the gate holds on those two; rung a must complete. P_slow analog
  (same statistic on the slow tertile) is report-only context for the
  verdict prose.
- FRESH DECOYS: committed protocol (60 never-injected same-class decoy
  messages per (band, seed, mode) cell, per-cell p95, gate = max over
  all coherent cells, bar 0.2). NEW base family: SUB 600000 /
  RES 620000 / SUPRA 640000; CC proves the collision matrix against
  ALL committed families (Phase-1, Gates 0-4, Gate-B/probe) at
  sandbox, before any GPU.
- ESP nested ok_slow per seed; symmetric intersection as above;
  per-seed values and mean +/- SE labeled in json and .md regardless
  of class.

## Deliverables
experiments/relay_gateL.py -- COMPANION harness, reuse-by-import of the
committed relay_gate0/relay_gate3 machinery; committed artifacts and
core/ untouched. results/R/gateL_coherent.json (per band x mode x seed
r2; reachability audit per cell; amplitude identity residuals + DC
fractions; ladder P_track table + chance floor; fresh decoys; ESP
flags + memberships; paired per-seed D_LIN + mean/median contrast
stats; classification; consequence line; env; sha chain incl. the
gate3_mechanism.json read-at-use assert) + short .md (anchor,
reachability summary, ladder, decoy max cell, LIN matrix with
committed baselines alongside, per-band classification, fence
restated, consequence-map line). ASCII throughout. STOP-and-report.

## Compute
42 stage-A integrations (24 coherent + 8 anchor + 9 ladder + 1 no-inj);
Gate-3's 24 ran in ~5 min wall on the 4080, so ~9-14 min GPU total;
decoy/ridge/projection fits CPU-cheap on stored features. One GPU
process, float64/x64, Oscillator_Reservoir_Program venv.

## Standing rules
Gate-first: CPU sandbox proving, with every check DEMONSTRATED TO FIRE
on a synthetic violation (Caught != Fixed; assume any untested check is
hollow):
  (1) injection unit test -- zero-mean coherent map correctness,
      corr(u,s) >= 0.99 on synthetic, power-match identity
      (E[u^2] = 0.25*E_inj[s] under the [P1] map), DC-fraction
      recording, and the seed-consumption audit showing the
      Rademacher family is never drawn in coherent cells;
  (2) reachability-floor branch fires on a synthetic sub-floor cell;
  (3) ladder P_track branches fire on synthetic fail cases and stay
      silent on synthetic healthy; REQUIRED PAIR [P3]: (i) tracking-
      absent synthetic -> gate fails; (ii) tracking-present synthetic
      with dominant in-band natural background -> the superseded
      raw-power check would fail spuriously, the P_track check
      passes;
  (4) decoy collision matrix vs ALL committed families;
  (5) classifier + precedence + -UNDERPOWERED + every NM branch on
      synthetic matrices (all branches fired);
  (6) subtrahend read-at-use -- gate3_mechanism.json sha assert wired,
      fails loudly on a byte-perturbed copy;
  (7) LOCKED-NUMBERS loud-fail reread test (inherited, must fire);
  (8) NM SEAL shape self-test (inherited);
  (9) no-truncation render assert (fires on synthetic truncation,
      silent on all healthy committed .mds);
  (10) delta and all windows evaluate-at-use -- nothing derived is
       stored.
Then verdict-engine test, then 1-seed smoke (anchor AM x SUB seed-0
digit-exact 0.981470; one coherent SUB run completing with
reachability values logged; ladder rung a at SUB), STOP; battery on
Jason's go; adversarial panel; pre-commit read-through; commit only on
Jason's exact word, author Jason Dury <jason@eridos.ai>, no co-author
lines. First-eyes: no masked lookups exist in this gate (the committed
subtrahend is public by construction); pre-registration is the
protection.

## Scope (pre-declared)
Readout-level observable-order question, symmetric to Gate-3's framing:
this gate establishes whether FIRST-ORDER SLOW-TERTILE COORDINATES
carry the message under coherent injection -- NOT the internal transfer
path, which the Stuart-Landau nonlinearity (|z|^2 z) mixes across
orders en route. A LIVE result licenses "a coherent linear channel
exists at the readout"; path attribution inside the medium is out of
scope. One operating point (K=0.24, span 1.5), stage-A, offline, no
chains. Drift-attribution remains out of scope and parked.

## Ratification record (2026-07-17; Jason's word, merged)
R1. Reachability-floor semantics: TRIPWIRE -- any sub-floor cell is
    gate-level NM (a miss implies harness fault; attrition would let
    a broken harness quietly shed cells).
R2. LIN-DEAD criterion as drafted: band intersection-mean vs max
    fresh per-cell null; conservatism direction accepted (harder to
    declare DEAD than Gate-3's 24/24; biases toward INTERMEDIATE).
R3. Ladder: seed-0 only (the instrument certifies a seed-generic
    code path, not a statistic).
R4. Fresh literals ADOPTED: 0.99 reachability floor; 10x ladder
    floor (per P3, applied to the chance-normalized P_track).

## Panel edits (2026-07-17; merged above at [P#] markers; PENDING Jason's confirm)
P1 (MAJOR). Injection map made ZERO-MEAN: u = a*(s - mean_inj(s)),
    a = 0.5*sqrt(E_inj[s]/Var_inj(s)); identity assert updated; DC
    fraction recorded; quadratic-arm audit note superseded
    accordingly. Trace: committed AM map is zero-mean by Rademacher
    (d1a7116); the uncentered draft spent the DC fraction of matched
    power on a static offset, exposing a DEAD verdict to a
    message-band-power objection.
P2 (minor). Median-D reported alongside mean-D, report-only.
    Trace: committed subtrahend outlier SUB|1 r2.LIN = -0.4367
    (gate3_mechanism.json, 878c154850c7).
P3 (MAJOR). Ladder statistic replaced: raw in-band power P_fast ->
    drive-tracking projection P_track with chance floor from the
    NO-INJ trajectory; monotone + 10x gate unchanged in form.
    Trace: SUPRA [10,28] lies inside the committed fast-tertile
    natural range [10.1,31.6] (d1a7116 framing); a raw-power floor
    there compares drive response against the oscillators' own
    carrier power and fails spuriously.
P4 (minor). Reachability corr convention pinned: max over the
    committed Gate-3 audit lag set, eval window, imported machinery.

P4' (2026-07-17, declared at sandbox pre-GPU; Gate-4 edit-2
precedent). P4's import-referent is INFEASIBLE-BY-CONSTRUCTION:
no committed corr-audit function exists; Gate-3's
|corr(u,s)| <= 0.032 is a prose literal in _write_md
(relay_gate3.py:507), not code. Substitution, verified by the
review seat from the committed record: gate statistic =
max over lags [0,32,64,96] of |Pearson corr(u, roll(s,k))| on
the committed eval window sl; the lag set is BY-COMMITTED-
EXPRESSION from am_window at span 1.5 (N_DELAYS=4, stride=32
via dt_in = 2*pi/(omega_max*SPP), SPP=2). Fresh implementation,
disclosed; recorded verbatim in gateL_sandbox.json. Tripwire
semantics unchanged. Declaration window closed at first GPU
record.

---
End of spec. Outcome feeds: the C3 scope-note resolution (either side),
the Phase-3 line-67 erratum wording (unblocked either way; filing is
Jason's call), and any Gate-L follow-on on path attribution if LIVE.
