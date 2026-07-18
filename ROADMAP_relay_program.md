# ROADMAP -- Relay Program forward path (post-Gate-L)

Status: NON-NORMATIVE navigation document. Nothing here is a source
for any number, threshold, or verdict; specs and commits are the
record. Every future gate still requires its own ratified spec.
Date: 2026-07-12. Drafted by the review seat (Claude, Fable) at
Jason's request, anticipating a capability change in this seat.
Disposition: Jason's call; default is alongside HANDOFF_* (gitignored,
not part of the public record).

## 0. Why this document exists, and how to read it
The program was built so that the machinery -- not the seat -- holds
the verdicts. What DOES live in the seat is design reasoning: branch
logic, gate sequencing, the shape of not-yet-specced gates. This
document banks that reasoning now, so the path forward does not
depend on the seat that drew it.
Rules of use: (a) every number below carries commit provenance and
was re-verified THIS SESSION from a fresh clone of decade_drive_2 at
origin/master 39560d8, except where marked attested; (b) blocks
labeled NON-NORMATIVE are design-seat priors and sketches -- they
bind nothing and must never be cited as findings; (c) where this
document and a ratified spec ever disagree, the spec wins without
discussion.

## 1. Position (one screen)
Committed and pushed through 39560d8. Ledger, scopes attached:
- G0 8361553: staged square-law relay crosses the ~2-decade passive
  horizon (PASS).
- G1 7d6f3f2: multi-hop loss law budget-like, TWO-REGIME; depth-limit
  refuted (offline H=5).
- G2 b49834f: steady decline at depth; late drift REAL (H=10).
- G3 d1a7116: SQ-carried all bands; C3 STANDS -- SCOPED (AM injection
  carries zero linear content; only reservoir re-encoding was tested;
  NOT a refutation of a linear resonant channel).
- GB c5a46e0 + probe 8794fe7: smooth attenuation, no cliff, no rate
  limit; M1 generic power law (A=29.86, p=1.043, 2sd [0.34,1.74],
  wRMSE 0.00860) beats one-pole 2.3x and cliff 5.27x; the LAW stays
  OPEN; locked P1 survived the blind probe (PREDICTIVE PASS, all
  three deviations positive).
- G4 39560d8: SHORT-WINS, D=+0.0564+/-0.0200, paired n=8, all seeds
  positive; SCOPED AT-CEILING right-censored (K<=0.24) and
  ARCHITECTURE-CONDITIONAL (offline decode-and-forward only).
- GATE-L: spec drafted 2026-07-12 (relay_gateL_coherent_spec.md,
  unratified); four taste calls open (tripwire-NM semantics on the
  reachability floor; DEAD-criterion conservatism direction; ladder
  seed count; the two fresh literal floors 0.99 and 10x).

## 2. Immediate path: closing Gate-L
Lifecycle, unchanged: adversarial panel on the draft -> Jason's
ratification (resolving the four taste calls; edits merged verbatim
as ratification edits, Gate-4 precedent) -> CC sandbox (every check
proven to fire) -> 1-seed smoke -> STOP -> battery on Jason's go ->
panel -> read-through -> commit word -> push word -> post-push trace.
Nothing below this line is reachable until Gate-L lands. The gate is
deliberately cheap (~42 stage-A integrations, ~9-14 min GPU) and its
design is finished; it is executable under any competent seat.

## 3. Gate-L branch pre-work (the decision tree, reasoned now)

### 3a. SHUT (all three bands LIN-DEAD)
Meaning: at K=0.24 / span 1.5 / stage-A, first-order slow-tertiary
coordinates do not carry the message even when handed coherent linear
content at matched power. Combined with G3, the square-law observable
is the demonstrated carrier from BOTH injection sides; C3 fully
settled at this operating point. The known residual scope is
"one operating point" -- carry it as scope language, not a queue item,
unless external review demands a sweep.
Consequences: erratum E1 unblocks with EXCLUSIVE-SQUARE-LAW wording;
the mechanism thread closes; queue follows Section 5 default.

### 3b. LIVE (any band)
Meaning: a coherent linear channel exists where injected coherently.
Much of the characterization is FREE from Gate-L's own matrix: which
bands, how strong (near-FULL vs barely over 0.2), the D-sign pattern.
Consequences: (i) E1 unblocks with DUAL-CHANNEL wording; (ii) C3
gains the fuller scope note pre-registered in the Gate-L spec;
(iii) a NEW architecture question activates -- Gate-4's scope
explicitly excluded coherent/amplify-and-forward relaying, so a live
linear channel makes Gate-C (Section 7b) scientifically motivated;
(iv) a cheap Gate-B analog for the LINEAR channel's attenuation
becomes natural. The GC-vs-GM priority is a Jason taste call the
roadmap does not make (Section 5).

### 3c. MIXED / INTERMEDIATE
Report the matrix as-is (the spec's class does this). Before ANY
program-level conclusion, run GL2 (Section 7c): a targeted power
extension at the intermediate band(s) only -- mechanical, own
pre-registration, fresh decoy family. Do not let an intermediate
band be argued into either camp by prose.

### NON-NORMATIVE expectations block (design-seat priors; bind nothing)
- SUB plausibly LIVE: a slow coherent drive is quasi-static
  first-order forcing (limit-cycle center wobble), directly
  linearly readable downstream; the Rademacher carrier is exactly
  what blocked this path in G3. If SUB comes back LIVE, that is the
  mundane-mechanism reading, not a surprise.
- SUPRA plausibly LIVE: injection-resonant (drive inside the fast
  tertile's natural range [10.1, 31.6]).
- RES least predictable.
- Whichever band fires, 3b activates; band identity shapes the scope
  note, not the branch.
- Record the D-sign pattern (Gate-B pattern-fold inheritance,
  observed-not-claimed).

## 4. The queue beyond L -- item cards
Format: WHAT / WHY / DEPENDS ON / DESIGN LOAD / SEAT-SAFETY / COST.

E1 ERRATUM (Phase-3 line-67, decade_drive).
  WHAT: replace "messages must live at envelope timescales" with the
  data-backed statement; wording per Gate-L branch (3a/3b).
  WHY: committed language already stages it ("No erratum files from
  this gate alone"; the WAIT lifts when Gate-L lands, either class).
  DEPENDS: Gate-L landed. LOAD: low, zero compute. SEAT-SAFETY: high
  (drafting from settled, committed sentences). Filing in
  decade_drive remains Jason's separate call, unchanged.

G4b K-EXTENSION (de-censor K*(s)).
  WHAT: extend the Phase-1 landscape grid above K=0.24, re-run the
  committed argmax rule, re-price the primary contrast.
  WHY: G4's magnitude is an AT-CEILING upper estimate; sign is
  settled. Only if the magnitude matters.
  DEPENDS: nothing. LOAD: low (committed machinery, one new axis).
  SEAT-SAFETY: high -- and RECOMMENDED AS THE SHAKEDOWN GATE for a
  new review seat (Section 6): a real but mechanically bounded gate
  that exercises the full lifecycle before anything design-heavy.

GD DRIFT-ATTRIBUTION.
  WHAT: instrumented chains locating WHERE the m0-referenced loss
  accrues per stage component (decode / re-encode / insertion).
  WHY: G4's e2e gap is DOMINATED by structural per-hop loss (0.0425
  of D=0.0564, committed decomposition); attribution makes that term
  actionable. Parked by prior decision; un-parking is Jason's call.
  DEPENDS: nothing hard. LOAD: medium (instrumentation design;
  mechanics are committed G1/G2). SEAT-SAFETY: medium.

GM ATTENUATION LAW.
  WHAT: a mechanism-derived law vs M1, Gate-B two-stage template
  (retrodict -> lock -> blind probe).
  WHY: the largest open scientific claim; "the LAW stays OPEN" is
  committed language twice. Banked seed in Section 7a.
  DEPENDS: nothing hard; benefits from Gate-L vocabulary either way.
  LOAD: HIGH (the derivation is the hardest remaining design task;
  the TEMPLATE is not). SEAT-SAFETY: template high / derivation low.

GC COHERENT RELAY (LIVE branch only).
  WHAT: price forwarding first-order coordinates vs the committed
  decode-and-forward baseline at matched span/hops.
  WHY: G4 is ARCHITECTURE-CONDITIONAL by its own scope; a live linear
  channel is the only thing that makes this gate meaningful.
  DEPENDS: Gate-L LIVE with non-marginal strength. LOAD: HIGH (what
  "coherent forwarding" concretely is in this architecture is an
  open design call). SEAT-SAFETY: low; do not attempt under a weak
  seat except in small pre-registered increments.

GL2 TARGETED POWER (MIXED branch only). Section 7c. LOAD low,
  SEAT-SAFETY high.

S1 RELAY SYNTHESIS (the program's own Phase-document).
  WHAT: the Relay Program's consolidated claims document, mirroring
  the Decade-Drive Phase-3 precedent: claims ledger with scopes
  VERBATIM from commit messages; verification chain; the C3 arc
  (claimed -> scoped at G3 -> resolved at Gate-L) as the spine;
  open items listed honestly (law OPEN unless GM lands; GD; the
  K-ceiling).
  WHY: the results now outnumber anyone's working memory; the
  synthesis is how the program survives seats, sessions, and time.
  DEPENDS: Gate-L landed. Does NOT wait on GM or GD -- list them
  open. (The classic failure is a synthesis that waits for
  completeness and therefore never ships.)
  LOAD: medium but mechanical under FILL-FROM discipline.
  SEAT-SAFETY: high IF every number is pulled from committed
  artifacts at write time; zero tolerance for recalled numbers.

H1 HARDWARE-MAPPING UPDATE (decision only).
  The Decade-Drive Phase-3 shipped a hardware mapping. Whether Relay
  results (staging, short-hop economics, square-law demod, and the
  Gate-L outcome) warrant an update is flagged as a decision point AT
  synthesis time. Jason's call; no default.

## 5. Recommended sequencing
DEFAULT (SHUT branch):
  L-close -> E1 -> G4b (shakedown; skip only if magnitude is
  explicitly not wanted) -> GD (if un-parked) -> S1 -> GM (when
  design capacity exists).
LIVE branch: L-close -> E1 -> G4b (shakedown) -> then the GC-vs-GM
  priority call is Jason's: GC grows a new limb (architecture), GM
  consolidates the trunk (law). The seat-capability constraint
  argues GM-template-with-deferred-derivation and GC deferred, but
  that is a constraint speaking, not the science.
MIXED branch: L-close -> GL2 -> then re-enter the tree at 3a/3b.
Rationale for the fixed early segment: E1 is free and owed; G4b
proves the process under the new seat on a bounded task BEFORE any
verdict anyone will argue about; S1 before GM because the synthesis
gains little from the law and the program gains much from the
synthesis.

## 6. Operating the program with a weaker review seat
Thesis: gates run under a weaker seat are exactly as valid as any,
PROVIDED the checklist fires. The process law (HANDOFF Section 5)
was built for this; the coming era is its stress test, not its
exception.

Task triage: mechanical items (E1, G4b, GL2, S1 filling, post-push
traces) are safe. Design-heavy items (GM derivation, GC) are
deferred, or proceed in small pre-registered increments where every
step is independently checkable.

Named hazards, each with its control:
1. RUBBER-STAMP PANELS. An empty panel is worse than a skipped one:
   it launders confidence. PROPOSED addition (needs Jason's
   ratification): panel output must be either artifact-traceable
   objections or an explicit no-findings statement LISTING the
   checks run. Silence is not a panel.
2. ATTESTED-AS-VERIFIED. Existing rule 14 already binds: every
   "verified" claim ships the recomputed numbers and artifact paths,
   verified-vs-attested stated explicitly. Enforce verbatim.
3. INVENTED SPECIFICS. Existing: FILL-FROM discipline; the
   caught-not-fixed filter (adopt structure, reject any number or
   equation without a traceable source) applies to the house seat
   under weakness exactly as it applies to external models.
4. SCOPE EROSION. Existing ledger rule: never quote a result without
   its scope. The scopes ARE the results.
5. THRESHOLD DRIFT. Existing: CONFORM-DONT-RENEGOTIATE;
   evaluate-at-use; tolerances never minted post-data.
6. AGREEMENT DRIFT over long sessions. Control: the STOP-and-report
   cadence already caps session scope; keep it tight -- one gate
   stage per sitting where possible.

Load rebalancing: Jason's pre-commit read-through deepens on exactly
what the checklist cannot mechanize -- scope language, consequence
wording, whether a verdict sentence claims more than its numbers.
External multi-model panels gain relative value under the
caught-not-fixed filter: several weak adversaries partially
substitute for one strong one.

Continuity mechanics unchanged: HANDOFF_* discipline; a fresh
instance's first act remains T1-T3 verification against the public
record, whatever the seat. If strong-seat access returns, design-
heavy items resume; nothing in the queue blocks on it.

## 7. Banked design reasoning (NON-NORMATIVE sketches; checkable, not findings)

### 7a. GM seed: the distributed-corner law
Committed ground (verified this session, c5a46e0): M1 wins; the
committed interpretive note ALREADY states why the one-pole lost --
the slow tertile's corner frequencies are distributed over
[1.00, 3.13] across 166 oscillators, the fitted window c=0.42-4.24
BRACKETS them, and a distributed-corner system shows intermediate
effective slope between 0 and 2 (p=1.04 sits there).
The seed: that corner distribution is NOT a free function -- it is
deterministic from committed build seeds. So the mechanism candidate
M3 = superposition of one-pole responses over the ACTUAL per-
oscillator corners, computable from committed artifacts, with zero
shape parameters (amplitude, at most one weighting exponent).
Maximally falsifiable: if M3 tracks M1 in-window, discriminate where
they diverge (extreme c) with a small pre-registered probe. The
Gate-B two-stage template ports VERBATIM (retrodict -> lock -> blind
probe; windows evaluate-at-use; dedupe check) -- a weak seat can
execute the template; only the derivation waits for design capacity,
and every equation in it must trace to a re-doable derivation or be
rejected (the external-review rule, applied internally).
Second committed structure any real law must speak to: the overlap
stratum sits systematically ABOVE the fit (all 5 residuals positive,
+0.024..+0.052, broadly increasing with c; committed reading: an
additional direct transfer path when the message band lies inside
the injection band, still power-borne per G3). Gate-L's beat-term
algebra (2*Re(conj(Z0)*dz)) is the natural vocabulary for modeling
that additive term -- whichever way Gate-L lands, its
instrumentation language feeds the stratum model.

### 7b. GC sketch conditions (LIVE branch only)
Contrast: e2e of a coherent-forwarding chain vs the COMMITTED G4
decode-and-forward cells at matched span/hops. Preconditions before
any spec: (i) LIN-LIVE strength well clear of 0.2 (a marginal
channel cannot carry a relay); (ii) a settled definition of
"coherent forwarding" in this architecture (state copy? Re/Im
re-injection? -- a real design call with no obvious default);
(iii) a reachability story for hop-2+ equivalent to Gate-L's audit.
Without all three, GC is not draftable.

### 7c. GL2 shape (MIXED branch only)
Gate-L's spec, restricted to the intermediate band(s); seeds extended
toward n=10; fresh decoy family (660000+ with proven collision
matrix); own delta evaluate-at-use; no other changes. Purely
mechanical; its only job is power.

## 8. Remaining strong-seat work (offer list, Jason picks)
(i) Adversarial self-panel on the Gate-L draft now, findings filed
against my own spec; (ii) GM design memo expanding 7a with the M3
derivation laid out stepwise for later independent checking;
(iii) S1 synthesis outline (section skeleton + FILL-FROM slots);
(iv) HANDOFF_4 at session end (always).

## 9. Provenance
Repo state: fresh clone, origin/master 39560d8, full T1-T3 trace
reported earlier this session. Artifacts recomputed this session:
gate3_mechanism.json (878c154850c7), gateB_broadband.json
(f2655b84), gateB_probe.json (222d9bd086e3), gate4_hoptrade.json,
plus commit metadata/messages for 8361553..39560d8. Attested only:
G0/G1/G2 internals beyond commit subjects (prior-session
verification per HANDOFF_3). The Gate-L spec is this session's
draft, unratified. This roadmap cites no other sources.

## 10. One line
Pre-register, mask, verify from artifacts, own errors by name -- and
let the machinery, not the seat, hold the verdicts. That sentence
was written for exactly the transition this document prepares for.

## Addendum (2026-07-17)
Status update, non-normative: the four Gate-L taste calls were
RATIFIED by Jason 2026-07-17 per the review seat's recommendations
(tripwire; DEAD criterion as drafted; ladder seed-0; both literals).
Adversarial panel run same day: P1-P4 filed and merged into spec v2
(zero-mean injection map; P_track ladder statistic; median-D
report-only; lag/window pin), pending Jason's confirm. CC execution
package staged (HANDOFF_CC_gateL.md). Queue position unchanged:
item 1 (Gate-L) in flight; items 2+ as sequenced above.
