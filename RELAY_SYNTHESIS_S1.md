# Relay Program -- S1 Synthesis
Author of record: Jason Dury <jason@eridos.ai> -- 2026-07-20.

Consolidated claims ledger for the Relay Program (repo decade_drive_2), mirroring the Decade-Drive Phase-3 precedent: every result carries its committed scope verbatim, the cross-band-routing (C3) arc is the spine, and open items are listed honestly. Numbers are filled from committed artifacts via a by-script values manifest (S1_values.json); this document is not a source for any number.

Operating point held across the program: K = 0.24, span 1.5 per hop. ESP-honest seed intersections: n = 8 (seeds 0-7) at G1, G3, G4, and Gate-L; G0 and G2 ran at n = 7. Every gate bridges hop-1 back to the committed decade_drive Phase-1 anchor 0.986454 (span-1.5 / K = 0.24, commit b0f7664).

## 1. Program question and inherited constraints

Question: can *engineered staging* carry information across a frequency span that a single passive hop cannot? The program tests the shipped prediction of the Decade-Drive Phase-3 document (Section 6, Branch B): an offline two-stage square-law relay transfers usable information across a compound three-decade *information path* where direct passive transfer is dead. "Compound span" is always an information-path claim (successive square-law demodulations end-to-end), never one physical spectrum.

Three constraints inherited from the Decade-Drive closure frame the work. DEFINITION ON THE RECORD: no committed artifact previously defined the labels C1/C2/C3 (commit messages already use "C3"); this document defines them, here, for the first time, traced to the committed README Background: in a multi-decade Stuart-Landau reservoir with the message in the fast band and readout in the slow band, cross-band routing runs through the |z|^2 square-law channel (C3), falls with frequency separation (C2), and dies past roughly two decades (C1).

- **C1 -- the horizon.** Passive cross-band routing dies past roughly two decades of separation, and populating the intervening spectrum with passively coupled oscillators does not help -- the horizon is a property of passive multi-decade coupling itself. The relay program inherits this wall; the only route past it is active staging (explicit per-stage square-law readout and reinjection).
- **C2 -- the attenuation.** Cross-band fidelity falls monotonically with frequency separation rather than at a cliff, so the interesting question about staging is a budget (how fidelity accumulates over hops), not a single pass/fail edge. This constraint is what the Gate-B arm later turns into a quantitative, falsifiable attenuation model.
- **C3 -- the channel (the spine).** Decade-Drive's Phase-3 claimed the message crosses bands via the quadratic |z|^2 (power-envelope) observable, recovered by slow-band square-law readout. This is the load-bearing mechanism claim the whole program interrogates; Section 2 is the story of how that single claim was scoped, probed, and finally resolved.

## 2. The C3 arc (the spine)

**Claimed (Phase-3).** Cross-band routing is carried by the |z|^2 square-law channel; the message injected in the fast band is recovered by slow-band square-law readout. (The Phase-3 line-67 wording that messages "must live at envelope timescales" is the sentence the E1 erratum will correct; see below.)

**Scoped (Gate-3, commit d1a7116).** The readout observable-order ablation returns, verbatim:

> **MECHANISM MATRIX -- SUB=SQ-carried, RES=SQ-carried, SUPRA=SQ-carried. C3 STANDS unmodified (SQ-carried across all classified bands, RES included); the broadband side-finding needs another explanation (report, no new claim). [SCOPED -- the injection supplies no linear message content; the LIN-carried branch tested only reservoir-mediated re-encoding into first-order coordinates, which did not occur. NOT an independent refutation of a linear resonant channel. See "Degeneracy and the scope of this verdict".]**

Cross-band routing is retained on the quadratic observable at every classified band (FULL r2: SUB 0.9865, RES 0.8601, SUPRA 0.6650). Crucially the verdict is SCOPED, not a refutation of a linear channel: the AM injection supplies zero coherent linear content (u = 0.5*sqrt(s)*w with u^2 = 0.25*s exact and eval-window |corr(u,s)| <= 0.032 in all three bands), and the linear-observable branch sat at or below its own never-injected decoy null in 24/24 cells (its real r2 is negative in 20/24). "SQ-carried" therefore reads as "the linear observable order is insufficient AND the message is retained on |z|^2" -- it does not test a coherent linear channel, which was never injected.

**Attenuation law probed (Gate-B, commit c5a46e0 + blind probe 8794fe7).** Stage-1 is a pre-registered retrodiction; stage-2 is an out-of-sample blind test. Verdict: EXPLAINED --

- Best smooth model M1 (SNR = A*c^-p): A = 29.86, p = 1.0429 (2sd [0.3441, 1.7417]), wRMSE = 0.0086. It beats the mechanism-derived one-pole form by 2.3x and the best cliff by 5.27x. The winning model is a *generic* power law: the specific attenuation LAW stays OPEN (committed interpretive note).
- The overlap stratum sits systematically above the fit (all 5 residuals positive, +0.0239 to +0.0524, over c = 5.41 to 16.73) -- a committed pre-data flag, consistent with an additional direct transfer path when the message band lies inside the injection band, still power-borne.
- Blind probe (stage-2): PREDICTIVE PASS -- all three held-out bands land inside their locked windows, deviations +0.0069, +0.0206, +0.0402 (all positive, folded to Gate-L).

**Resolved DUAL-CHANNEL (Gate-L, commit c11ff5e).** The symmetric question -- does the first-order slow-tertile coordinate carry the message under *coherent* linear injection (corr(u,s) = 1)? -- returns, verbatim:

> **COHERENT-LIN-LIVE (SUB, RES, SUPRA LIN-LIVE): SUB=LIN-LIVE; RES=LIN-LIVE; SUPRA=LIN-LIVE**

A coherent linear channel is LIVE at every band. Coherent r2_LIN rounds to 1.0000 at all three bands, against the committed Gate-3 LIN baseline (read-at-use, sha-asserted): D_LIN = +1.0558 +/- 0.0548 (SUB, delta 0.1096), +1.0426 +/- 0.0155 (RES, delta 0.0310), +1.0210 +/- 0.0056 (SUPRA, delta 0.0200, floor binds). All 24 per-seed D are positive (24/24 across SUB, RES, SUPRA); the tightest leave-one-seed-out is seed 0 (margin +0.9348); the live floor is clean (max fresh LIN decoy p95 = 0.1764 at cell SUB|2|LIN, below the 0.20 bar). The SQ and FULL columns are fenced pre-data -- "All falsifiable content lives in the LIN column" -- so a high coherent SQ r2 is not read as a quadratic-mechanism signature. One accepted gap: the P_slow ladder analog was not recorded at the battery (trajectories not stored; does not affect the verdict or any gated instrument).

Consequence for C3: the claim resolves DUAL-CHANNEL. The |z|^2 power-envelope channel carries a power-encoded message (Gate-3, SCOPED), AND a coherent first-order linear channel also exists where injected coherently at SUB, RES, and SUPRA (Gate-L, one operating point K = 0.24 / span 1.5, stage-A, offline, no chains; readout-level -- an observable-order result at the readout, not an internal transfer-path attribution). Both readings are scoped exactly as committed; neither is over-generalized past its injection.

**E1 -- the erratum (FILED).** FILED 2026-07-19 -- decade_drive commit db5fec6c14a6ef8b89b0e286c36bf96997ea4e85 (Docs/ERRATA.md, amend-not-overwrite; lines 67/207 quoted byte-exact; correcting record decade_drive_2 c11ff5e). The pre-registered WAIT lifted on the Gate-L landing (class COHERENT-LIN-LIVE selects DUAL-CHANNEL wording); filed on Jason's word 2026-07-19, with a discoverability pointer in the Phase-3 header at decade_drive cf053a9dea7d61efbed4cd682da9c7d0ec153a41 ('(0495842 numbering)' anchor).

## 3. Relay engineering ledger

**G0 -- staging (commit 8361553): PASS.**

> **PASS (horizon is architectural): relay e2e beats the fresh direct span-3.0 baseline AND the e2e decoy floor by > seed sigma on the ESP-honest paired intersection**

Relay end-to-end r2(m2,m0) = +0.9264 +/- 0.0233 vs a fresh paired direct span-3.0 baseline -0.0017 +/- 0.0064 (paired delta +0.9281); ESP-honest paired intersection n = 7; the ~2-decade passive horizon is architectural, crossed by staging. Provenance note: commit 8361553 carries two poses -- the first-pose control was underpowered (it returned NO-MEASUREMENT because the bandwidth-violation control did not collapse), the re-posed bandwidth-violation control collapsed soundly, and the PASS is the re-posed record; the verdict-row numbers are bit-identical between poses. Scope: offline two-stage square-law relay, compound span 3.0 = information path.

**G1 -- two-regime loss law (commit 7d6f3f2): class A, BUDGET-LIKE.**

> **A -- BUDGET-LIKE (pre-registered; TWO-REGIME, NOT a clean constant-per-hop cost): the registered linear rho_k slope is flat (-0.0002 +/- 0.0055, |slope| <= max(2*SE,0.01)=0.0109; floor INERT since 2*SE binds). Ladder PRICED -- B/depth-limit REFUTED: per-hop r2_hop is flat with depth (slope t=-1.02, no error amplification). SHAPE: a one-time first-relay INSERTION loss rho_2=0.941 (largest single drop) + a near-flat STEADY-STATE mean rho_3..5~0.960 carrying a mild late drift (slope -0.0148, t=-4.06, 7/8 seeds neg) that is UNRESOLVED at n=8/H=5 (lives mostly in the last hop). H_half=16.0 (from the OVERALL budget rho 0.956, insertion loss included) is an EXTRAPOLATION (3.2x beyond measured H=5, off a U-shaped rho) -- not a measured horizon.**

The registered loss-law slope is flat (-0.0002 +/- 0.0055) and the depth-limit is REFUTED (per-hop reconstruction is depth-flat -- no error amplification). Shape is two-regime: a one-time first-relay insertion loss (rho_2 = 0.9412) plus a near-flat steady state carrying a mild late drift that is UNRESOLVED at H = 5. Scope: offline H = 5 chain.

**G2 -- real drift at depth (commit b49834f): class B, STEADY DECLINE.**

> **B2 -- STEADY DECLINE (pre-registered): the m0-referenced steady-state ratio rho_k (k=3..10) FALLS with depth, slope -0.0135 +/- 0.0015 < -max(2*SE,0.01)=-0.0100 (t=-8.85 vs zero; the margin -0.0035 is measured vs the 0.01 min-effect FLOOR, not a noise scale, so the DIRECTION is decisive). The Gate-1 late drift is REAL at depth. MECHANISM -- this is an END-TO-END, m0-referenced decline (systematic drift away from the source), NOT per-hop error amplification: per-hop reconstruction r2_hop (each stage vs its OWN immediate input) is depth-flat (slope t=1.50 over hops 2..10, n.s.), so per-stage compounding is refuted while the m0-referenced decline stands -- r2_hop scores a drifted moving target and is blind to the m0 loss. MAGNITUDE modest, horizon EXTRAPOLATED: rho_ss=0.9277, rho falls 0.970->0.872 over hops 3->10; measured r2_cum(H=10)=0.5069 +/- 0.0284 has NOT crossed the 0.5 half-power point; H_half=11.0 is a one-hop trend-extrapolation, so 'bounded' is inferred by trend continuation, not observed. Insertion loss rho_2=0.937 (one-time, excluded).**

Extending to H = 10, the m0-referenced steady-state ratio (k = 3..10) FALLS with depth: slope -0.0135 +/- 0.0015 (past the 0.01 min-effect floor -- a threshold, not a noise scale -- so the direction is decisive). The Gate-1 late drift is REAL at depth; it is an end-to-end m0-referenced decline, NOT per-hop error amplification (per-hop reconstruction stays depth-flat). Magnitude modest, horizon extrapolated: rho_ss = 0.9277, measured r2_cum(H=10) = 0.5069 has not crossed the 0.5 half-power point.

**G4 -- short hops win (commit 39560d8): SHORT-WINS.**

> **SHORT-WINS (primary short-vs-long @ S=3.0): D = e2e(3 x 1.0) - e2e(2 x 1.5) = +0.0564 +/- 0.0200 (paired n=8); delta = max(2*SE,0.02) = 0.0401. SHORT hops win. [SCOPED: K<=0.24, AT-CEILING right-censored -- see verdict scope]**

At matched total span S = H*s, many short hops beat few long hops for the |z|^2 decode-and-forward relay: D = e2e(3 x 1.0) - e2e(2 x 1.5) = +0.0564 +/- 0.0200 (paired n = 8, delta 0.0401). The advantage is structural: the short arm pays a far smaller per-hop insertion cost (insertion rho_2 0.9848 for 3 x 1.0 vs 0.9412 for 2 x 1.5), and the committed decomposition attributes the gap to structural per-hop loss (routing->e2e loss 0.0577 for the long arm's 2 hops vs 0.0152 for the short arm's 3 at near-equal routing). Robustness: SHORT-WINS survives 8/8 single-seed drops; the tightest drop is seed 1 (survival margin +0.0062).

Both committed scope tags travel with the verdict. The AT-CEILING tag -- `[SCOPED: K<=0.24, AT-CEILING right-censored -- see verdict scope]` -- is in the verdict line above: a shared coupling ceiling under-tunes longer hops more, so the primary D is an AT-CEILING upper estimate expected to compress for K > 0.24, while the SHORT-WINS sign is robust. The architecture-conditional tag, verbatim:

> [ARCHITECTURE-CONDITIONAL.] This trade prices the relay AS BUILT: an OFFLINE DECODE-AND-FORWARD chain in which each hop ridge-decodes the |z|^2 channel (Gate-3-pinned) to the message estimate and RE-MODULATES it into a fresh span-s network, with the per-hop insertion cost from Gate-1's re-injection plateau (insertion rho_2 ~ 0.985 span-1.0, ~0.941 span-1.5, from the committed decomposition). Compound span S=H*s is INFORMATION-PATH accounting (H successive square-law demodulations end-to-end), NOT one physical spectrum. The short-hop advantage is a property of THIS scheme's economics and does NOT generalize to repeaters in general: amplify-and-forward, coherent/analog relaying, soft-information forwarding, joint multi-hop decoding, or any change to the decoder or per-hop insertion cost could reweight or invert the trade.

## 4. Established vs open

**Established (each with its committed scope):**
- Engineered staging crosses the horizon -- G0 PASS, offline two-stage square-law relay, information-path span 3.0 (relay e2e +0.9264 vs direct -0.0017).
- Multi-hop loss is budget-like and two-regime; depth-limit refuted -- G1 class A, offline H = 5.
- Steady decline at depth is REAL -- G2 class B at H = 10; the horizon (H_half) is extrapolated, not observed (r2_cum(H=10) = 0.5069 has not crossed 0.5).
- Cross-band routing is |z|^2-carried at all bands -- G3 SQ-carried, SCOPED to the AM (power) injection; NOT a refutation of a linear channel.
- A coherent first-order linear channel ALSO exists -- Gate-L COHERENT-LIN-LIVE (SUB, RES, SUPRA), one operating point K = 0.24 / span 1.5, stage-A. Together: C3 = DUAL-CHANNEL.
- Attenuation is smooth (no cliff, no rate limit) -- Gate-B EXPLAINED (cliff loses 5.27x) and the locked M1 interpolation survived a blind probe (PREDICTIVE PASS). The specific attenuation LAW is NOT established (see below).
- Short hops beat long hops -- G4 SHORT-WINS (D = +0.0564 +/- 0.0200, n = 8), SCOPED AT-CEILING right-censored (K <= 0.24) and ARCHITECTURE-CONDITIONAL (offline decode-and-forward only).
- The coupling optimum lies above the tested range -- G4b MEASURED (commit e14ae6ada4af0ea3c0e2b08f8cde3dffd942cb08): K*(s) remains right-censored at the doubled ceiling (K_hat = 0.48 = new grid max at spans 1.0/1.5/2.0; span-2.0 still climbing, mean 0.7448 -> 0.9313 across K = 0.24 -> 0.48, edge margin +0.0395); span 3.0 is NOISE-ARGMAX (context-only; UNSTABLE-ARGMAX and RAW-vs-ESP-honest DIVERGENT disclosed); no ESP wall in the tested range (4/160 extension-cell failures, span-3.0 seeds 7-8, recovering by K = 0.40); anchors 40/40 digit-exact. Scope: measurement gate; Gate-4's contrast verdict is not re-litigated.

**Open:**
- **GM -- the attenuation law identity.** M1 is a generic power law, explicitly not a mechanism; "the LAW stays OPEN" is committed language. The candidate M3 (distributed-corner superposition over the actual per-oscillator corners) is banked but un-derived.
- **GD -- the drift mechanism.** WHERE the G2 m0-referenced loss accrues per stage (decode / re-encode / insertion) is out of scope in every gate so far; instrumented drift-attribution chains are un-run.
- **K beyond 0.48.** The argmax remains right-censored at the doubled ceiling; the extend-or-stop call is folded into GM design (ruled 2026-07-19).
- **GC -- coherent relay.** Unlocked by the Gate-L LIVE branch; G4 is ARCHITECTURE-CONDITIONAL and explicitly excludes coherent/analog forwarding. What "coherent forwarding" concretely is in this architecture is an open design call; the gate is not yet draftable.

## 5. Method note

Every verdict was pre-registered: each gate has a ratified spec (relay_gate*_spec.md) binding functional forms, thresholds, strata, and acceptance rules before data; pass windows are evaluated at use from byte-locked (D, SE, n) and are never stored, and no tolerance is minted post-data. Each gate is bridged to the committed record by an anchor (hop-1 must reproduce the Phase-1 anchor 0.986454 at K = 0.24 / span 1.5, digit-exact to 6 dp; bit-exactness is retained only as a diagnostic). Specificity rests on fresh never-injected same-class decoy nulls with a collision-free seed census; comparisons use ESP-honest pairing (a seed failing any arm is dropped everywhere); NO-MEASUREMENT discloses only the tripped instrument and keeps the verdict sealed until the resolution is ratified blind; robustness is leave-one-seed-out; and Gate-B's locked P1 predictions were tested out-of-sample by a blind probe. The specs govern; this synthesis only assembles their committed outputs.

## -- Plain-language explainer --

Picture a row of tuning forks of very different pitches, and a message written as a slow swell in the loudness of the high-pitched end. If you just line the forks up and let them nudge each other, the message can only travel so far down the row before it smears into nothing -- about two 'octaves' worth of pitch difference, and after that it is gone. That dead-end is not a bug you can pad your way around; it is baked into passive coupling. The Decade-Drive work found that the message that DOES get through rides on the *loudness* (the square of the signal), not on the raw wiggle.

This program asks a plumbing question: if the passive row dies at two octaves, can you beat it by adding *repeaters* -- little stations that read the loudness out, clean it up, and re-broadcast it into a fresh stretch? The answer is yes. One relay already carries the message a full three octaves where the direct route is stone dead (G0). Chaining more repeaters, the quality drains slowly and predictably rather than falling off a cliff (G1, G2) -- with a real, slow sag once the chain gets long. And given a fixed total distance, many short hops beat a few long hops, because each hand-off pays a fixed toll and short hops keep that toll small (G4). Each of those results carries a fine-print label: the short-hop win is measured at the edge of the tuning knob and only for this repeater design.

The spine of the story is one question that kept getting sharper: HOW does the message cross bands? The original claim was 'only through loudness.' We first confirmed the loudness channel works and, honestly, noted our test could not have seen a different channel because we never fed one in (G3). We then showed the fade-with-distance is smooth and even predicted three unseen cases correctly (Gate-B). Finally we fed in the OTHER kind of signal -- a clean, in-step wiggle -- and it sailed through at every pitch (Gate-L). So the honest final answer is not 'only loudness' but 'two channels': the loudness channel we always knew, plus a direct in-step channel that is real when you actually supply it. That correction is now on the record as erratum E1 (decade_drive Docs/ERRATA.md, commit db5fec6). Still open: the exact equation for the fade (GM), exactly where the long-chain sag comes from (GD), how far the short-hop win holds once you turn the knob past its current stop (G4b), and whether a relay could forward the in-step channel directly (GC).

