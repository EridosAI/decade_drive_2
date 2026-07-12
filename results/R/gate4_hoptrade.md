# Relay Gate-4 -- Hop-length trade at matched total span (S = H*s)

Spec: relay_gate4_hoptrade_spec.md (sha256 f8445bdba331). Harness: experiments/relay_gate4.py (sha256 1b9d419c2720).
Wall-clock 14 min. Seed scheme collision-free: True.

Framing: Matched total span S = H*s: (1 x 3.0), (2 x 1.5), (3 x 1.0) all carry the message a compound span of 3 decades along the INFORMATION PATH (H successive square-law demodulations end-to-end), NOT one physical spectrum; each repeater re-injects into a fresh span-s network. K per span = K*(s) argmax of the committed Phase-1 landscape (each hop tuned as well as the committed data allows -- the architect's question). This gate prices the hop-length trade (short vs long) for the |z|^2 channel Gate-3 pinned.

## Verdict

**SHORT-WINS (primary short-vs-long @ S=3.0): D = e2e(3 x 1.0) - e2e(2 x 1.5) = +0.0564 +/- 0.0200 (paired n=8); delta = max(2*SE,0.02) = 0.0401. SHORT hops win. [SCOPED: K<=0.24, AT-CEILING right-censored -- see verdict scope]**

## Deviations (ship with the record; ratified NM-resolution 2026-07-11)

1. **Anchor implementation was STRICTER than ratified (conformed, not renegotiated).** The spec's anchor line reads 'digit-exact'; the house standard is 6dp (REF_TABLE storage, every smoke target, Gate-3 per-seed anchor). The first implementation pinned bit-exact (diff == 0.0), stricter than ratified. Under the ratified digit-exact (6dp) standard all hop-1 cells pass. Bit-exactness is retained as DIAGNOSTIC only: the (1x2.0) seed-3 cell differs by 3 ULP (3.33e-16), attributed to floating-point reduction-order in the span-2.0 batch integration (the batch-shape ULP-drift is documented in relay_gate0._hop's docstring; Gate-4 reports it as diagnostic in _anchor_report; span 2.0 is the first span the program reproduces at this level). No post-data tolerance was minted -- the ratified text governs (the Gate-B H1-window precedent). The Gate-1 replay instrument keeps its bit-exact binding (sandbox-certified).
2. **The sealed verdict was disclosed under NM (process deviation, named).** The STOP report at the NM checkpoint disclosed the sealed verdict -- the gated contrast (D, delta, classification), the pricing table, the decomposition, and the per-seed values -- while NO-MEASUREMENT formally stood, violating NM-before-verdict blindness (resolution decisions must be made blind). Contained: the resolution derives from the ratified spec text alone (outcome-independent -- identical ruling had the sealed verdict been LONG-WINS), the drifted cell is on the consistency-only (1x2.0) config and never touched the gating contrast, and the recs are byte-locked. Standing rule, now CODE-ENFORCED in this _write_md (NM verdict suppresses every sealed section; NM-shape self-test in verdict_test): an NM STOP report discloses the instrument failure ONLY; the sealed verdict stays sealed until the NM resolution is ratified.

## K*(s) lookup (argmax_K mean r2_d0 over the committed landscape; first eyes at sandbox)

- span 1.0: K*(s) = **0.24** (margin 0.0022)  [K=0.0:-1.020, K=0.08:+0.891, K=0.12:+0.983, K=0.16:+0.998, K=0.24:+1.000]
- span 1.5: K*(s) = **0.24** (margin 0.0438)  [K=0.0:-1.706, K=0.08:+0.734, K=0.12:+0.876, K=0.16:+0.942, K=0.24:+0.986]
- span 2.0: K*(s) = **0.24** (margin 0.1361)  [K=0.0:-4.672, K=0.08:+0.409, K=0.12:+0.535, K=0.16:+0.609, K=0.24:+0.745]
- span 3.0: K*(s) = **0.24** (margin 0.0011) [NOISE-ARGMAX]  [K=0.0:-27.219, K=0.08:-0.006, K=0.12:-0.008, K=0.16:-0.005, K=0.24:-0.004]

## Instrument checks (pre-registered order, before the contrast read)

1. **Anchors** (every GPU config hop-1 vs committed cell). GATE = digit-exact (6dp, ratified): ALL PASS. Diagnostic (not gated): bit-exact 39/40; ULP-level cells [1x2.0 seed 3 = 3 ULP (3.33e-16)] = FP reduction-order (span-2.0 batch).
2. **Decoy floors** (per-stage + e2e p95 means): max 0.0159; bar 0.2 -> clean.
3. **Gate-1 replay** (K=0.24 (2 x 1.5) arm hops 1-2): BIT-EXACT (arm 2x1.5).
4. **ESP pairing**: symmetric intersections printed per contrast; per-config intersections in the pricing table.

## Pricing table (per-config e2e mean +/- SE over per-config ESP intersection)

-       1x3.0 (H=1, span=3.0, K=0.24, context): e2e = -0.0017 +/- 0.0064 (n=7, primary) [SOURCED-FROM-COMMITTED, context-only]
-       2x1.5 (H=2, span=1.5, K=0.24, trade): e2e = 0.9284 +/- 0.0203 (n=8, primary)
-       3x1.0 (H=3, span=1.0, K=0.24, trade): e2e = 0.9847 +/- 0.0049 (n=8, primary)
-       1x2.0 (H=1, span=2.0, K=0.24, trade): e2e = 0.7570 +/- 0.0189 (n=7, secondary)
-       2x1.0 (H=2, span=1.0, K=0.24, trade): e2e = 0.9848 +/- 0.0049 (n=8, secondary)

## Contrasts (D = e2e(minuend) - e2e(subtrahend), per-seed paired)

- **PRIMARY (short-vs-long @ S=3.0)**: D = 0.0564 +/- 0.0200 (paired n=8, seeds [0, 1, 2, 3, 4, 5, 6, 7]); delta = 0.0401 -> **SHORT-WINS** (GATING).
- **SECONDARY (@ S=2.0, consistency-only)**: D = 0.2291 +/- 0.0175 (paired n=7, seeds [0, 1, 2, 3, 4, 5, 6]); delta = 0.0349 -> **SHORT-WINS** (consistency-only).

## Robustness -- leave-one-seed-out (primary contrast)

- **SHORT-WINS survives 8/8 single-seed drops** (ALL kept-means clear their max(2*SE,0.02) delta; all deltas 2*SE-governed). Removing the largest-|D| seed 4 (the fat tail) STRENGTHENS the verdict (sd 0.0567 -> 0.0291: the fat tail is the variance source, not the load-bearing seed). The TIGHTEST drop is seed 1 (survival margin +0.0062).

## Decomposition (m0-referenced rho_k; reported, NOT gating)

- 2x1.5: insertion rho_2 = 0.9412; per-hop rho {'2': 0.9411994007639677}
- 3x1.0: insertion rho_2 = 0.9848, slope 0.0151 +/- 0.0049; per-hop rho {'2': 0.9848323295694211, '3': 0.999956397996125}
- 2x1.0: insertion rho_2 = 0.9848; per-hop rho {'2': 0.9848323295694211}

## Verdict scope (ratified addendum; sharpened with the gradient asymmetry, panel-confirmed)

[SCOPED: K<=0.24, AT-CEILING right-censored.] K*(s)=0.24 at every span, but the routing curves are UNEQUALLY converged there: span-1.0 (short arm) is SATURATED (r2_d0=0.99995, 5.4e-05 from the hard r2=1 ceiling, K0.16->0.24 gain +0.0022, slope 0.028/K) while span-1.5 (long arm) is only near-saturated (r2_d0=0.98607, gain +0.0438, slope 0.55/K) and span-2.0 still climbs steeply (r2_d0=0.74476, gain +0.1361, slope 1.70/K) -- a ~20x-60x gradient asymmetry. A shared ceiling under-tunes LONGER hops more, so the primary D is an AT-CEILING UPPER estimate expected to COMPRESS for K>0.24. The SHORT-WINS SIGN is robust: (i) the long arm sits within 1.4% of routing saturation, so the censored differential headroom (~0.014/hop) is far below the e2e gap, which is dominated by STRUCTURAL per-hop loss (routing->e2e loss 0.0577 for the long arm's 2 hops vs 0.0152 for the short arm's 3 at near-equal routing), not coupling suboptimality; (ii) WITHIN K<=0.24 the short-minus-long routing gap SHRINKS monotonically as K rises (+0.1574@K0.08 -> +0.0139@K0.24), so K=0.24 is already the LEAST-favorable in-range coupling for SHORT and it still wins by the full D with all seeds positive. What the right-censored grid CANNOT certify above the ceiling: the MAGNITUDE, that the primary stays FULL-STRENGTH, and (by pure e2e-headroom arithmetic, long headroom > D) that a LONG-WINS reversal is impossible -- a reversal is mechanistically implausible but NOT formally excluded. The SECONDARY (S=2.0) contrast is more censoring-exposed (long arm far below saturation) so it is NOT ceiling-stable; it is consistency-only and non-gating. Behavior above K=0.24 is out of scope for this gate.

## Protocol scope (architecture-conditionality)

[ARCHITECTURE-CONDITIONAL.] This trade prices the relay AS BUILT: an OFFLINE DECODE-AND-FORWARD chain in which each hop ridge-decodes the |z|^2 channel (Gate-3-pinned) to the message estimate and RE-MODULATES it into a fresh span-s network, with the per-hop insertion cost from Gate-1's re-injection plateau (insertion rho_2 ~ 0.985 span-1.0, ~0.941 span-1.5, from the committed decomposition). Compound span S=H*s is INFORMATION-PATH accounting (H successive square-law demodulations end-to-end), NOT one physical spectrum. The short-hop advantage is a property of THIS scheme's economics and does NOT generalize to repeaters in general: amplify-and-forward, coherent/analog relaying, soft-information forwarding, joint multi-hop decoding, or any change to the decoder or per-hop insertion cost could reweight or invert the trade.

## Scope

Offline decode-and-forward chains; S_total = H*s (compound-span information path). K per span = K*(s) argmax of the committed Phase-1 landscape (the architect's 'each hop tuned as well as the data allows' question). Pass windows are NOT stored (delta evaluated at verdict from byte-locked (D, SE, n)). STOP-and-report.
