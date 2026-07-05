# Relay Gate-1 -- multi-hop loss-law record (H=5)

Spec: relay_gate1_multihop_spec.md (sha256 ca078238ec8f). Harness: experiments/relay_gate1.py (sha256 8831da5ecd2f).
Seeds run: [0, 1, 2, 3, 4, 5, 6, 7]. K = 0.24. Wall-clock 22 min. Seed scheme collision-free: True.
Provenance: numbers produced by the battery run (harness sha256 c262004a7e02, 22 min GPU); verdict RE-FRAMED (two-regime amendment) by --reread (harness sha256 8831da5ecd2f) with every arm's recs asserted byte-identical -- no measured number changed.

Framing: Compound span 1.5*H is a claim about the INFORMATION PATH (H successive square-law demodulations end-to-end), NOT one physical spectrum; each repeater re-injects into a fresh span-1.5 network. This gate measures the SHAPE of the loss curve (budget vs compounding); the chain-vs-direct floor was settled at Gate-0 (commit 8361553) and direct arms are NOT re-run (committed b0f7664 floors cited).

## Verdict

**A -- BUDGET-LIKE (pre-registered; TWO-REGIME, NOT a clean constant-per-hop cost): the registered linear rho_k slope is flat (-0.0002 +/- 0.0055, |slope| <= max(2*SE,0.01)=0.0109; floor INERT since 2*SE binds). Ladder PRICED -- B/depth-limit REFUTED: per-hop r2_hop is flat with depth (slope t=-1.02, no error amplification). SHAPE: a one-time first-relay INSERTION loss rho_2=0.941 (largest single drop) + a near-flat STEADY-STATE mean rho_3..5~0.960 carrying a mild late drift (slope -0.0148, t=-4.06, 7/8 seeds neg) that is UNRESOLVED at n=8/H=5 (lives mostly in the last hop). H_half=16.0 (from the OVERALL budget rho 0.956, insertion loss included) is an EXTRAPOLATION (3.2x beyond measured H=5, off a U-shaped rho) -- not a measured horizon.**

## Instrument checks (pre-registered order, before the A/B read)

1. **Anchor (hop-1 == committed span-1.5/K=0.24)**: intersection mean 0.986454 (SE 0.001116, n=8); target 0.986 +/- 0.0200; deviation +0.000454 -> OK.
2. **Decoy floors at depth** (intersection p95 means): per-stage {'1': -0.0342, '2': -0.0322, '3': -0.0205, '4': -0.0324, '5': -0.0492}, e2e-at-depth -0.4355; elevated bar 0.2 -> clean.
3. **Filter-violation at depth** ([2,9] message, first repeater [0.2,0.9]): n=8, e2e mean -1.2537 (bar e2e r2 < 0.1 every seed) -> COLLAPSED (sound). Signature rms_in 0.007138, scale 20.6.
4. **ESP-honest paired intersection** (ESP-ok across ALL 5 stages): [0, 1, 2, 3, 4, 5, 6, 7] (n=8).

## Cumulative fidelity r2(m_k, m0) (intersection means +/- SE)

- hop 1: **+0.9865 +/- 0.0011** (per-seed [0.981, 0.988, 0.99, 0.982, 0.99, 0.986, 0.987, 0.987])
- hop 2: **+0.9284 +/- 0.0203** (per-seed [0.962, 0.898, 0.972, 0.948, 0.797, 0.946, 0.96, 0.942])
- hop 3: **+0.9023 +/- 0.0197** (per-seed [0.931, 0.884, 0.946, 0.919, 0.775, 0.905, 0.948, 0.911])
- hop 4: **+0.8724 +/- 0.0181** (per-seed [0.896, 0.86, 0.921, 0.885, 0.76, 0.859, 0.92, 0.879])
- hop 5: **+0.8228 +/- 0.0221** (per-seed [0.842, 0.819, 0.896, 0.832, 0.693, 0.812, 0.886, 0.801])

## Loss law (rho_k = r2_k / r2_{k-1}; validity guard r2_{k-1} > 0.2)

- **loss-law slope = -0.0002 +/- 0.0055** (class A); threshold max(2*SE,0.01) = 0.0109; margin (thr-|slope|) = +0.0107 -> inside flat band (A-consistent). Pin B: a positive slope beyond +threshold is instrument-suspicion, never A.
- classification: **A**; budget rho = 0.9556; extrapolated H_half = 16.0.
- rho by level: {'2': 0.9412, '3': 0.972, '4': 0.967, '5': 0.9424} (valid-n {'2': 8, '3': 8, '4': 8, '5': 8}).

## Loss-law SHAPE (two-regime; amendment -- descriptive, does not change the registered A)

- **B / depth-limit REFUTED (mechanism):** per-hop reconstruction fidelity r2_hop = r2(m_k, processed-m_(k-1)) is FLAT with depth ({'1': 0.9865, '2': 0.9862, '3': 0.9841, '4': 0.9866, '5': 0.9827}; slope t=-1.02) -> no error amplification; each stage reconstructs its input equally well at any depth. The ladder is PRICED, not merely 'not yet bounded'.
- **Regime 1 -- one-time first-relay insertion loss:** rho_2 = 0.9412 +/- 0.0210 (the LARGEST single-hop drop; hop-1 clean injection -> hop-2 first relay).
- **Regime 2 -- near-flat steady-state:** mean rho_3..5 = 0.9605 +/- 0.0040 (distinct from the loss-law OVERALL budget rho 0.9556, which pools in the first-relay insertion loss). Slope -0.01482 +/- 0.00366 (t=-4.06, 7 seeds neg). This mild late drift is UNRESOLVED at this depth: it lives mostly in the LAST hop (mean(rho_last - rho_3)=-0.0296 vs mean(rho_mid - rho_3)=-0.0050), so at n/H=5 a genuine steady decline is NOT separable from a single noisy last hop.
- **H_half caveat:** H_half=16.0 is an EXTRAPOLATION (3.2x beyond the measured H=5, off a U-shaped rho) assuming a tail constancy the data mildly violate -- a projected budget, NOT a measured horizon. Depth extension (H>5) is the clean test.

## Scramble robustness line (characterisation only, never verdict)

- scrambled-stage-3 chain e2e = +0.8266 +/- 0.0236 vs compliant +0.8228 +/- 0.0221 (n=8) -- loss-law topology-generic (3-sigma/0.1 bar = code-level operationalization).

## Scope

Offline H=5 chain; compound span 1.5*H = an INFORMATION-PATH claim (H successive
square-law demodulations), NOT one physical spectrum. STOP-and-report: no Gate-2
(hop-length), no mechanism-decomposition, no interpretation beyond the A/B/C mapping.
