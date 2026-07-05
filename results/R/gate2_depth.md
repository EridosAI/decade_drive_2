# Relay Gate-2 -- depth-extension loss-law record (H=10)

Spec: relay_gate2_multihop_spec.md (sha256 4e8bd139a3cb). Harness: experiments/relay_gate2.py (sha256 77d46d0a5333).
Seeds run: [0, 1, 2, 3, 4, 5, 6, 7]. K = 0.24. Wall-clock 32 min. Seed scheme collision-free: True.
Provenance: numbers from the battery run (harness sha256 eefb10d94630, 32 min GPU); verdict RE-FRAMED by --reread (sha256 77d46d0a5333), recs asserted byte-identical.

Framing: Compound span 1.5*H is an INFORMATION-PATH claim (H successive square-law demodulations end-to-end), NOT one physical spectrum; each repeater re-injects into a fresh span-1.5 network. Depth extension H=5->10; the insertion loss rho_2 is EXCLUDED from the trend statistic by pre-registration (a known one-time cost). No new chain-vs-direct claim; committed b0f7664 floors cited.

## Verdict

**B2 -- STEADY DECLINE (pre-registered): the m0-referenced steady-state ratio rho_k (k=3..10) FALLS with depth, slope -0.0135 +/- 0.0015 < -max(2*SE,0.01)=-0.0100 (t=-8.85 vs zero; the margin -0.0035 is measured vs the 0.01 min-effect FLOOR, not a noise scale, so the DIRECTION is decisive). The Gate-1 late drift is REAL at depth. MECHANISM -- this is an END-TO-END, m0-referenced decline (systematic drift away from the source), NOT per-hop error amplification: per-hop reconstruction r2_hop (each stage vs its OWN immediate input) is depth-flat (slope t=1.50 over hops 2..10, n.s.), so per-stage compounding is refuted while the m0-referenced decline stands -- r2_hop scores a drifted moving target and is blind to the m0 loss. MAGNITUDE modest, horizon EXTRAPOLATED: rho_ss=0.9277, rho falls 0.970->0.872 over hops 3->10; measured r2_cum(H=10)=0.5069 +/- 0.0284 has NOT crossed the 0.5 half-power point; H_half=11.0 is a one-hop trend-extrapolation, so 'bounded' is inferred by trend continuation, not observed. Insertion loss rho_2=0.937 (one-time, excluded).**

## Instrument checks (pre-registered order, before the A2/B2 read)

1. **Anchor (hop-1 == committed span-1.5/K=0.24)**: mean 0.986353 (SE 0.001284, n=7); target 0.986 +/- 0.0200; deviation +0.000353 -> OK.
2. **Prefix bridge (hops 1-5 == committed Gate-1, digit-exact)**: checked 8 seeds -> ALL REPLAY.
3. **Decoy floors at depth** (intersection p95 means): per-stage {'1': -0.037, '2': -0.034, '3': -0.024, '4': -0.034, '5': -0.051, '6': -0.023, '7': -0.029, '8': -0.036, '9': -0.043, '10': -0.025}, e2e-at-depth -0.4019; bar 0.2 -> clean.
4. **Filter-violation at depth** ([2,9] msg, first repeater [0.2,0.9]): n=7, e2e mean -1.2032 (bar < 0.1 every seed) -> COLLAPSED (sound). Signature rms_in 0.007145, scale 20.5.
5. **ESP-honest paired intersection** (ESP-ok across ALL 10 stages): [0, 1, 2, 3, 4, 5, 7] (n=7).

## Cumulative fidelity r2(m_k, m0) (intersection means +/- SE)

- hop  1: **+0.9864 +/- 0.0013** (per-seed [0.981, 0.988, 0.99, 0.982, 0.99, 0.986, 0.987])
- hop  2: **+0.9238 +/- 0.0228** (per-seed [0.962, 0.898, 0.972, 0.948, 0.797, 0.946, 0.942])
- hop  3: **+0.8958 +/- 0.0214** (per-seed [0.931, 0.884, 0.946, 0.919, 0.775, 0.905, 0.911])
- hop  4: **+0.8656 +/- 0.0194** (per-seed [0.896, 0.86, 0.921, 0.885, 0.76, 0.859, 0.879])
- hop  5: **+0.8137 +/- 0.0233** (per-seed [0.842, 0.819, 0.896, 0.832, 0.693, 0.812, 0.801])
- hop  6: **+0.7654 +/- 0.0239** (per-seed [0.802, 0.768, 0.862, 0.79, 0.66, 0.744, 0.731])
- hop  7: **+0.7067 +/- 0.0232** (per-seed [0.752, 0.722, 0.782, 0.749, 0.609, 0.67, 0.662])
- hop  8: **+0.6482 +/- 0.0222** (per-seed [0.692, 0.669, 0.72, 0.687, 0.561, 0.61, 0.599])
- hop  9: **+0.5796 +/- 0.0234** (per-seed [0.61, 0.621, 0.666, 0.602, 0.489, 0.532, 0.536])
- hop 10: **+0.5069 +/- 0.0284** (per-seed [0.559, 0.554, 0.622, 0.507, 0.443, 0.443, 0.421])

## Steady-state loss law (rho_k, k=3..10; insertion rho_2 EXCLUDED by pre-registration)

- **steady-state (k=3..10) slope = -0.0135 +/- 0.0015** (class B); threshold max(2*SE,0.01) = 0.0100; margin -0.0035 -> trend resolved. Pin B: positive slope beyond +threshold = instrument-suspicion, never A2.
- classification **B**; steady budget rho_ss = 0.9277; measured r2_cum(H=10) = 0.5069 +/- 0.0284; constant-rho_ss budget residual (pred-meas) +0.0229 +/- 0.0246 (DESCRIPTIVE ONLY -- a constant-rho_ss endpoint match is near-tautological for a smooth trend and does NOT adjudicate flat-vs-falling; the pre-registered slope test is the sole discriminator).
- rho by level: {'3': 0.9698, '4': 0.9666, '5': 0.9393, '6': 0.9404, '7': 0.9232, '8': 0.9169, '9': 0.8932, '10': 0.8723}.
- **per-hop error-amplification (compounding) check** (separate from the loss-law verdict above; scoped to per-STAGE amplification, NOT the m0-referenced decline): per-hop reconstruction r2_hop by hop {'1': 0.9864, '2': 0.9864, '3': 0.9828, '4': 0.9862, '5': 0.9816, '6': 0.9848, '7': 0.9865, '8': 0.9877, '9': 0.9876, '10': 0.9851} slope t=1.50 (over hops 2..10, hop-1 anchor excluded) -> error amplification REFUTED (flat r2_hop -- each stage reconstructs its OWN immediate input equally well at any depth; this does NOT refute an m0-referenced decline).
- one-time first-relay INSERTION loss rho_2 = 0.9367 +/- 0.0237 (excluded from the trend statistic).
- H_half = 11.0 (EXTRAPOLATION (r2 did not cross 0.5 within H=10)).

## Scope

Offline H=10 chain; compound span 1.5*H = an INFORMATION-PATH claim, NOT one physical
spectrum. Single variable = depth (no scramble; topology settled at Gate-1). STOP-and-report:
Gate-3 (hop-length) and the mechanism-decomposition gate are separate decisions.
