# Relay Gate-B stage-2 probe -- locked-P1 falsifiability test

Wall-clock 98s. Protocol: span 1.5, K=0.24, stage-A slow-tertile demod (reproduces the committed band-sweep exactly, by imported code path). Seeds: 0, 1.

ESP: the stage-A trackability protocol carries NO ESP replica by construction -- it reuses relay_gate0._stage_a_r2 exactly as the committed band-sweep (no repeater, no replica), so there is no per-seed ESP flag here. The specificity instrument is the never-injected same-band decoy null (reported per band below).

Framing: Gate-B stage-2 probe: reproduce the committed band-sweep protocol (span 1.5, K=0.24, stage-A only, slow-tertile demod) at the three HELD-OUT bands and test the LOCKED P1 predictions. Falsifiable, not retrodictive: pred_r2/sigma_pred were byte-locked at stage-1 (committed) BEFORE any of these runs. Windows are NOT stored -- the pass rule max(2*sigma_pred, 0.05) is evaluated here from the byte-locked (pred, sigma).

## Verdict

**PREDICTIVE PASS -- all three held-out bands land within max(2*sigma_pred, 0.05) of the locked M1 prediction. The stage-1 attenuation explanation (smooth M1 power law) is CORROBORATED out-of-sample; the Phase-3 line-67 erratum path stays open pending Gate-L.**

## Protocol-identity anchor (instrument)

- compliant [0.2,0.9] seed 0: reproduced 0.9814697253 vs committed 0.9814697253 (diff 0.0e+00) -> digit-exact(6dp) OK; bit-exact True (diagnostic). The holdout hops use the SAME imported protocol -> protocol identity certified.

## Held-out predictions (locked at stage-1, tested here; windows evaluated, not stored)

- **H1** band [1.556,7.000] -- non-overlap INTERPOLATION: in the fit domain (c bracketed by fitted centers 2.85 and 3.93, rho=4.5); the band itself was never swept = out-of-SAMPLE: mean_obs = +0.902693 (per-seed seed 0: +0.887718, seed 1: +0.917669) vs pred +0.895793 +/- 0.0229; signed dev +0.0069, |dev| = 0.0069 <= window max(2*sigma,0.05) = 0.0500 -> **PASS** (decoy p95 -0.0583).
- **H2** band [5.657,25.456] -- overlap-stratum test: EXTRAPOLATION beyond the fitted c-range [0.42,4.24] and outside the fitted stratum (message overlaps the injection band): mean_obs = +0.711588 (per-seed seed 0: +0.604314, seed 1: +0.818863) vs pred +0.691036 +/- 0.0981; signed dev +0.0206, |dev| = 0.0206 <= window max(2*sigma,0.05) = 0.1961 -> **PASS** (decoy p95 -0.0411).
- **H3** band [3.464,5.196] -- NARROW band at a swept center (width-dependence discriminant; the fit only saw rho=4.5, and M1/M2 have NO width dependence -- this prediction equals the model value at c=4.2426 regardless of rho; width-INdependence is the claim under test): mean_obs = +0.908909 (per-seed seed 0: +0.877132, seed 1: +0.940687) vs pred +0.868674 +/- 0.0255; signed dev +0.0402, |dev| = 0.0402 <= window max(2*sigma,0.05) = 0.0510 -> **PASS** (decoy p95 -0.0424).

## Pattern observations (observed, not claimed; folded to Gate-L)

- **H2 callback (a pre-registered expectation met).** Stage-1 pre-flagged the overlap stratum as sitting ABOVE the fit ("H2 may land high in its window"). It did: mean_obs is +0.0206 above the locked prediction (well within the wide window). Surfaced as the predicted success it is.
- **Sign pattern.** All three deviations are POSITIVE (+0.0069 / +0.0206 / +0.0402); 3-of-3 same-sign is P=0.25 under symmetric independence -- NOT significant, claimed as nothing -- but consistent with the known overlap-positive structure and an M1 curve sitting slightly low. H3's direction (a narrower rho=1.5 band decoding slightly better than the width-independent model predicts) is physically sensible and noise-compatible at n=2. Gate-L inherits the observation.

## Scope

Falsifiability payload for the stage-1 EXPLAINED verdict: the M1 power-law interpolation is tested out-of-sample at three held-out bands. This tests the M1 INTERPOLATION, not any specific attenuation mechanism (the law stays open; see stage-1). Relevant to Gate-L and the Phase-3 line-67 erratum path. STOP-and-report.
