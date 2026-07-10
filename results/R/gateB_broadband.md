# Relay Gate-B -- broadband trackability mechanism (analysis; retrodiction + locked predictions)

Spec: relay_gateB_broadband_spec.md (sha256 b536a43d66e5). Harness: experiments/relay_gateB.py (sha256 2cf36c7299ab).
Inputs: gate0_bandsweep.json sha256 881c44bee77620fc45218e1c4b34975b7cd8f60d4cd6925b8583e33c3eb6d10e (pinned match: True); gate3_mechanism.json recs sha16 44475823f7f72d4c (pinned match: True).
Wall-clock 0.0 s (CPU).
Provenance: numbers from the stage-1 analysis (harness sha 1147740e1697); verdict RE-RENDERED by --reread (sha 2cf36c7299ab), measurement table asserted byte-identical.

Framing: ANALYSIS gate; stage-1 is a RETRODICTION on committed data (no new integrations). Fit r2 vs band center c = sqrt(lo*hi) on NON-OVERLAP band means only; OVERLAP stratum reported, not gating. Falsification weight rests on the LOCKED P1 predictions (stage-2, separate go) and on Gate-L.

Honesty clause (mandatory): stage-1 is a RETRODICTION -- the sweep means and Gate-3 points are committed, public, and known to the spec author before the spec was drafted. Pre-registration binds the functional forms, parameter counts, weights, strata, dedupe rule, and acceptance thresholds -- NOT data blindness. Falsification weight rests on the locked P1 predictions (stage-2) and on Gate-L. No fit was run before ratification; all numbers of record come from the ratified pipeline.

## Verdict

**EXPLAINED -- broadband trackability is quantitatively attributed to pointwise square-law demodulation + linear-propagation attenuation: best smooth model M1 (M1: SNR=A*c^-p) fits the 6 non-overlap band-points with residuals within max(2*SE,0.03) on 6/6 and within max(2*SE,0.06) on 6/6; wRMSE_CLIFF/wRMSE_M1 = 5.27 >= 1.5. The Phase-3 line-67 erratum becomes DRAFTABLE (drafting/filing in decade_drive is Jason's separate call and waits for Gate-L consistency). LOCKED P1 predictions emitted (falsifiability payload; stage-2 is a separate go).**

## Interpretive note (ships with EXPLAINED; Jason-ratified text)

NOTE (interpretive, ships with EXPLAINED): the winning model is the GENERIC smooth power law (p = 1.04, 2sd [0.34, 1.74]); the mechanism-derived one-pole form (asymptotic slope 2) fits 2.3x worse and p = 2 lies outside the 2sd interval. What this stage establishes is: no cliff and no rate limit -- the best cliff the data allows is a constant (nu_x unidentified above the last point), losing 5.27x -- NOT any specific attenuation law. A single pole is also too crude a derivation here: the slow tertile spans corner frequencies across [1.00, 3.13] (166 oscillators) and the fitted window c = 0.42-4.24 brackets them, where a distributed-corner system shows intermediate effective slope between 0 and 2. The attenuation law remains OPEN; stage-2 tests the M1 power-law interpolation, not a mechanism form. The overlap stratum sits systematically ABOVE the extrapolated fit (all 5 residuals positive, +0.024 to +0.052, broadly increasing with c) -- consistent with an additional direct transfer path when the message band lies inside the injection band, still power-borne per Gate-3 (SUPRA: SQ-carried). Recorded pre-data: H2 may land high in its window. Relevant to Gate-L.

## Band-points (dedupe outcome + strata)

- dedupe: 2 merged (bit-identical), 0 discrepant (reported verbatim below).
- NO-PROTOCOL-DRIFT (free result of the dedupe rule): two DIFFERENT committed harnesses (the gate0-era band sweep and the gate3 mechanism battery) produced BIT-IDENTICAL cells on every shared (band, seed) -- the measurement protocol did not drift between them.
  - band [0.2, 0.9]: sweep[0] == gate3[SUB] bit-identical on seeds [0, 1, 2] -> MERGED (union of seeds).
  - band [2.0, 9.0]: sweep[1] == gate3[RES] bit-identical on seeds [0, 1, 2] -> MERGED (union of seeds).

| center | band | stratum | n | mean r2 | SE | weight | source |
|---|---|---|---|---|---|---|---|
| 0.4243 | [0.200,0.900] | non-overlap | 8 | 0.9865 | 0.0011 | 2500.0 | merged(bit-identical) |
| 1.5000 | [0.707,3.182] | non-overlap | 3 | 0.9368 | 0.0288 | 1209.5 | gate0_bandsweep |
| 2.0672 | [0.974,4.385] | non-overlap | 3 | 0.9397 | 0.0090 | 2500.0 | gate0_bandsweep |
| 2.8489 | [1.343,6.043] | non-overlap | 3 | 0.9073 | 0.0376 | 707.0 | gate0_bandsweep |
| 3.9261 | [1.851,8.328] | non-overlap | 3 | 0.8903 | 0.0238 | 1761.6 | gate0_bandsweep |
| 4.2426 | [2.000,9.000] | non-overlap | 8 | 0.8601 | 0.0169 | 2500.0 | merged(bit-identical) |
| 5.4106 | [2.551,11.478] | OVERLAP | 3 | 0.8609 | 0.0278 | 1294.0 | gate0_bandsweep |
| 7.4566 | [3.515,15.818] | OVERLAP | 3 | 0.8253 | 0.0412 | 590.0 | gate0_bandsweep |
| 10.2761 | [4.844,21.799] | OVERLAP | 3 | 0.7584 | 0.0640 | 244.2 | gate0_bandsweep |
| 14.1618 | [6.676,30.042] | OVERLAP | 3 | 0.6983 | 0.0832 | 144.4 | gate0_bandsweep |
| 16.7332 | [10.000,28.000] | OVERLAP | 8 | 0.6650 | 0.0446 | 503.4 | gate3_mechanism |

## Fits (WLS on non-overlap band means; weights 1/max(SE,0.02)^2; param CIs +/- 2sd, A exp-transformed)

- M1 (2 par): A = 29.86 [12.12, 73.54], p = 1.0429 [0.3441, 1.7417]; wRMSE = 0.0086; chi2_red = 0.21
- M2 (1 par, lambda=0.1 pinned): A = 107.7 [84.17, 137.7]; wRMSE = 0.0201; chi2_red = 0.90
- M-CLIFF (2 par): r_top = 0.9222, nu_x = 4.4548 **[DEGENERATE]**; wRMSE = 0.0453 (step model; no covariance -> no CI). nu_x UNIDENTIFIED above the last center 4.2426 (no collapse in range; the fit degenerates to a constant r_top and the printed nu_x is an arbitrary point in the degenerate region)
- best smooth = **M1**; wRMSE_CLIFF / wRMSE_M1 = **5.27** (EXPLAINED needs >= 1.5; CLIFF-PREFERRED needs < 1).

## Residuals vs best smooth (M1; acceptance: tight max(2*SE,0.03) on >=5/6, loose max(2*SE,0.06) on 6/6)

| center | n | mean | fit | resid | tol_tight | tol_loose | tight | loose |
|---|---|---|---|---|---|---|---|---|
| 0.4243 | 8 | 0.9865 | 0.9865 | -0.0000 | 0.0300 | 0.0600 | ok | ok |
| 1.5000 | 3 | 0.9368 | 0.9514 | -0.0145 | 0.0575 | 0.0600 | ok | ok |
| 2.0672 | 3 | 0.9397 | 0.9333 | +0.0063 | 0.0300 | 0.0600 | ok | ok |
| 2.8489 | 3 | 0.9073 | 0.9093 | -0.0019 | 0.0752 | 0.0752 | ok | ok |
| 3.9261 | 3 | 0.8903 | 0.8776 | +0.0127 | 0.0477 | 0.0600 | ok | ok |
| 4.2426 | 8 | 0.8601 | 0.8687 | -0.0086 | 0.0338 | 0.0600 | ok | ok |

- tight 6/6, loose 6/6.

## OVERLAP stratum (reported, NOT gating; model extrapolated beyond its fit domain)

- c = 5.4106 (sweep[6], n=3): mean 0.8609, extrapolation 0.8369, resid +0.0239
- c = 7.4566 (sweep[7], n=3): mean 0.8253, extrapolation 0.7860, resid +0.0393
- c = 10.2761 (sweep[8], n=3): mean 0.7584, extrapolation 0.7245, resid +0.0340
- c = 14.1618 (sweep[9], n=3): mean 0.6983, extrapolation 0.6530, resid +0.0453
- c = 16.7332 (gate3[SUPRA], n=8): mean 0.6650, extrapolation 0.6126, resid +0.0524

## LOCKED P1 predictions (model M1; locked at stage-1, before any stage-2 go; no threshold moves after data)

Intervals: sigma_pred = sqrt(g^T Cov g + 0.02^2) -- fit covariance plus the pinned SE floor ONLY; NO model-misspecification term. H2 is an extrapolation and H3 a width-transfer claim: their intervals quantify fit uncertainty, not model error -- that is exactly what stage-2 tests.

- **H1**: c = 3.3000, rho = 4.5, band = [1.5556, 7.0004] (non-overlap INTERPOLATION: in the fit domain (c bracketed by fitted centers 2.85 and 3.93, rho=4.5); the band itself was never swept = out-of-SAMPLE) -> pred r2 = **0.8958 +/- 0.0229** (displays; the json records full precision, byte-locked)
- **H2**: c = 12.0000, rho = 4.5, band = [5.6569, 25.4558] (overlap-stratum test: EXTRAPOLATION beyond the fitted c-range [0.42,4.24] and outside the fitted stratum (message overlaps the injection band)) -> pred r2 = **0.6910 +/- 0.0981** (displays; the json records full precision, byte-locked)
- **H3**: c = 4.2426, rho = 1.5, band = [3.4641, 5.1962] (NARROW band at a swept center (width-dependence discriminant; the fit only saw rho=4.5, and M1/M2 have NO width dependence -- this prediction equals the model value at c=4.2426 regardless of rho; width-INdependence is the claim under test)) -> pred r2 = **0.8687 +/- 0.0255** (displays; the json records full precision, byte-locked)

- Stage-2 scoring (pinned now): PREDICTIVE PASS iff |mean_obs - pred_r2| <= max(2*sigma_pred, 0.05) for ALL THREE bands (conjunction), EVALUATED AT STAGE-2 from this formula and the byte-locked full-precision (pred_r2, sigma_pred) -- windows are NOT stored. Any other pattern: report as-is. No threshold moves after data.

## Scope

Retrodiction on committed data; the fit only constrains the NON-OVERLAP stratum (rho=4.5, c in [0.42, 4.24]). H1 is in-domain interpolation on a never-swept band (out-of-sample); H2 extrapolates beyond the fitted c-range and stratum; H3 transfers the fit to a narrower band (the models claim width-independence). The OVERLAP extrapolations and P1 are what the stage-2 probe (separate go) and Gate-L can falsify. No erratum files from this gate alone.
