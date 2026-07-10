# Relay Program -- Gate B Spec: Broadband Trackability Mechanism
# (analysis-first, retrodiction + locked predictions)

Status: Drafted (Claude). Ready for ratification + CC handoff.
Date: 2026-07-10
Builds on: Gate-0 band-sweep NO-BAND-FOUND (gate0_bandsweep.json) +
Gate-3 CLOSED d1a7116 (B2 SCOPED; "broadband side-finding needs
another explanation"). Precedes Gate-4 (independent) and Gate-L.
Gate type: ANALYSIS gate. Stage-1 is CPU-only on committed data --
no new integrations. Stage-2 is an optional 6-run GPU probe on a
separate go. STOP-and-report.

## Objective (one sentence)
Determine whether broadband trackability (smooth monotone r2 decline,
no band untrackable within Nyquist) is quantitatively explained by a
minimal "pointwise square-law demodulation + linear-propagation
attenuation" model -- and emit LOCKED held-out-band predictions that
make the explanation falsifiable rather than merely retrodictive.

## Inputs and strata (committed data only)
- gate0_bandsweep.json: 10 bands x 3 seeds, K=0.24 (record its sha).
- gate3_mechanism.json (d1a7116, recs sha 44475823f7f72d4c): FULL r2,
  3 bands x 8 seeds.
- Band coordinate: c = geometric center sqrt(lo*hi) (the committed
  convention; CC re-verifies against stored centers).
- DEDUPE RULE (pre-registered): where sweep and Gate-3 measured the
  same band with overlapping seeds, bit-compare the cells. Identical
  -> merge as one band-point (union of seeds). Not identical -> both
  enter as separate measurements of that band AND the discrepancy is
  reported verbatim (free protocol-drift detector).
- Strata by the committed overlaps_fast_tertile flag:
  NON-OVERLAP (primary, fit + acceptance): centers ~0.42, 1.50, 2.07,
  2.85, 3.93, 4.24 (6 band-points after dedupe).
  OVERLAP (reported, not gating): ~5.41, 7.46, 10.28, 14.16, 16.73
  (message inside the injection band; different geometry).

## Models (pre-registered; fit r2 vs c on band means)
- M1 (generic smooth, 2 params): SNR(c) = A * c^(-p),
  r2 = SNR/(1+SNR). Prior slope range p in [0.5, 4].
- M2 (mechanism-derived, 1 param): SNR(c) = A / (4*lambda^2 + c^2),
  lambda = 0.1 pinned from committed config (SL amplitude-mode
  one-pole; corner 2*lambda = 0.2).
- M-CLIFF (incumbent rate-limit picture, 2 params): r2 = r_top for
  c < nu_x, else 0.05.
Fit: weighted least squares on band-mean r2; weight_b =
1/max(SE_b, 0.02)^2 (SE from per-seed spread; floor pinned so
high-n Gate-3 points cannot dominate). No other knobs.

## Acceptance and consequence map (pinned now)
- EXPLAINED: best of {M1, M2} has non-overlap residuals within
  max(2*SE_b, 0.03) on >=5 of 6 band-points AND within
  max(2*SE_b, 0.06) on 6 of 6, AND RMSE_CLIFF >= 1.5 x RMSE_best.
  -> Broadband finding attributed to pointwise demodulation +
  attenuation. The Phase-3 line-67 erratum ("messages must live
  slower than every band they traverse") becomes DRAFTABLE --
  drafting/filing in decade_drive is Jason's separate call, and
  waits for Gate-L consistency. Locked predictions emitted.
- SHAPE-UNEXPLAINED: acceptance fails -> report residual structure
  verbatim; no erratum; Gate-L proceeds regardless.
- CLIFF-PREFERRED: RMSE_CLIFF < RMSE_best_smooth -> the rate-limit
  picture partially survives; report; escalate to targeted sweep
  design. (Given the committed worst cell is 0.54 vs cutoff 0.1,
  this branch is expected dead -- it stays in the map because
  expected-dead branches get written down, not assumed.)
- NO-MEASUREMENT: input artifacts fail sha/digit re-verification;
  dedupe bit-compare cannot be resolved; fit non-convergence.

## Locked predictions (P1 -- written into the stage-1 artifact
## BEFORE any stage-2 go; the falsifiability payload)
Predicted r2 with intervals (fit covariance + SE floor) emitted at
stage-1 for three held-out bands whose geometry is pinned NOW.
NORMATIVE quantities are (center-derivation, rho) ONLY; band edges
are computed in the harness as [c/sqrt(rho), c*sqrt(rho)]; all
decimal displays below are for reading and carry no authority; the
json records full precision.
  H1: c := 3.30 exactly (author-chosen interpolation point, derived
      from no committed quantity), rho=4.5
      -> band [1.5556, 7.0004]  (non-overlap interpolation).
  H2: c := 12.0 exactly (author-chosen, derived from no committed
      quantity), rho=4.5
      -> band [5.6569, 25.4558]  (overlap-stratum test).
  H3: c := sqrt(2*9) = 4.2426406871... (the committed orig_violation
      swept center, BY EXPRESSION), rho=1.5
      -> band [3.4641, 5.1962]  (NARROW band at a swept center --
      width-dependence discriminant; the fit only ever saw rho=4.5).

## Stage-2 probe (optional; SEPARATE GPU go from Jason)
2 seeds x {H1, H2, H3} = 6 stage-A runs (~90 s), Phase-1 protocol,
fresh collision-proven decoy bases, anchor arm not required (no
committed reference exists for these bands; the instrument chain is
carried by protocol identity + decoys). PREDICTIVE PASS iff
|mean_obs - pred| <= max(2*sigma_pred, 0.05) for all three.
Any other pattern: report as-is. No threshold moves after data.

## Honesty clause (mandatory, ships in the .md)
Stage-1 is a RETRODICTION: the sweep means and Gate-3 points are
committed, public, and known to the spec author before this spec.
Pre-registration binds the functional forms, parameter counts,
weights, strata, dedupe rule, and acceptance thresholds -- not data
blindness. Falsification weight rests on P1/stage-2 and on Gate-L.
No fit was run before ratification; all numbers of record come from
the ratified pipeline.

## Instruments / sandbox (CC, before stage-1 numbers)
Synthetic-recovery: WLS recovers planted {A,p} and planted cliffs;
selection rule fires both ways on synthetic curves. Dedupe
bit-compare proven on known-identical and known-different cells.
Center convention re-verified against stored values. ASCII-only
verdict; amend-not-overwrite via --reread as in Gate-3.

## Deliverables
results/R/gateB_broadband.json (input shas, strata, dedupe outcome,
fits, params +/- CI, residual table, model comparison, LOCKED P1,
consequence line) + short .md. STOP-and-report. Stage-2 on its own
go; commit only on Jason's exact word, author Jason Dury
<jason@eridos.ai>, no co-author.

## Compute
Stage-1: CPU, minutes. Stage-2 (optional): ~90 s GPU, 6 runs.
