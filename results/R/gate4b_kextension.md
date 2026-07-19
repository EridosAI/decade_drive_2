# Relay Gate-4b -- K-extension (de-censor K*(s))

Spec: relay_gate4b_kextension_spec.md (sha12 09f63de5249f). Harness sha12 ecdd24154fab.
Verdict: **GATE-4B MEASURED**

## Instrument checks (gate-level)

- lookup-replay (committed K-rule, digit-exact vs gate4 k_star): OK
- anchor row (span,K=0.24 seeds 0-9, 6dp gate): OK
- fresh decoys (26-family census clear; per-cell p95 <= 0.20): OK (max p95 0.0676)
- thin-intersection (every extension point n >= 2): OK

## Anchor (digit-exact 6dp gate; bit-exact DIAGNOSTIC)

- span 1.0: digit6 OK | bit-exact 10/10 (diagnostic)
- span 1.5: digit6 OK | bit-exact 10/10 (diagnostic)
- span 2.0: digit6 OK | bit-exact 10/10 (diagnostic)
- span 3.0: digit6 OK | bit-exact 10/10 (diagnostic)

## Per-span mean-vs-K (union grid) + class

### span 1.0 -- STILL-CENSORED (K_hat=0.48)
```
K       0.00  0.08  0.12  0.16  0.24  0.28  0.32  0.40  0.48
mean  -1.02   0.89   0.98   1.00   1.00   1.00   1.00   1.00   1.00
n        10     10     10     10     10     10     10     10     10
ESPdeg  deg     .      .      .      .      .      .      .      . 
```
- margin over left neighbor K=0.4: 0.0000

### span 1.5 -- STILL-CENSORED (K_hat=0.48)
```
K       0.00  0.08  0.12  0.16  0.24  0.28  0.32  0.40  0.48
mean  -1.71   0.73   0.88   0.94   0.99   0.99   1.00   1.00   1.00
n        10     10     10     10     10     10     10     10     10
ESPdeg  deg    deg     .      .      .      .      .      .      . 
```
- margin over left neighbor K=0.4: 0.0006

### span 2.0 -- STILL-CENSORED (K_hat=0.48)
```
K       0.00  0.08  0.12  0.16  0.24  0.28  0.32  0.40  0.48
mean  -4.67   0.41   0.53   0.61   0.74   0.79   0.83   0.89   0.93
n        10     10     10     10     10     10     10     10     10
ESPdeg  deg    deg    deg     .      .      .      .      .      . 
```
- margin over left neighbor K=0.4: 0.0395

### span 3.0 -- NOISE-ARGMAX-UNSTABLE-ARGMAX (K_hat=0.32)
```
K       0.00  0.08  0.12  0.16  0.24  0.28  0.32  0.40  0.48
mean  -27.22  -0.01  -0.01  -0.00  -0.00  -0.00  -0.00  -0.00  -0.00
n        10     10     10     10     10     10     10     10     10
ESPdeg  deg    deg    deg    deg     .      .      .      .      . 
```
- margin over left neighbor K=0.28: 0.0002
- margin over right neighbor K=0.4: 0.0005
- DIVERGENT-ARGMAX: RAW K_hat=0.32 vs ESP-honest K_hat=0.16
- UNSTABLE-ARGMAX (LOSO moves): {"8": 0.4}

## Fresh decoy (max cell)
- {"span": 2.0, "K": 0.4, "seed": 2, "decoy_p95": 0.06756643026135896} (bar 0.20)

## Consequence map (reported; no automatic actions)

- span 1.0: STILL-CENSORED (K_hat=0.48) -- banks the extend-or-stop call (right-censored at the NEW ceiling 0.48); further extension is a future call.
- span 1.5: STILL-CENSORED (K_hat=0.48) -- banks the extend-or-stop call (right-censored at the NEW ceiling 0.48); further extension is a future call.
- span 2.0: STILL-CENSORED (K_hat=0.48) -- banks the extend-or-stop call (right-censored at the NEW ceiling 0.48); further extension is a future call.
- span 3.0: NOISE-ARGMAX-UNSTABLE-ARGMAX (K_hat=0.32) -- winning |mean| < 0.05; context-only, no de-censoring claim.
- span 3.0: DIVERGENT-ARGMAX -- RAW vs ESP-honest argmax disagree; interpretation is Jason's at read-through.

## Provenance
- committed landscape: phase1_routing.json sha256 2e739315141e88c3c5c698f88ed6f84efaae46f7257397c756f58ee4c3965590 (decade_drive b0f7664)
- fresh decoy bases: {"1.0": 700000, "1.5": 720000, "2.0": 740000, "3.0": 760000} | family span 1859 | 26 committed families censused
- wall_clock_s: 8916.0
- env: {"python": "3.12.3", "numpy": "2.4.6", "jax": "0.10.1", "jax_enable_x64": true, "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39", "JAX_PLATFORMS": "<default>", "CUDA_VISIBLE_DEVICES": "<unset>"}
