# decade_drive_2 -- Relay Program

Can **engineered staging** carry information across a frequency span that a single
passive hop cannot? This program tests the shipped prediction of the Decade-Drive
Phase-3 document (Section 6, Branch B): an **offline two-stage square-law relay**
transfers usable information across a compound 3-decade information path where
direct passive transfer is dead.

A new program from a clean baseline -- successor to
[decade_drive](https://github.com/EridosAI/decade_drive) per its Terminus, not a
reopening of it.

## Background (what Decade-Drive established)

In a multi-decade Stuart-Landau oscillator reservoir with a message injected into
the fast band and readout restricted to the slow band, cross-band routing runs
through the **|z|^2 square-law channel**, falls with frequency separation, and
**dies past ~2 decades** (committed Phase-1, ESP-robust @ K=0.24: r^2 =
1.000 / 0.986 / 0.751 / 0.162 / -0.003 at spans 1.0 / 1.5 / 2.0 / 2.5 / 3.0).
Populating the intervening spectrum with passively coupled oscillators is not a
relay -- the horizon is a property of passive multi-decade coupling itself. The
remaining route past it is **active staging**: explicit per-stage square-law
readout and reinjection.

## The gate (Gate 0 -- offline two-stage relay)

See [`relay_gate0_spec.md`](relay_gate0_spec.md). Offline, cheap, zero new core
machinery; STOP-and-report, no sweep.

```
stage A (span 1.5) --m1--> repeater (band-limit + rescale) --m1'--> stage B (span 1.5) --> m2
```

- **Stage A** = a Phase-1 replica (N=500, span 1.5, lambda=0.1, ER mean-degree 10,
  beta=1.0, SPP=2, K=0.24). AM message m0 in the fast tertile; slow-tertile |z|^2
  readout reconstructs m1.
- **Replication anchor (built-in bridge to the committed record):** stage A must
  reproduce the committed Phase-1 value **0.986** --
  `|mean - 0.986| <= max(2*SE, 0.02)` -- or the gate is NO-MEASUREMENT. The 0.02
  window is set to exclude 0.945 (the K=0.16 value at this span), so reproducing
  the wrong operating point still fails.
- **Repeater** = band-limit m1 to MSG_BAND and affine-rescale to m0 statistics
  (the detect-filter-remodulate step, made explicit and logged).
- **Stage B** = fresh network, same construction, takes m1' as its injected
  message; slow-tertile readout gives m2.
- **Verdict** = r^2(m2, m0) vs a **fresh, paired direct span-3.0** run (its floor
  ~ 0) and per-stage decoys, on the ESP-honest paired seed intersection. Naive
  expectation ~0.97 (0.986^2). PASS => the horizon is architectural (a routing
  budget); FAIL => it binds even active staging (a sharper negative).

"Compound span 3.0" is a claim about the **information path** -- the message
survives two successive square-law demodulations end-to-end -- not about one
physical spectrum. The comparison target is the direct 3-decade hop, re-run fresh
in the same batch.

## Status

| Unit | What | State |
|---|---|---|
| Gate 0 | Offline two-stage square-law relay probe | **spec'd** (`relay_gate0_spec.md`); not yet run |
| -- implementation | CPU sandbox -> 1-seed smoke -> full gate | pending |

## Provenance

The Phase-1 routing machinery is copied **byte-identical** from decade_drive (the
transitive import closure of `experiments/D_phase1_routing.py`): the
co-rotating-frame integrator, reservoir / graph-Laplacian builders, ESP
(consistency) gate, log-frequency band partition, and -- inline in
`D_phase1_routing.py` -- the AM message generator, decoy machinery, ridge+CV |z|^2
demod-R^2 metric, and the degree-matched scramble control. The anchor requirement
bridges every relay run back to decade_drive's committed `b0f7664`
(span-1.5 @ K=0.24 = 0.986).

## Layout

```
core/         reservoir, magnus, integrator_corotating, consistency, bands, stuart_landau
experiments/  D_phase1_routing.py   (Phase-1 machinery; AM/decoy/demod/scramble inline)
results/R/    Gate-0 outputs (gate0_relay.json + a .md verdict record)
relay_gate0_spec.md   the pre-registered Gate-0 design
```

## Environment

Python 3.12+, `jax` + `diffrax` (CUDA; RTX 4080 Super), **float64 required** --
`jax.config.update("jax_enable_x64", True)` at startup. One GPU process at a time.

## Standing rules

- Gate-first; STOP-and-report at every checkpoint; single-variable discipline
  (K fixed at 0.24 for the verdict row); probe bounded by the reservoir (N=500).
- Verify load-bearing claims in a CPU sandbox before accepting numbers --
  specifically the repeater transform, the decoy-protocol match to Phase-1, and
  the paired-intersection logic.
- The curated verdict record is a committable `.md` (verbose run-logs are `.log`,
  gitignored).
- Do not edit `core/integrator_corotating.py` before the replication anchor
  passes -- it is the load-bearing bridge to the committed 0.986.

## License

MIT -- see [LICENSE](LICENSE).
