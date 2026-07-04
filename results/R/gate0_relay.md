# Relay Gate-0 -- full-gate record (Stage 3)

Spec: relay_gate0_spec.md (sha256 a9930793e43c). Harness: experiments/relay_gate0.py (sha256 dd3aa0044e66, uncommitted at run time).
Provenance anchors: decade_drive b0f7664 (Phase-1). Seeds run: [0, 1, 2, 3, 4, 5, 6, 7]. K primary = 0.24. Wall-clock 62 min.

## Verdict

**NO-MEASUREMENT (violation control did NOT collapse: viol=0.664 vs compliant=0.926 -- envelope-of-envelope bookkeeping wrong or check underpowered)**

## Instrument checks (evaluated before the verdict, pre-registered order)

1. **Anchor**: stage-A intersection mean = 0.986421 (SE 0.001289, n=7); target 0.986 +/- 0.0200; deviation +0.000421 -> OK.
2. **Decoy floors** (intersection means): stage-A p95 -0.0371, stage-B p95 -0.0309, e2e p95 -0.4238; elevated bar 0.2 -> clean.
3. **Bandwidth-violation control** (MSG_BAND [2.0,9.0] rad/s vs standard [0.2,0.9]; n=7): violation e2e mean +0.6636 vs compliant +0.9264 (collapse bar: <= max(0.5 x compliant, 0.1)) -> DID NOT COLLAPSE. Violation message = the compliant seed's draw time-compressed 10x into [2.0,9.0] (paired scheme: same generator seed, same carrier).
4. **ESP-honest paired intersection**: [0, 1, 2, 3, 4, 5, 6] (n=7; a seed failing ANY of relay-A/relay-B/direct is dropped everywhere).

## Verdict-row numbers (K=0.24, intersection means +/- SE)

- relay end-to-end r2(m2,m0): **+0.9264 +/- 0.0233** (per-hop: stage-B r2 +0.9862)
- fresh direct span-3.0: **-0.0017 +/- 0.0064**
- paired delta e2e - direct: +0.9281 (sd 0.0498) -> beats: True
- paired delta e2e - e2e-decoy-p95: +1.3501 (sd 0.2150) -> beats: True

## Scramble robustness line (characterisation only, never verdict)

- scrambled-stage-A relay e2e = +0.9264 +/- 0.0229 vs compliant +0.9264 +/- 0.0233 (n=7) -- staging topology-generic (consistent with Phase-1) (3-sigma/0.1 bar = code-level pre-run operationalization; the spec clause is qualitative).

## Secondary K row (K=0.16 bracket -- optional, non-verdict)

- relay e2e +0.8344 +/- 0.0219 vs direct +0.0095 +/- 0.0104 (n=2; stage-A anchor row mean +0.9276) -- bracket line only, never verdict.

## Replication table (measured vs committed b0f7664, 6dp)

32/32 rows match to 6dp. Non-matching rows (if any):
- none -- every measured Phase-1 row reproduces the committed value.

## Scope

Offline two-stage square-law relay; compound span 3.0 = a claim about the
INFORMATION PATH (two successive square-law demodulations end-to-end), not one
physical spectrum (spec 'Honest framing'). STOP-and-report: no sweep, no multi-hop
extension, no interpretation beyond the pre-registered outcome mapping.
