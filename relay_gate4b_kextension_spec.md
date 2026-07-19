# Relay Program -- Gate 4b Spec: K-extension (de-censor K*(s))

Status: Ratified 2026-07-19 by Jason via the roadside paste
(HANDOFF_CC_roadside_20260719, W2); taste calls T1-T4 ratified as
defaulted, override window open until battery-go.
Builds on: Gate-4 CLOSED (SHORT-WINS, scoped AT-CEILING
right-censored: K*(s) = 0.24 = grid max at every span, gradient
asymmetry ~20-60x) + the committed Phase-1 (span x K) landscape
(decade_drive b0f7664, the file whose full sha256 is recorded in
the committed Gate-4 provenance -- assert against that record).
Gate type: MEASUREMENT gate, single-hop Phase-1-protocol routing
cells. No chains. No contrast verdict: Gate-4's SHORT-WINS and D
stand as committed and are not re-litigated here. STOP-and-report.

## Objective (one sentence)
De-censor the K*(s) lookup: extend the coupling grid above the
committed ceiling K=0.24 and locate, per span, the true argmax of
single-hop routing fidelity -- or the constraint that binds first.

## Design
Cells: Phase-1 protocol single-hop r2_d0 per (span, K, seed),
committed mechanics (D_phase1_routing.py + core/reservoir.py path
as used by the relay gates), one GPU process, float64/x64.
Spans (T2, default ALL FOUR -- K-rule uniformity, Gate-4 edit 5,
no special-casing): 1.0, 1.5, 2.0, 3.0.
Extension grid (T1 default): K in {0.28, 0.32, 0.40, 0.48} -- one
half-step point just above the old ceiling to resolve curvature,
then 0.08 steps to twice the committed ceiling.
Union grid for all argmax statements: the committed K_GRID
(0.0, 0.08, 0.12, 0.16, 0.24) UNION the extension grid; committed
rows enter by their committed values (never re-run except the
anchor row), extension rows by this gate's cells.
Seeds (T4 default): 0-9, matching the committed landscape
statistic (mean over seeds 0-9).
Anchor row: re-run ALL FOUR committed (span, K=0.24) rows, seeds
0-9 (40 cells), gated digit-exact 6dp vs the committed landscape
per-seed values; bit-exact retained as DIAGNOSTIC only (the
span-2.0 ULP reduction-order precedent is on record; 6dp gates).
Compute: 4 spans x 4 new K x 10 seeds = 160 new cells + 40 anchor
cells; CC prints a wall-clock estimate at sandbox from the Gate-4
per-cell timing before smoke.

## Pre-registered readout (no post-hoc moving)
Primary statistic per (span, K): the COMMITTED lookup statistic,
ported verbatim from the committed relay_gate4.py K-rule
implementation (RAW mean of r2_d0 over seeds 0-9; NO ESP-gating in
the rule; argmax tie -> first/lowest K; NOISE_FLOOR 0.05 for the
noise-argmax flag). The port is proven at sandbox by replaying the
committed K*(s) table from the committed landscape and matching
the committed gate4_hoptrade.json k_star block digit-exact.
Reported alongside, NON-GATING (T3): the ESP-honest mean (ESP-ok
seeds only, membership printed). If the two argmax locations
disagree at any span, the .md prints DIVERGENT-ARGMAX for that
span with both locations; interpretation is Jason's at
read-through.
Per-span verdict classes (union-grid argmax K_hat(s)):
  DE-CENSORED(s):    K_hat < max(extension grid). Includes
                     K_hat = 0.24 now-interior. Report K_hat,
                     margin over both neighbors, and the full
                     mean-vs-K table.
  STILL-CENSORED(s): K_hat = max(extension grid) (right-censored
                     at the NEW ceiling). Further extension is a
                     future call, not this gate's.
  NOISE-ARGMAX(s):   ported unchanged from Gate-4 edit 5 (winning
                     |mean| < 0.05).
Suffixes (per span, appended where triggered):
  -TIE-WITHIN-SE: the top two union-grid means differ by less
     than 1 SE of their per-seed paired difference (extension
     points; committed-vs-extension ties use the extension row's
     per-seed values against the committed per-seed values, ESP
     construction permitting, else flagged UNPAIRABLE-TIE);
     the tied K set prints. Evaluate-at-use, never stored.
  -ESP-DEGRADED: K_hat lands on a point where >= 5 of 10 seeds
     fail ESP ok_slow (eps 1e-2). Any such point in the table is
     marked ESP-DEGRADED regardless of K_hat. If the
     ESP-DEGRADED boundary is what stops the climb (all K above
     some K_e degraded and K_hat at or below K_e), the .md states
     ESP-BOUNDED-CEILING as an observation for that span.
  -UNSTABLE-ARGMAX: leave-one-seed-out (10 drops) moves K_hat for
     any drop; the drop set and alternate locations print
     (location-statistic LOSO analog; observation, not a gate).
NM (gate-level): any anchor-row 6dp miss; any decoy elevated
(p95 > 0.2); the sandbox lookup-replay mismatching the committed
k_star block; any extension point with < 2 recorded seeds.
Per-point n prints always; no point below n=10 without a named
attrition cause.
Consequence map (reported, no automatic actions): DE-CENSORED
feeds the Gate-4 scope note's forecast (compression above ceiling)
and future S1 revision -- NO retro-edit of Gate-4;
STILL-CENSORED banks the extend-or-stop call; ESP-BOUNDED-CEILING
is a physical-ceiling observation relevant to GC and GM design.
No mechanism claims from this gate.

## Instruments
- Lookup replay (sandbox, pre-GPU): recompute the committed K*(s)
  table from the committed landscape blob (sha-asserted against
  the committed Gate-4 provenance record, full value) and match
  the committed gate4_hoptrade.json k_star block digit-exact.
- Anchor row as above (gate 6dp; bit-exact diagnostic).
- Fresh decoys: one family per span, bases 700000, 720000,
  740000, 760000, family span per the committed footprint rule;
  FULL collision matrix vs all 26 committed families (census and
  max committed footprint per the Gate-L record) proven at
  sandbox before any GPU. Per-cell decoy p95, bar 0.2.
- ESP nested ok_slow per cell, eps 1e-2.
- Locked-numbers reread (--reread re-decides from unchanged recs,
  loud-fail on numeric drift); NM-shape self-test; no-truncation
  render asserts; declaration windows on mode and time axes;
  evaluate-at-use for every derived quantity (tie-SE, margins);
  verdict-engine synthetic test covering every class x suffix
  branch, proven to fire.
- Sandbox persists per-check fire-evidence structures (the Gate-L
  panel note, adopted): each check stores its comparison payload,
  not a bare pass flag.

## Deliverables
results/R/gate4b_kextension.json (per-cell r2_d0/ESP/decoy, union
tables, per-span classes + suffixes, k_star_extended block,
committed-lookup replay evidence, env, sha chain, reread) + short
.md (anchor report, decoy max cell, per-span mean-vs-K tables,
class table, ESP boundary map, consequence lines). ASCII-only.

## Pipeline
CPU sandbox -> GPU-light smoke (seed-0: all four anchor cells
digit-exact 6dp + one extension cell per span logged) -> HARD
STOP -> battery on Jason's explicit word ("G4B BATTERY GO",
T-overrides may attach) -> STOP -> panel + read-through -> commit
only on Jason's exact word. Author Jason Dury <jason@eridos.ai>,
no co-author. Committed artifacts and core/ untouched.
