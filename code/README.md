# GO-DynURPlanner — M0 core

Analytic parameterized space (URPlanner lineage) with **continuous-time
collision ground truth** for dynamic environments. This is the M0 milestone
of the Loop-6 research blueprint (`../research-log/06_research_blueprint.html`).

## What's here

| Module | Content |
|---|---|
| `godynur/panda.py` | Franka Emika Panda modified-DH FK (Craig convention, matches URPlanner Eq.4), URPlanner 4-segment link consolidation, corrected chord-error bound ε_lin (conservativeness lemma term) |
| `godynur/geometry.py` | Static segment-vs-AABB overlap (URPlanner Eq.6–10), UOAR |
| `godynur/continuous.py` | Exact continuous-time checker over a control interval: breakpoint enumeration (linear/quadratic roots of the slab rationals), first-contact time τ\*, swept-overlap integral (per-piece Gauss–Legendre), lemma inflation |
| `tests/` | 20 tests: FK structural properties, overlap vs brute-force sampling, τ\* vs dense time scan, **tunneling regression case** (discrete endpoint check passes, continuous checker catches the crossing), swept-integral vs dense quadrature, Δt→0 reduction to static UOAR, conservativeness spot checks |

## Run

```bash
cd code && python3 -m pytest tests/ -q
```

Requires `numpy`, `pytest`. No simulator, no GPU — everything is analytic,
consistent with URPlanner's platform-agnostic training philosophy.

## Verified parameters (from URPlanner full text, 2026-07-06)

- Modified DH: `T = Rot(α)·Trans(a)·Rot(q)·Trans(d)` (paper Eq.4)
- Link consolidation for Panda: pairs (1,2),(3,4),(5,6),(7,flange) → 4 segments
- Safety offset `a_o = 5 cm` (paper §VII-C; practical range 2.5–5 cm)
- UOAR weight ζ=1; APE2 M=2, N=3 (Table I)
- Link cylinder radius `a_r`: **not stated numerically in the paper text** —
  treated as a config parameter, to be aligned with the group's setup.

## A correction the tests caught (important for the paper's lemma)

The blueprint's first-draft chord bound `Σᵢ Rᵢδθᵢ²/8` is **invalid** — it
drops the Hessian cross terms between simultaneously moving joints
(empirical counterexample: deviation 2.36 mm > claimed bound 1.99 mm).
The corrected, provably conservative bound implemented here is

```
ε_lin = ¼ (Σᵢ √Rᵢ · |δθᵢ|)²
```

(derivation in `panda.py` docstring). At the recommended per-step action
clip |δθᵢ| ≤ 0.05 rad it stays ≤ ~2 cm, within the lemma's inflation budget.
Measured conservatism is ~30×; tightening via exact Hessian norms is a known
improvement path.

## M1 — tunneling-rate experiment (DONE, first pass)

`experiments/m1_tunneling.py`: a greedy goal-reaching policy using the
STANDARD discrete per-step collision check runs in dynamic scenes
(5 speed tiers x 60 episodes); every step is audited by the continuous
checker and by S-substep discrete checks (S = 1..16).

Results (seed 0, 300 episodes, 155 true-collision events —
`experiments/results/m1/`):

- **Standard checking (S=1) missed 14.8% of true contacts; S=16 still
  missed 2.6%** — the "just subsample more" rebuttal now has a cost curve.
- **Miss rate is NOT monotone in obstacle speed** (blueprint H1 corrected):
  29% missed even at 0.1 m/s, because tunneling is driven by *relative*
  sweep and the arm's own distal links move ~1 m/s per step. Discrete
  collision accounting is unsound at all obstacle speeds.

## M1-hardened (DONE) — 3 seeds x 100 eps x 5 speeds, 764 audited collisions

`experiments/m1_hardened.py` -> `experiments/results/m1_hardened/`. Adds
Wilson 95% CIs, per-contact thickness attribution, wall-clock comparison.

- **Pooled S=1 miss rate: 19.0% [16.4%, 21.9%]**; S=16 still misses 0.8%.
- **Speed trend INVERTED vs naive intuition**: miss rate is *highest at the
  lowest speed* — 45.6% [34.3%, 57.3%] at 0.1 m/s vs 13.9% at 2.0 m/s.
  Slow-regime contacts are arm-sweep grazes (in-and-out within one step);
  fast obstacles plow in and remain overlapping at the endpoint. Discrete
  protocols are least sound exactly where benchmarks feel safest.
- **Thickness intuition also fails**: compact obstacles missed MORE (22.7%)
  than thin plates (15.8%) — driver is grazing probability (small
  cross-section), not thinness. Tentative interpretation, flagged as such.
- **Cost**: exact audit 1.50 ms/step < S=16 discrete (1.92 ms, still leaky).
  Pure Python/NumPy on both sides.

- **Severity ("are missed contacts severe?")**: missed contacts are
  typically milder (median max-overlap 1.4 cm vs 10.2 cm for caught ones)
  BUT carry a deep-contact tail: q75 = 3.6 cm, worst case 25.1 cm of link
  inside the obstacle, fully invisible to the standard check.

Remaining caveats: scripted policy (protocol study; trained-policy audit =
M1b), a_r=0.06 m placeholder pending advisor confirmation.

## M2a — analytic voxelizer (DONE)

`godynur/voxelizer.py`: closed-form AABB -> 32^3 occupancy rasterization,
0.074 ms/call — the observation bridge AND the free supervision generator
for occupancy forecasting (`rasterize_future`). Brute-force-verified.

## M2-pre — forecasting kill test (DONE: keystone alive, conditional pass)

`experiments/m2_forecast_feasibility.py`. v1 failed and exposed two test
design errors (no identity path in the decoder; 50 ms horizon too short —
half a voxel of displacement leaves nothing to learn). v2 (U-Net skips +
last-frame passthrough + 0.2 s horizon) passes the pre-registered criterion:

- beats BOTH controls at 0.5 / 1.0 / 2.0 m/s (2.5x persistence at 1.0,
  4.5x at 2.0); loses to persistence below 0.25 m/s where displacement is
  sub-voxel (physically nothing to forecast).
- k3 > velocity-blind k1 at ALL speeds, both versions: the mechanism
  ("forecasting forces motion info into the latent") is confirmed twice.

Conditions carried into M2-full: absolute IoU still weak (GPU training +
dice/focal loss are the upgrade path); the "forecaster doubles as the
deployment shield's prediction source" role is NOT validated at current
precision — fallback is constant-velocity extrapolated voxels (analytic);
auxiliary horizon fixed at ~0.2 s.

## DynArmEnv — the M2/M3 training environment (godynur/env.py)

**Lab code lineage**: the group's stack descends from Morvan Zhou's
train-robot-arm-from-scratch (Bilibili BV1nW411a7Qg, 2017) — hand-written
analytic env, `step(a) -> (s, r, done)` / `reset() -> s`, joint-increment
actions, no simulator. URPlanner's parameterized task space is that pattern
scaled up. `DynArmEnv` keeps the exact Morvan interface (drop-in for the
lab's DDPG/TD3/APE2 code) and upgrades the internals:

- moving obstacles (speed-tiered, for curriculum);
- **continuous-time exact** collision accounting (cannot tunnel) + free
  instrumentation (`last_discrete_missed`: would the endpoint check have
  missed this collision?);
- switchable reward: `'uoar'` (URPlanner Eq.12) vs `'ct'` (D-UOAR-CT with
  swept integral + continuous TTC) — the M3 ablation axis;
- `grid_history()` -> (k, 32, 32, 32) occupancy stack for the M2 encoder,
  analytic rasterization (simulator-free).

## M1b notes (recon done)

DRP is **CoRL 2025** (not just arXiv). Pre-release = inference + IMPACT
checkpoints; needs Python 3.8 conda + CUDA pointnet2_ops + robofin. This
machine has an RTX 3070 (8GB, WSL2) — locally plausible but a real
environment build; consider the lab GPU box instead.
