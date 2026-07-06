# GO-DynURPlanner — Continuous-Time Collision Reasoning for Dynamic Manipulator Motion Planning

Research in progress — NUS master's project extending URPlanner
([arXiv:2505.20175](https://arxiv.org/abs/2505.20175)) toward dynamic
(moving-obstacle) environments.

## Abstract

Standard reinforcement-learning motion planners for manipulators check
collisions once per control step, at the start and/or end configuration.
This project builds an **exact continuous-time collision checker** over
the URPlanner parameterized space (analytic forward kinematics, segment-vs-AABB
overlap, and closed-form first-contact time within a control interval) and
uses it to audit that discretization assumption directly. Across pooled
dynamic-scene trials, the standard discrete per-step check misses
**~19% of true contacts**, with the failure worst — not best — in the
quasi-static regime, contrary to the naive intuition that slow obstacles are
easy to catch. Building on the same continuous-time machinery, an
occupancy-forecasting auxiliary task (predicting near-future voxel occupancy
from an analytic rasterizer) is validated with a pre-registered kill test.
Both lines of work run inside `DynArmEnv`, a training environment that keeps
the exact `step()/reset()` interface of the lab's Morvan-lineage codebase
while replacing static-scene, discrete-time internals with continuous-time,
dynamic-obstacle ones.

## Repo Map

| Path | Contents |
|---|---|
| `code/` | Library (`godynur/`), experiments (`experiments/`), tests (`tests/`) — see `code/README.md` for the module-level walkthrough |
| `code/ASSUMPTIONS.md` | Interface assumptions inferred from the Morvan-lineage codebase and the URPlanner paper, flagged by confidence level, pending reconciliation with the lab's actual code |
| `research-log/` | 7 research loops documenting the path from initial topic to the current direction (in Chinese) |
| `index.html` | Timeline view of the research process |
| `briefing_for_advisor.html` | Advisor-facing summary of progress and open questions |
| `research-topic.md` | The original project topic statement |

## Quickstart

```bash
cd code
python3 -m pytest tests/ -q
python3 demo.py
```

Requires `numpy` and `pytest`. `torch` is only needed for the occupancy
forecasting and TD3 experiments (`experiments/m2_forecast_feasibility.py`,
`godynur/forecast.py`, `godynur/td3.py`) — not for the core library or its
test suite.

## Key Results (from `code/README.md`)

| Finding | Value |
|---|---|
| Pooled discrete (S=1) miss rate vs continuous ground truth | 19.0% [16.4%, 21.9%] (Wilson 95% CI) |
| Miss rate at 0.1 m/s (worst case — quasi-static regime) | 45.6% [34.3%, 57.3%] |
| Miss rate at 2.0 m/s | 13.9% |
| Missed-contact severity, max overlap: median / worst case | 1.4 cm / 25.1 cm |
| Per-step wall-clock: exact continuous audit vs S=16 discrete (still leaky at 0.8%) | 1.50 ms vs 1.92 ms |
| Occupancy-forecasting kill test | passes at 0.5 / 1.0 / 2.0 m/s (beats persistence and velocity-blind controls) |

## Disclaimer

- `a_r = 0.06 m` (link cylinder radius) is a **placeholder** — the URPlanner
  paper text does not state this value numerically. All results using it are
  directionally stable but magnitudes will shift once confirmed by the lab.
- Several interface details (state vector composition, action scaling, env
  API) are **inferred, not confirmed** — see `code/ASSUMPTIONS.md` for the
  full list with confidence levels and reconciliation steps.
- These are **pre-publication research results**: unreviewed, subject to
  revision, and not yet validated against the lab's own codebase.
