# GO-DynURPlanner — Continuous-Time Collision Reasoning for Dynamic Manipulator Motion Planning

English | [简体中文](README.zh-CN.md)

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
easy to catch. Two further contributions build on the same machinery: a
**certified action shield** (APE2-Shield) that raises training-time success
rate ~4x and roughly halves the true collision rate by certifying candidate
actions before execution, converting residual collisions into provably
unavoidable events; and an **honest negative result** showing that shaping
the reward with the same continuous-time cost does *not* help once the
obvious pathology (an unbounded penalty term) is fixed — evidence that this
machinery's value lies in measurement and certification, not reward design.
A grid-observation + occupancy-forecasting extension is in progress (results
so far inconclusive at low seed count, see `research-log/09_*`). The project
has been reconciled against the advisor's real (TensorFlow) implementation of
URPlanner: forward kinematics verified numerically exact, several interface
assumptions corrected accordingly — see `code/ASSUMPTIONS.md`.

## Repo Map

| Path | Contents |
|---|---|
| `code/` | Library (`godynur/`), experiments (`experiments/`), tests (`tests/`) — see `code/README.md` for the module-level walkthrough |
| `code/ASSUMPTIONS.md` | Interface assumptions vs. the URPlanner paper and the advisor's real codebase — reconciled where the real code has arrived, flagged by confidence level elsewhere |
| `code/advisor_code_catalog.md` | Summary of the advisor's (unpublished, not included in this repo) codebase structure, produced during reconciliation |
| `paper/PAPER1_PLAN.md` | Working thesis, contributions, figure/section plan, and submission gaps for the first paper |
| `paper/references.md` | Consolidated, status-tagged citation ledger (verified / corrected / unverified) |
| `research-log/` | 9 research loops documenting the path from initial topic to the current direction (in Chinese) |
| `index.html` | Timeline view of the research process |
| `briefing_for_advisor.html` | Advisor-facing summary of progress and open questions |
| `HANDOFF.md` | Session handoff notes: project state, specific escalation triggers, process lessons |
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

## Key Results

| Finding | Value |
|---|---|
| Pooled discrete (S=1) miss rate vs continuous ground truth | 19.0% [16.4%, 21.9%] (Wilson 95% CI) |
| Miss rate at 0.1 m/s (worst case — quasi-static regime) | 45.6% [34.3%, 57.3%] |
| Miss rate at 2.0 m/s | 13.9% |
| Missed-contact severity, max overlap: median / worst case | 1.4 cm / 25.1 cm |
| Per-step wall-clock: exact continuous audit vs S=16 discrete (still leaky at 0.8%) | 1.50 ms vs 1.92 ms |
| Occupancy-forecasting kill test | passes at 0.5 / 1.0 / 2.0 m/s (beats persistence and velocity-blind controls) |
| **APE2-Shield** (certified action selection) vs no shield, equal training budget | success 0.675 vs 0.175, true-collision rate 0.24 vs 0.585 |
| Reward shaping with the same continuous-time cost (D-UOAR-CT) vs. plain discrete reward | **honest negative** — trails plain reward (0.47/0.40 vs 0.65/0.25) even after fixing an unbounded-penalty pathology |
| Grid observation + occupancy-forecasting auxiliary task, decoupled-seed rerun | inconclusive at n=2 per arm; suggestive pattern that the auxiliary task reduces training-collapse risk rather than raising peak performance — flagged as a hypothesis, not a result |

See `research-log/07_execution_m0_m1.html` through `09_shield_verdict_and_advisor.html` for the full derivations, and `paper/PAPER1_PLAN.md` for how these are assembled into a submission.

## Disclaimer

- `a_r = 0.06 m` (link cylinder radius) is a **placeholder** — the URPlanner
  paper text does not state this value numerically. All results using it are
  directionally stable but magnitudes will shift once confirmed by the lab.
- The advisor's real (TensorFlow) implementation has been obtained and partly
  reconciled against (`code/ASSUMPTIONS.md` §0): forward kinematics and the
  collision-overlap algorithm are verified numerically/structurally identical;
  several other interface details (action scaling, state composition, reward
  recipe, goal-completion threshold) were found to differ from this repo's
  defaults and are documented, not yet retroactively applied to existing
  results (all A/B comparisons above used a consistent environment within
  themselves, so relative conclusions hold regardless).
- These are **pre-publication research results**: unreviewed and subject to
  revision.
