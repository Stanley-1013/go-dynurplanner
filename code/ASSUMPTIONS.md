# Interface Assumptions — to be reconciled when 師兄's code arrives

> **2026-07-07 advisor reply (partial reconciliation)**: URPlanner code
> arriving tonight. a_o = 5 cm confirmed (matches item as extracted).
> Grid semantics ANSWERED: binary is hard to learn — use SDF, coarse
> resolution first (implemented: grid_mode='sdf', grid_n=16), plus
> closest-obstacle-point distance/direction features in the vector state
> (implemented: closest_point_in_state -> [d, unit direction, point rel
> flange], analytic segment-box closest point via convex ternary search).
> Note: MD-features in the OBSERVATION do not conflict with URPlanner's
> MD-independent REWARD claim; in the parameterized space the closest
> point is closed-form cheap.

Everything below is INFERRED (from the Morvan video lineage + the URPlanner
paper full text). Each item lists: what we assumed, the evidence, and what
to do when the lab's actual code arrives. Items marked ⚠ are placeholders
that WILL need a value/decision from the lab.

## 1. Environment interface (Morvan-style) — HIGH CONFIDENCE
- **Assumed**: `env.reset() -> s`, `env.step(a) -> (s, r, done)` (3-tuple,
  no gym `info`), class attrs `state_dim` / `action_dim` / `action_bound`,
  `sample_action()`. Hand-rolled analytic env, no gym dependency.
- **Evidence**: 師兄 explicitly pointed to Morvan's
  train-robot-arm-from-scratch (BV1nW411a7Qg); read its `final/env.py`.
- **On code arrival**: diff `DynArmEnv` against the lab env's exact
  signature (does their step return 4-tuple? do they normalize state the
  same way?) and adapt `godynur/env.py` — all internals are
  interface-independent.

## 2. State vector composition — MEDIUM CONFIDENCE
- **Assumed**: `[q_norm(7), flange(3), goal(3), goal-flange(3), dwell(1)]`
  + optional per-obstacle `[rel_pos(3), vel(3)]`. URPlanner Eq.(2) uses
  `[q, p_T, p_G, Δp, Δo, D]` — we dropped orientation error (Δo) and the
  binary D flag's exact form; our goal is position-only for now.
- **On code arrival**: match Eq.(2) exactly incl. orientation error and
  their normalization constants.

## 3. Action scaling — MEDIUM CONFIDENCE
- **Assumed**: joint increments Δq, clipped at ±0.05 rad/step (chosen for
  the conservativeness-lemma inflation budget, ≈1 rad/s at 20 Hz).
- **Paper**: Eq.(3) confirms joint-increment actions; per-step magnitude
  not stated in extractable text.
- **On code arrival**: adopt their action_bound and dt; re-derive ε_lin
  (one function call) and re-run M1 numbers if dt differs from 0.05 s.

## 4. ⚠ Link cylinder radius a_r = 0.06 m — PLACEHOLDER
- Paper text never states the number (a_o = 5 cm IS stated, §VII-C).
- All M1/M1-hardened results use 0.06 m; direction of conclusions is
  insensitive, magnitudes will shift. **Ask 師兄; re-run
  `experiments/m1_hardened.py` (12 min) once known.**

## 5. Simulation platform — CONFIRMED CoppeliaSim (verification layer only)
- Paper: baselines (DDPG:CAR/COR) run in a "virtual twin system established
  in CoppeliaSim"; MD measurements use CoppeliaSim's MD module; real robot
  via MoveIt. Training itself happens in the parameterized space (that's
  URPlanner's whole point) — matching our DynArmEnv role split.
- **Later (M3/M4)**: add a CoppeliaSim scene (ZeroMQ remote API) for
  trajectory verification + demo videos, mirroring the paper's Table VII
  protocol. NOT needed for training.

## 6. Robot — CONFIRMED Franka Emika Panda, 7-DoF
- Modified DH (Craig) per paper Eq.(4); consolidation (1,2),(3,4),(5,6),
  (7,flange) → 4 segments (paper §III-A). Joint limits from Franka specs.
- Panda DH table itself is from public Franka docs (not in the paper's
  extractable text) — structural FK properties are test-verified
  (`tests/test_panda.py`), but **cross-check numerically against the lab's
  FK or CoppeliaSim model when available**.

## 7. Pose reward phi_aux — APPROXIMATED
- URPlanner Eq.(11): r_pose = -(e_p + e_o) + phi_aux + phi_G. phi_aux's
  exact form is in their ref [9] Eqs.(12-13) (not obtained). Implemented as
  0.5*exp(-err/0.08); phi_G = 1 inside tolerance; orientation error e_o
  dropped for now (position goals). Verified empirically: stage-0 reaching
  goes from unlearnable (0.2 rolling success @1600 eps with bare -err +
  dwell-5) to 0.73 @500 eps. **Swap in the exact phi_aux when [9]/lab code
  arrives.**

## 8. Training hyperparameters (Table I, for later APE2 reproduction)
- lr 1e-3, memory 6e4, soft update 0.01, batch 64, ξ=0.98, ζ=1,
  M=2, N=3 (→ 7 candidates), T=2e5, H=1, ω1=0.6, ω2=0.1, N_D=80, φ=0.1,
  T_B=2e3. Our TD3 baseline uses its own standard hypers; APE2
  reproduction should use these.
