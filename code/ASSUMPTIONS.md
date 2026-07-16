# Interface Assumptions

> **2026-07-08: advisor's real code arrived (0801pretrain.zip, kept OUT of
> git per .gitignore — unpublished lab code). Reconciliation against
> `Franka_Env_Scene2.py`, `panda_fk.py`, `Ray_Box_Intersection.py`,
> `main.py` below. Full catalog of E3AC variants / dual-memory / ED2 /
> CoppeliaSim interface files delegated to a subagent — see item 9.**
 — to be reconciled when 師兄's code arrives

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

## 0. REAL CODE RECONCILIATION (2026-07-08) — supersedes items below where noted

- **FK: VERIFIED EXACT.** Numerically compared `panda.py.joint_origins()`
  against `panda_fk.py`/`Franka_Env_Scene2.get_fk_solution()` for
  q=[0,0,0,-90,0,90,45]deg: joint2/joint4/joint6 positions match to full
  float precision (both go through the same 7-row DH table, same
  a/d/alpha values). **Only the final tool offset differs by design**:
  theirs = joint7 + 0.207m(135° twist) + 0.05m (task-specific
  gripper/tool TCP, only in the ENV class's 9-row table — the standalone
  `panda_fk.py` module has just the 8-row/0.207m version, itself
  inconsistent with the env — sloppy but harmless since env is what's
  used); mine = joint7 + 0.107m (bare standard Panda flange). Not a bug —
  a config choice. Swap-in point: replace `panda.py`'s last `_MDH` row
  with `[0, 0.257, 0]` (net) if exact TCP matching is ever needed.
- **Collision geometry: VERIFIED SAME ALGORITHM.**
  `Ray_Box_Intersection.Box.get_intersect_length` is the identical slab
  method as `geometry.segment_box_overlap` (t0/t1 clamped to [0,1],
  near/far intersection points) — line-for-line equivalent logic.
- **Env interface: WRONG, must fix.** Real `step()` returns a **5-tuple**
  `(next_state, reward, done, pose_error, orient_error)`, not the 3-tuple
  I assumed. `reset()` still returns `s` alone. No `sample_action()` on
  the class itself in the clean sense (there's one but it buggily
  references a module-global `env` — don't replicate that bug).
- **Action scale: WRONG, must fix.** `action_bound = [-1.5, 1.5]` in
  **degrees**, converted via `action * pi/180` — i.e. real per-step joint
  change is **≤0.02618 rad**, roughly HALF my 0.05 rad assumption. Also:
  algorithm is called **E3AC** (one of the three backbones URPlanner's
  APE2 wraps — DDPG/TD3/E3AC per the paper), implemented as class methods
  `extensive_exploration_strategy` (candidate generation) and
  `evaluate_and_choose_optimal_action` (hybrid eval) directly on the E3AC
  class — this IS APE2's mechanism, just named/organized differently.
  Framework is **TensorFlow 1.x** (`tf.set_random_seed`,
  `tf.compat.v1`-era), not PyTorch — a real stack mismatch with our
  code (see item 9 for how to handle).
- **State: WRONG, must fix.** Real state is **20-D**: `[q(7), TCP_pos(3,
  /3), goal_pos(3, /3), Δpos(3), Δorient(3, /180), on_goal_flag(1)]` —
  **includes orientation** (I dropped it). Confirms URPlanner Eq.(2)'s
  Δo term is real and used; my position-only simplification should be
  flagged explicitly as a scope-reduction in any writeup, not silently
  matched to the paper's state definition.
- **Reward: WRONG, must fix — and this is the real find.** The actual
  φ_aux-equivalent is NOT a smooth exponential proximity bonus (my
  guess) — it's a **discrete step-improvement PBRS**: `+0.05` if
  distance-to-goal decreased this step else `-0.05`; separately `+0.03`/
  `-0.03` for orientation improvement/worsening. Plus base pose term
  `-Δpos_norm - Δorient_norm`, plus `-total_intersection_length /
  total_link_length` (their name for our UOAR term — confirms UOAR's
  real form: normalize by `total_link_length = 0.9101`, no ζ weighting
  visible at this call site), plus `+1` per step while inside tolerance.
  **`goal_dwell` = 50 consecutive in-tolerance steps for `done=True`**
  (Morvan's original value) — I had changed this to 1; WRONG, revert to
  50 (or re-verify why I thought 1 was closer to "URPlanner phi_G
  semantics" — the real code says otherwise: φ_G is a per-step +1, not
  the termination condition itself).
- **Obstacle model: scene-specific, not general.** `Franka_Env_Scene2.py`
  (their STATIC baseline reproduction of URPlanner, no dynamic obstacles
  at all) hardcodes exactly 4 fixed AABBs and checks only **3 link
  segments** (joint2→joint4, joint4→joint6, joint6→end — link1
  base→joint2 is never checked, presumably provably clear of their fixed
  boxes in that scene). My 4-segment convention
  `[(0,2),(2,4),(4,6),(6,8)]` is the paper's general recommendation and
  fine to keep for a general dynamic-obstacle env — just note the
  baseline reproduction target uses fewer segments.
- **CoppeliaSim confirmed a second way**: `sim.py`/`simConst.py` present
  (classic V-REP/CoppeliaSim legacy remote API bindings) — matches Loop 7
  full-text finding.

## 0.5 Catalog of remaining files (2026-07-08, full report in
`advisor_code_catalog.md`, produced by a cataloging subagent, spot-checked)

- **Our eta-annealed hybrid evaluation design is independently validated.**
  The paper's APE2 mechanism is only actually LIVE in one file
  (`main_E3AC_foresee.py`): `eta_1 = clip(counter/200000, 0, 1)` blends a
  one-step lookahead reward (computed by literally stepping their env and
  restoring state — cheap since it's a closed-form math env) with the
  critic Q-value. This is exactly what our `env.peek_reward()` +
  `APE2Shield`'s eta-anneal does, just implemented more cleanly (stateless
  peek vs. step-then-restore). No change needed to `ape2.py` — this
  confirms the design, doesn't correct it.
- **Candidate generation differs**: real code uses **3 distinct noise
  distributions** (Gaussian σ=0.1, Gaussian σ=0.05, OU θ=0.05/σ=0.1) × N
  repeats + 1 raw action, not our M-scales-of-Gaussian-only simplification.
  Documented simplification, not urgent to match exactly.
- **Critic ensemble**: real code hardcodes **5 critics** (not a generic K),
  loss = `Eta_1(0.4)*td_error_avg + Eta_2(0.6)*td_critic_i +
  Omiga(0.1)*(Q_i-Q_avg)^2`. We use TD3's twin-critic-min instead — a
  legitimate, more modern alternative; note the difference, don't chase it.
- **ED2/DC mechanism confirmed, with a caveat**: two real memories
  (interaction + expert `.npy` file) with a step-counter schedule
  controlling the sampling ratio — this IS the paper's DC design. BUT the
  "Diffusion" variant does NOT run a diffusion model online — it just
  loads a pre-generated `diffusion_expert_memory.npy`. If we ever build
  ED2, the online-diffusion-generator is NOT in this handoff to port from.
  Also: **the lab's own files disagree on schedule direction** (grow-
  expert-from-0% vs. decay-expert-from-~67%-to-0%) — ask the advisor which
  is canonical before implementing, don't infer from filenames/dates.
- **"2step" ≠ dynamic obstacles** — confirmed it's a two-phase
  fill-buffer-then-train procedure, unrelated to motion prediction. Rules
  out a possible false lead.
- **CoppeliaSim confirmed concretely**: `sim.simxStart('127.0.0.1', 19999,
  ...)` — legacy V-REP remote API, localhost:19999. Useful for the later
  real-robot/verification section.
- **Lab-code footguns to remember, not fix (not our code)**: save/restore
  path strings are hand-edited per run and drift out of sync in at least 2
  files; one file undersamples ~5% of its expert memory due to an index
  bound bug. Don't trust any specific checkpoint-name -> script mapping
  without checking the live `save()` call.

## 1. Environment interface (Morvan-style) — SUPERSEDED, see item 0

- **Assumed**: `env.reset() -> s`, `env.step(a) -> (s, r, done)` (3-tuple,
  no gym `info`), class attrs `state_dim` / `action_dim` / `action_bound`,
  `sample_action()`. Hand-rolled analytic env, no gym dependency.
- **Evidence**: 師兄 explicitly pointed to Morvan's
  train-robot-arm-from-scratch (BV1nW411a7Qg); read its `final/env.py`.
- **Corrected by item 0**: the real training env's `step()` returns a
  5-tuple `(s, r, done, pose_error, orient_error)`. The Morvan lineage was
  the right ancestor to look at, but the group's actual code extended the
  interface — this item is kept for the paper trail, item 0 is the current
  truth.

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

## 10. ⚠ Panda acceleration/jerk limits — PLACEHOLDER (2026-07-17)
- The new kinodynamic safety track uses per-joint `DDQ_MAX = [5, 5, 5, 5,
  7.5, 7.5, 7.5] rad/s²` and `DDDQ_MAX = [50, 50, 50, 50, 75, 75,
  75] rad/s³`. These are physically reasonable initial engineering values,
  with higher limits for the three distal joints in the same spirit as their
  higher existing `DQ_MAX`; they are **not verified Panda controller limits**.
  Confirm against the lab robot's configured limits before using the shield
  for experiments or hardware, then re-run all kinodynamic safety tests.
