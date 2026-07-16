# Kinodynamic Certified Safety Layer — Roadmap & Task Queue

> Living document for a long-running, self-paced Claude+Codex loop. **Read
> this file FIRST in any new session/wakeup before touching this work** —
> context resets between loop iterations, this file + git history is the
> only durable state. Update §6 (status log) and check off §3 every
> iteration. Do not re-derive what's already here.

## 0. Origin & scope boundary

- Builds on the existing GO-DynURPlanner project. Paper 1's C1–C4 results
  (measurement, conservativeness lemma, APE2-Shield, honest reward-shaping
  negative — see `paper/PAPER1_PLAN.md`, `HANDOFF.md`) are **DONE, do not
  re-run**. This roadmap is a NEW extension track, additive to that work.
- Trigger: user's 2026-07-17 design note — extend RL action from
  position-delta to nominal velocity; replace the current linear-
  interpolation continuous-time shield with a jerk-based kinodynamic
  certified action shield (full derivation lives in that chat turn, not
  reproduced verbatim here — §1 below is the condensed operational spec).
- User's explicit directives governing how this loop runs: (a) run as a
  self-paced `/loop`, collaborating with the **Codex CLI** for
  implementation, (b) do broad/deep experiments and **increase seed
  counts** beyond the n=2 used so far, (c) pace iterations to avoid
  hitting temporary token limits — steady, sustainable progress over a
  long horizon, not one giant burst, (d) don't skimp on model strength for
  research/judgment steps; use cheap models/tools for mechanical steps,
  (e) Codex model = **gpt-5.6 series** (`gpt-5.6-codex`).

## 1. Condensed architecture spec

- Per-joint state `x=(q,v,a)`; jerk `j` is the control primitive over one
  control period `h` (existing `dt=0.05s`, see §2). Linear-jerk-within-step
  ⇒ `a(τ)` linear, `v(τ)` quadratic, `q(τ)` cubic, `τ∈[0,h]`.
  Discrete update: `x_{t+1} = A x_t + B j_t`,
  `A=[[1,h,h²/2],[0,1,h],[0,0,1]]`, `B=[h³/6, h²/2, h]ᵀ`.
- RL emits **nominal velocity** `v_nom = v_t + Δv_scale · u_t`,
  `u_t∈[-1,1]` — this is structurally the SAME shape as the action mapping
  that already exists (`q_{t+1}=q_t+Δq_scale·u_t` via tanh actor, see
  §2) — a generalization, not a rewrite of the RL/actor side.
- Safety is NOT delivered via reward shaping — C4 already found CT-reward
  shaping doesn't help safety; hard guarantees must stay in a certified
  shield. New shield = nearest-to-nominal jerk sequence subject to:
  `(q,v,a,j)` box constraints, an N-step terminal braking set, **continuous-
  interval** (not just endpoint) `q(τ)`/`v(τ)` bound satisfaction (critical
  points at `a(τ)=0` and `v(τ)=0`, closed form — this is the whole reason
  Phase 1 exists, see its regression test), plus the existing collision
  shield generalized from linear to cubic joint trajectories.
- Receding horizon: plan N steps, execute only `j_0`, replan next period.
- Explicitly rejected alternative (documented so it isn't re-litigated):
  collapsing everything into a single `v_{t+1}=F(v_t,u_t)` with no `a_t`
  dependence. Cannot be both exact and non-conservative — `a_t` has
  independent physical effect on stopping distance and must stay in the
  safety layer's state, even though the RL-facing action mapping stays
  `a_t`-free for a stable action semantics.

## 2. Current codebase mapping (verified 2026-07-17, Explore agent)

- `env.py` `DynArmEnv`: state is **position-only** (`q_norm(7)`, flange(3),
  goal(3), goal-flange(3), on-goal flag(1), + optional obstacle/closest-
  point features). No `v`/`a` in state anywhere. Action = **delta-q**,
  `action_bound=[-0.05,0.05]` rad/step, added directly to `q` and clipped
  to `[Q_MIN,Q_MAX]` (env.py:38-40,139-142,156). `dt=0.05s` (env.py:38).
  `step()` returns 3-tuple `(state, reward, done)`, no `info`.
- `continuous.py`: exact first-contact-time collision shield — but its
  motion model is **linear (constant-velocity) interpolation** of segment
  endpoints over `[0,dt]` (continuous.py:1-9,205-219). This is consistent
  with today's position-only kinematics but will NOT be valid once joint
  trajectories become cubic — Phase 4 must generalize this, not bypass it.
  Key calls: `first_contact_time(seg, mbox, dt)`, `interval_collision_free`,
  `swept_overlap_integral`, `lemma_inflation(eps_lin, a_max, dt, eps_v)`.
- `ape2.py` `APE2Shield`: generates `2MN+1` delta-q candidates around the
  base policy action, scores by `eta·Q + (1-eta)·R_IR`, picks the best
  candidate that certifies via `env.peek_tau_star`; falls back to scaling
  by α∈{0.5,0.25,0.125,0}; if still infeasible, picks the candidate that
  maximizes τ* and records `stats["no_safe"]`. Only touches `continuous.py`
  via `env.peek_tau_star`.
- `td3.py`: standard TD3, actor `Tanh` output scaled by `action_scale` in
  `act()` — i.e. today's actor ALREADY outputs a `u_t∈[-1,1]` that's
  affine-mapped to a physical action. Swapping the target quantity from
  delta-q to delta-v is a parameter/semantics change, not an architecture
  change.
- `panda.py`: `Q_MIN`/`Q_MAX` (rad), `DQ_MAX` (rad/s, panda.py:41,
  `[2.175,2.175,2.175,2.175,2.61,2.61,2.61]`) — **exists but is dead code**,
  never used to bound anything. **No acceleration or jerk limit constants
  exist anywhere in the repo.**
- Tests: `test_continuous.py` and `test_ape2.py` assert tunneling-immunity,
  conservativeness (inflated-box-clear ⇒ tight-box-clear), exact
  `lemma_inflation` formula, `peek_reward`==`step()` reward, certified
  actions always have `last_tau_star is None`. These invariants must keep
  holding — new work is additive, not a replacement, until explicitly
  swapped in a later phase.
- Baseline: `cd code && source .venv/bin/activate && PYTHONPATH=.
  python -m pytest tests -q` → **46 passed** (2026-07-17, clean baseline
  before this track's first change).

## 3. Phased task queue

Check off as landed **and verified** (pytest green + read-back), not just
dispatched. Do not start a phase until the previous one's tests are green.

### Phase 1 — single-joint (q,v,a,j) model + continuous-interval certification (no obstacles, no wiring into RL)
- [ ] `code/godynur/kinodynamics.py`: discrete update (A,B), continuous
      τ-domain `a(τ)/v(τ)/q(τ)`, **exact interval-extrema check** (critical
      regression test: endpoint-legal but mid-interval-illegal case — this
      is the concrete proof of why Phase 1 exists)
- [ ] N-step braking-feasibility check (conservative bang-bang/S-curve
      closed form acceptable for v1; Phase 2 replaces with a real QP)
- [ ] accel/jerk limit placeholders added to `panda.py` (+ dated entry in
      `code/ASSUMPTIONS.md` following the existing placeholder convention,
      e.g. item 4's `a_r`)
- [ ] `code/tests/test_kinodynamics.py`, full suite green, nothing else
      touched
- Status: DISPATCHED to Codex 2026-07-17 (background), log at
  `code/experiments/logs/codex_phase1.log`

### Phase 2 — multi-joint jerk-horizon QP/LP safety filter
- [ ] N-step jerk-sequence optimization: `min ||v1-v_nom||²_W +
      λ_j Σ|j_k|²` s.t. `(q,v,a,j)` box + terminal braking set
- [ ] solver choice (check venv for `cvxpy`/`osqp`/`scipy.optimize`; add
      minimal dependency if needed) + realtime solve-time benchmark
- [ ] tests: infeasible-state handling (must degrade gracefully, never
      silently violate), solve-time budget, safety-margin sensitivity

### Phase 3 — integrate fixed velocity-action mapping into `DynArmEnv` (opt-in, non-breaking)
- [ ] add `v_nom` action mode alongside existing delta-q mode, flag-gated,
      existing mode stays default until this track's own experiments
      validate the new one
- [ ] observation additions: `a_t`, braking margin (±), last intervention
      magnitude
- [ ] safety-intervention penalty term `-λ‖v_nom - v_exec‖²`

### Phase 4 — extend `continuous.py` collision shield to cubic joint trajectories
- [ ] generalize `MovingSegment`/`first_contact_time` from linear to cubic
      `q(τ)` motion (or a documented, explicitly-conservative piecewise-
      linear approximation as an interim step)
- [ ] compare discrete-endpoint vs. continuous-time vs.
      continuous+braking shield: collision rate, joint-limit violation
      rate, inter-sample violation count

### Phase 5 — occupancy forecasting + TD3/APE2 integration, full experiments, **seed scale-up**
- [ ] wire `grid_td3.py`/`forecast.py` obstacle prediction into the N-step
      safety horizon
- [ ] full A/B experiments across shield variants. **Raise seed count** —
      existing H2/M5 results used n=2 only (HANDOFF.md: "n=2 hypothesis
      pending scale-up"); target n≥5 minimum, n=8–10 for anything destined
      for the paper, with Wilson CIs per the existing C1 convention
      (`m5_grid.py --seed-salt` already fixes the seed-confound bug found
      in Loop 9 — reuse that pattern, don't reintroduce the confound)
- [ ] intervention-rate-over-training metric — does the policy learn to
      self-limit, or does it permanently lean on the shield?

## 4. Evaluation metrics (apply from Phase 4 onward)

- **Safety**: collision rate, joint-limit violation rate, inter-sample
  violation count, fraction of states with no feasible emergency brake.
- **Efficiency**: task completion time, path length, jerk/energy cost.
- **Shield behavior**: intervention rate, `‖v_nom - v_exec‖`, QP solve
  time, infeasible-solve rate.
- **Learning**: intervention rate trend over training (should trend down
  if the policy is actually internalizing the limits).

## 5. Model routing policy for this loop

- **Implementation / mechanical work** (write a module+tests to a written
  spec, apply an already-proven pattern, run an experiment script): Codex
  CLI — `codex exec -m gpt-5.6-codex -c model_reasoning_effort=<medium|high>
  -s workspace-write -C <repo root> -o <output-file>`. Use `high` effort
  for anything deriving a new closed-form/constraint (Phase 1, Phase 2
  design); `medium` once a pattern is proven and being reapplied
  (batch experiment scripts, more seeds of an existing script).
- **Verification**: this commander session reads back the diff and runs
  pytest itself — Codex's own "done" claim is never trusted alone, same
  rule as any other subagent (dispatch.md §6). Long training runs get
  spot-checked logs, not just an exit code.
- **Judgment forks** (QP-infeasible-state strategy, terminal braking set
  choice, reconciling with the advisor's real E3AC code if it becomes
  relevant, how to frame an honest-negative result for the paper):
  escalate to Opus (`Agent` tool, `model: opus`) or ask the user — same
  triggers HANDOFF.md already defined for the base project, they still
  apply here.
- **Bookkeeping** (env/log checks, git commits, updating this file, running
  pytest): this session directly, no delegation needed — cheap and fast.
- **Pacing**: each loop wakeup does ONE phase-task-sized unit of work
  (dispatch → verify → commit → log → schedule next wakeup), not a whole
  phase at once, to keep any single turn's token usage bounded.

## 6. Loop status log (append one line per iteration, newest first)

- 2026-07-17 iter1: roadmap created; architecture mapped via Explore
  agent; baseline `pytest` confirmed 46/46 green. Phase 1 dispatched to
  Codex (`gpt-5.6-codex`, background) — see §3 Phase 1 status line.

## 7. How to resume after context loss

1. Read this file in full.
2. `git log --oneline -15` and `git status` — confirm what's actually
   landed vs. only checked off on paper.
3. Re-run the baseline pytest command in §2.
4. Check §6's last entry for a dispatched-but-unverified task; verify or
   re-dispatch it before starting anything new.
