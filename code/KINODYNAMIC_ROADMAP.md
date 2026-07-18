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
- [x] `code/godynur/kinodynamics.py`: discrete update (A,B), continuous
      τ-domain `a(τ)/v(τ)/q(τ)`, **exact interval-extrema check** (critical
      regression test: endpoint-legal but mid-interval-illegal case — this
      is the concrete proof of why Phase 1 exists)
- [x] N-step braking-feasibility check (conservative bang-bang/S-curve
      closed form acceptable for v1; Phase 2 replaces with a real QP)
- [x] accel/jerk limit placeholders added to `panda.py` (+ dated entry in
      `code/ASSUMPTIONS.md` following the existing placeholder convention,
      e.g. item 4's `a_r`)
- [x] `code/tests/test_kinodynamics.py`, full suite green, nothing else
      touched
- Status: COMPLETE and verified 2026-07-17; 54 tests passed (46 existing +
  8 Phase-1 tests).

### Phase 2 — multi-joint jerk-horizon QP/LP safety filter
- [x] solver choice: `code/.venv` had NO solver at all (`scipy`, `cvxpy`,
      `osqp`, `qpsolvers` all absent — checked 2026-07-17). Installed
      `scipy` (1.18.0) — smallest dependency that can solve this size of
      QP (few vars/step × N horizon, box + simple linear constraints) via
      `scipy.optimize.minimize`. Not added to a tracked requirements file
      because none exists in this repo (`.venv` itself is gitignored,
      deps have always been implicit) — flagging here so a fresh venv
      setup knows to `pip install scipy`. Revisit with `osqp`/`cvxpy`
      only if Phase 5 realtime solve-time benchmarks demand it.
- [x] N-step jerk-sequence optimization: `min ||v1-v_nom||²_W +
      λ_j Σ|j_k|²` s.t. `(q,v,a,j)` box + terminal braking set
- [x] realtime solve-time benchmark
- [x] tests: infeasible-state handling (must degrade gracefully, never
      silently violate), solve-time budget, safety-margin sensitivity
- **Perf finding (commander, 2026-07-17, direct measurement, m=7 joints,
  h=0.05s, SLSQP via scipy.optimize.minimize)**: `n_steps=3` → 11.7ms mean
  /12.7ms p95; `n_steps=5` → 31.9ms/41.1ms; `n_steps=8` → 72.9ms/87.7ms.
  **This is too slow to call every env.step() at anything but the
  shortest horizon during RL training** — at n_steps=8, 200k training
  steps would cost ~4 hours in the safety filter alone. Phase 3's
  per-step live shield MUST default to a short horizon (`n_steps=3`);
  longer horizons are fine for offline/eval-time analysis only. If
  training-scale runs (Phase 5) still find this a bottleneck, switch to
  `osqp`/`cvxpy` (warm-started QP, likely 10-100x faster) rather than
  reducing `collocation_points` or `max_retries`, which would weaken the
  certification, not just the speed.

### Phase 3 — integrate fixed velocity-action mapping into `DynArmEnv` (opt-in, non-breaking)

**IMPORTANT DESIGN CORRECTION (commander, 2026-07-17, confirmed via Opus
consult) — read before touching this phase.** Direct measurement (see
Phase 2's perf finding above) showed Phase 2's hard `v_N=0,a_N=0` terminal
EQUALITY constraint, at the only real-time-affordable horizon (`n_steps=3`,
0.15s), suppresses almost ALL requested motion — even a modest per-joint
`v_nom` (~10-30% of `DQ_MAX`) comes back certified at ≈0. Requiring a full
stop within the SAME short horizon used for the real-time solve conflates
two different things: "box limits must hold over the next control period"
(needs to be real-time-cheap) and "a stop must remain reachable eventually"
(the actual safety requirement — doesn't need to be found by the same
expensive QP). Opus confirmed this is the standard MPC terminal-*set*-vs-
terminal-*point* distinction and endorsed the fix below (agent id
`a1459dd7f55f19c5f` if this session needs to resume that consult).

**Corrected design, replaces the Phase 2 §3 items above as written:**
1. Real-time QP: `n_steps=2` (not 3), terminal equality DROPPED from the
   QP entirely — it only certifies box (q,v,a,j) limits over the next 2
   control periods. Needs a new `safety_qp.solve_safety_qp(...,
   require_terminal_stop: bool = True)` param; `False` skips the terminal
   `LinearConstraint`. Default `True` preserves Phase 2's existing
   tests/behavior unchanged.
2. Terminal SET membership (not point) check, run separately and cheaply
   on the state reached after the executed step: reuse Phase 1's
   `braking_feasible` witness search, but it currently only returns
   `bool` — add a sibling `braking_witness_jerk(...) -> float | None`
   that returns the FIRST jerk of the same conservative bang-bang/S-curve
   witness (refactor the shared search logic out of `braking_feasible` so
   both call one internal helper — don't duplicate the math). Evaluate
   this membership check with **derated authority** (`0.85 * a_max`,
   `0.85 * j_max` — Opus's margin against one control period of latency),
   horizon `n_brake=15-20` (cheap, O(n) closed-form, not a QP).
3. **Fallback for recursive feasibility** (Opus's addition, closes the gap
   between replans): the env stores the last accepted witness jerk. If a
   step's QP+membership check fails, execute the STORED witness's braking
   jerk rather than re-solving arbitrary — this is what makes the
   guarantee inductive (always either extending a certified-safe state or
   committing to an already-certified brake), not just "hope the next
   solve works."
- [x] **Phase 3a (do this first, additive, small)**: `require_terminal_stop`
      flag on `safety_qp.solve_safety_qp`; `braking_witness_jerk` on
      `kinodynamics.py` (shared logic with `braking_feasible`, no
      duplication); tests for both; full suite green, nothing else touched.
- [x] **Phase 3b (after 3a verified)**: add `v_nom` action mode to
      `DynArmEnv`, flag-gated (`action_mode="delta_q"` stays default,
      byte-identical existing behavior), wiring in the corrected 3-part
      design above, incl. the last-witness fallback state.
- [x] observation additions: `a_t`, terminal-membership flag, last
      intervention magnitude
- [x] safety-intervention penalty term `-λ‖v_nom - v_exec‖²`

### Phase 4 — extend `continuous.py` collision shield to cubic joint trajectories

**DESIGN NOTE (commander + Opus consult, 2026-07-17) — a near-miss worth
recording.** My first proposal here was UNSOUND and Opus caught it before
dispatch, via a clean counterexample — recorded because it's exactly the
class of bug this whole research track exists to catch, and almost shipped
it myself. Keep this note; don't let a future iteration re-propose the
same broken shortcut.

*What I proposed (WRONG):* keep the existing linear-chord collision
checker unchanged, but inflate its existing `eps_lin` bound
(`(1/4)(Σ√Rᵢ|δθᵢ|)²`, quadratic in the raw endpoint-to-endpoint angular
delta `|δθᵢ|=|q(h)ᵢ-q(0)ᵢ|`) by substituting a padded
`|δθᵢ|+2·overshootᵢ` in place of `|δθᵢ|`, where `overshootᵢ` is how far
Phase 1's exact `q_min/q_max` pokes outside the endpoint-spanned range.

*Why it's wrong:* `eps_lin`'s quadratic form bounds a genuinely
second-order effect (chord vs. FK of a MONOTONIC sub-motion) — it
implicitly assumes single-signed velocity, which cubic `q(τ)` no longer
guarantees. Counterexample: `q(0)=q(h)=0` with a mid-interval hump to `ε`
(possible now that q(τ) can reverse direction) → my formula gives
`|δθ|=0`, padded bound `≈ Rε²`, but the TRUE Cartesian excursion is
`≈√R·ε` — a FIRST-order quantity. For small `ε`, `Rε² ≪ √R·ε`: the padded
bound is smaller than the true deviation, i.e. **unsound**, exactly the
"checker misses a real excursion" failure this project's own C1 finding
was about. Folding overshoot into the quadratic term hides a first-order
effect inside a second-order formula.

**Corrected design (Opus-endorsed, use this):** triangle-inequality split
into two additive terms, not one inflated quadratic term:
- **Term A** (unchanged): today's `eps_lin(|δθᵢ|)` — chord vs.
  `FK(linear-interpolant(τ))`, monotonic by construction, existing proof
  applies as-is, no change to its formula or inputs.
- **Term B** (new): `Σᵢ √Rᵢ · dᵢ`, where `dᵢ = max_τ |qᵢ(τ) − linᵢ(τ)|`
  over `τ∈[0,h]` — the deviation of the TRUE cubic joint trajectory from
  the straight-line interpolant BETWEEN ITS OWN ENDPOINTS (not the same
  quantity as Phase 1's `interval_extrema`, which bounds `q(τ)` itself,
  not `q(τ)-lin(τ)`; needs its own small closed-form extremum search —
  `q(τ)-lin(τ)` is cubic in τ, vanishes at both endpoints by construction,
  its derivative is quadratic in τ so ≤2 interior critical points, same
  technique as Phase 1's critical-point search, different target function).
- `eps_total = eps_lin(|δθ|) + Σᵢ √Rᵢ·dᵢ`, fed wherever the existing
  `eps_lin` currently feeds into `lemma_inflation`/the margin.

**Validation is the actual soundness gate, not the formula alone**
(Opus's guidance): dense-sample the TRUE `q(τ)` → FK(q(τ)) trajectory,
compute the empirical max deviation from the chord, assert
`empirical ≤ eps_total` on adversarial constructions — (a) `q(0)≈q(h)`
with large jerk/accel producing a big mid-interval hump (the reversal
case that broke my original proposal), (b) small-angle regime (exposes
first-order dominance), (c) multiple joints reversing simultaneously
(cross terms), (d) configurations sitting exactly on the existing
collision margin. Log worst bound/true ratio; **fail if any ratio < 1**;
separately flag ratios ≫1 as excess conservatism (not a failure, but
worth noting, mirrors this repo's existing "measured conservatism ~30×"
disclosure style for the original `eps_lin`).

- [x] `dᵢ` closed-form deviation-from-chord bound per joint (new function
      in `kinodynamics.py`, reusing the critical-point-finding pattern,
      not `interval_extrema` itself — different target function)
- [x] `eps_total` composition (Term A + Term B) wired into wherever
      `continuous.py`/`panda.py` currently consumes `eps_lin`, for the
      velocity-mode cubic-trajectory case only — must NOT change the
      existing linear/delta_q path's `eps_lin` usage or its tests
- [x] adversarial dense-sampling validation tests per the four scenarios
      above; fail loudly (assert, not silently pass) if bound/true < 1
- [ ] compare discrete-endpoint vs. continuous-time vs.
      continuous+braking shield: collision rate, joint-limit violation
      rate, inter-sample violation count (this sub-item is Phase 5-scale
      experiment work, not required to land with the module itself)

### Phase 5 — occupancy forecasting + TD3/APE2 integration, full experiments, **seed scale-up**

**STATUS SUMMARY (2026-07-18, all iter18-33 threads concluded)**: the
safety shield itself (Phase 1-4) is verified END-TO-END, including at
scale — Phase 5j's audit found zero violations across 18,000 steps with
genuine (19-21%) shield engagement. The OPEN item is RL training
convergence for `kinodynamic_shield` specifically — three fixes tried
(replay-buffer consistency, missing velocity observation, missing
safety-margin observation), none alone resolved it, tested up to a
decisive 3000-episode single-seed run. This is now a genuine open
research question (see `ANALYSIS.md`'s top-of-file update), not a
quick-fix target. Items below are checked/unchecked accordingly —
**do not scale to n=8-10 or attempt another quick fix without a
deliberate decision on how much further to invest in the RL-convergence
question specifically** (systematic hyperparameter sweep, different
algorithm, or accept the shield as the standalone contribution).

- [ ] wire `grid_td3.py`/`forecast.py` obstacle prediction into the N-step
      safety horizon (not started — blocked behind the RL-convergence
      question in practice, low value to add prediction to a policy
      that isn't learning the base task yet)
- [x] **Phase 5a**: `experiments/m6_kinodynamic_shield.py` — three-arm
      comparison (`no_shield`/`ape2_shield`/`kinodynamic_shield`), disjoint
      per-arm seed ranges (`ARM_SEED_STRIDE=1e6`, avoids the Loop 9
      seed-confound even across arms), per-`EVAL_EVERY`-block intervention
      rate logged (supports the "Learning" metric below directly). All
      three arms independently smoke-verified by the commander (not just
      Codex's kinodynamic-only smoke) — 8 episodes, all ran cleanly, no
      crashes. Measured per-episode wall time at this tiny scale: `no_shield`
      ≈0.39s, `ape2_shield`≈0.89s, `kinodynamic_shield`≈5.6s (QP overhead
      dominates, consistent with the Phase 2 perf finding).
- [ ] full A/B experiments across shield variants. **Raise seed count** —
      existing H2/M5 results used n=2 only (HANDOFF.md: "n=2 hypothesis
      pending scale-up"); target n≥5 minimum, n=8–10 for anything destined
      for the paper, with Wilson CIs per the existing C1 convention
      (`m5_grid.py --seed-salt` already fixes the seed-confound bug found
      in Loop 9 — reuse that pattern, don't reintroduce the confound)
  - **Phase 5b sizing decision (commander, 2026-07-17)**: historical
    M4 (`experiments/logs/m4v2.log`) reaches final curriculum stage by
    ~ep 200 but success rate is STILL rising at its ep-3000 ceiling
    (noshield 0.33→0.40→0.47 in its last 3 checkpoints) — not fully
    converged even there. Given `kinodynamic_shield`'s ~5.6s/episode,
    the historical 3000-episode default would cost ~4.7h/seed (~23.5h
    for 5 seeds sequential) — too large a first commitment. Chose
    **800 episodes × 5 seeds** as the first real Phase 5b pass: well past
    curriculum completion, captures a meaningful early/mid training
    trend, caps `kinodynamic_shield` at ~1.24h/seed (~6.2h for 5 seeds
    sequential, dominant arm). Launched as 3 PARALLEL background
    processes (one per arm, `--seed-salt 0 --seeds 5 --episodes 800`),
    logs at `experiments/logs/m6_<arm>.log`, results under
    `experiments/results/m6_kinodynamic/`. This is a first pass, not the
    final paper numbers — plan a longer/larger Phase 5c rerun (closer to
    the historical 3000-episode budget, n=8-10) once these results are
    sane-checked, per the roadmap's own n≥5-minimum/n=8-10-for-paper
    two-tier target.
- [x] intervention-rate-over-training metric — does the policy learn to
      self-limit, or does it permanently lean on the shield? **Answer at
      this scale: no** — flat 0.19-0.22 across all 4 training checkpoints,
      all 5 seeds, no downward trend. See Phase 5c analysis below for why
      this is confounded and shouldn't be over-interpreted yet.
- [x] **Phase 5c analysis** — full writeup:
      `experiments/results/m6_kinodynamic/ANALYSIS.md`. **Critical
      finding**: `kinodynamic_shield` never advanced past curriculum
      stage 0 in any of 5 seeds (succ 0.00-0.07 throughout), while
      `no_shield`/`ape2_shield` both learned normally. Root cause
      identified (not just speculated — verified by reading the code):
      `m6_kinodynamic_shield.py` stores the RL actor's raw NOMINAL action
      in the replay buffer, but `env.py`'s `_step_velocity` can silently
      substitute a different actually-executed jerk sequence when the
      shield intervenes (~20% of steps, measured) — corrupting ~1/5 of
      stored transitions with an action label that doesn't match the
      real dynamics outcome. `ape2_shield` doesn't have this problem
      (its shield certifies the action BEFORE `env.step()`, so the
      stored action is always what was actually executed).
      **Do not scale up seeds further until this is fixed** — see the
      analysis doc's "Recommended next steps" for the fix and re-
      validation order.

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
  CLI — `codex exec -m gpt-5.6-sol -c model_reasoning_effort=<medium|high>
  -s workspace-write -C <repo root> -o <output-file>`. Use `high` effort
  for anything deriving a new closed-form/constraint (Phase 1, Phase 2
  design); `medium` once a pattern is proven and being reapplied
  (batch experiment scripts, more seeds of an existing script).
  **Verified 2026-07-17: `gpt-5.6-codex` is NOT usable on this account**
  (`invalid_request_error: not supported when using Codex with a ChatGPT
  account` — that model requires API-key auth). `gpt-5.6-sol` is the
  actual usable 5.6-series model here and is what "codex 5.6 series"
  means in practice for this loop — use it, don't re-try `gpt-5.6-codex`.
  Also: CLI must be ≥0.144.5 (`codex --version`); 0.140.0 rejects
  `gpt-5.6-sol` with "requires a newer version of Codex" — if a future
  session hits that error again, `npm install -g @openai/codex@latest`
  (verify the reinstall actually produced a working binary — one attempt
  here left a broken `@openai/codex-linux-x64` optional-dep install that
  needed a second explicit `npm install -g @openai/codex@latest` to fix).
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

- 2026-07-18 iter42: **user redirected again, correctly**: before running
  Codex's proposed A/B experiment, do proper academic grounding — align
  with URPlanner/the advisor's real recipe wherever alignment should
  happen, rather than keep experimenting in an internally-consistent
  bubble with nothing external to verify against. Went back to the
  advisor's REAL SOURCE CODE directly (`advisor_code/0801pretrain/
  Franka_Env_Scene2.py`, `main.py` — not just the `ASSUMPTIONS.md`
  summary, which is accurate but less precise than the source) and
  extracted the EXACT reward formula and constants, superseding item 7's
  "APPROXIMATED" framing:
  - `r = -delta_pos - delta_orient` (base pose term, using goal/TCP
    distances normalized by `dist_norm=3`, `orient_norm=180`) + step-
    improvement PBRS (`+0.05`/`-0.05` if normalized distance
    decreased/increased this step, `+0.03`/`-0.03` for orientation) +
    collision term (`-total_intersection_length/total_link_length`,
    `total_link_length=0.9101`, confirmed no separate ζ weighting,
    matching our default `zeta_c=1.0` numerically) + **`+1` reward EVERY
    STEP while fully within tolerance** (position AND orientation).
  - Position/orientation tolerance: `pe=0.02` (2cm — tighter than our
    `goal_tol=0.05`), `oe=6°`.
  - **`goal_dwell=50`, and — found by checking `main.py`'s actual
    training loop, not just the env file's own unrelated 5-episode
    smoke-test block which has an unrelated `MAX_EPISODES_STEPS=100` —
    the REAL per-episode budget is `MAX_EPISODES_STEPS=300`**, not our
    100. Their 50-step dwell requirement costs ~17% of a 300-step
    episode; naively importing `goal_dwell=50` into our 100-step budget
    without also scaling the episode length would cost 50% of the
    episode and be a mismatched, unbalanced transplant.
  **Why this matters for the RL-convergence question specifically**:
  with `goal_dwell=1` (ours), the episode ends the INSTANT tolerance is
  entered — there is no reward mechanism that specifically rewards
  DECELERATING AND STAYING near the goal, only reaching it once,
  momentarily. With `goal_dwell=50` + the sustained `+1`/step-while-
  in-tolerance term (real recipe), arriving and stopping is directly
  rewarded for up to 50 consecutive steps — a large, structural
  incentive our current reward completely lacks. This is a genuinely
  plausible, previously-unidentified structural explanation for "moves
  substantially, never learns to settle" that neither the commander's
  six attempts nor Codex's review surfaced — found only by going back to
  the actual reference implementation as the user directed.
  **Decision**: implement an aligned configuration (new opt-in mode —
  existing `reward_mode='uoar'`/`goal_dwell=1`/`max_steps=100` stay the
  untouched default, per this project's established "never silently
  retroactively change existing results" convention) matching the real
  formula/tolerances/dwell/episode-length as closely as sensible for our
  broader (non-static) goal region, test it with a modest episode count
  first (not committing to the full Codex-proposed 3000-episode/5-seed
  scale until this shows at least a preliminary positive signal), THEN
  proceed to Codex's nominal-vs-executed-action A/B on top of this
  aligned base rather than on the current, known-divergent-from-ground-
  truth reward.
- 2026-07-18 iter41: **the user, correctly skeptical of the commander's
  "stop this thread" recommendation, asked for an independent critical
  review by Codex (GPT-5.6) rather than accepting it — this caught real
  gaps the commander missed.** Wrote a full, rigorous report
  (`RL_CONVERGENCE_FULL_REPORT.md`) with the complete reasoning chain,
  a chart, and an explicit statistical-weakness section, then dispatched
  it to Codex for read-only, skeptical review (not implementation).
  Codex's verdict: **does NOT agree stopping is justified yet**, and
  found two real issues the commander's six-attempt investigation
  missed, both verified directly against the code (not taken on faith):
  1. **The `evaluate()` function hardcodes `env.set_difficulty(*STAGES[-1])`
     unconditionally** (`m6_kinodynamic_shield.py:172`) — every
     checkpoint's success/collision numbers across ALL SIX attempts were
     measured at the HARDEST curriculum stage (3 obstacles, 0.25 m/s),
     even though training never left stage 0 (0 obstacles). This is an
     inherited convention from `m4_shield.py`/`m3_curriculum.py`
     (deliberate there, for arms that DO progress through stages), but
     for `kinodynamic_shield` specifically it means most of the
     "success stuck near 0%" and "collision rate rose to 0.83" readings
     are confounded with obstacle-avoidance failure against dynamics the
     policy never trained on — NOT clean evidence about base-reaching
     capability. (The one exception: Phase 5f's standalone scripted-vs-
     trained diagnostic used its own separate, correctly-stage-0
     evaluation logic — that finding, trained policy 0% success/100%
     timeout even with no obstacles, stands independent of this bug.)
  2. **Attempt 1 (the "replay-buffer action-consistency fix," iter19-20)
     may not have been a fix at all.** Codex's argument, verified against
     `env.py`/`td3.py`/`m6_kinodynamic_shield.py` directly: under the
     standard RL framing where the safety shield is PART OF THE
     ENVIRONMENT'S TRANSITION FUNCTION (analogous to actuator
     saturation — a well-established pattern, you don't relabel actions
     post-hoc for a saturating actuator, you let the critic learn
     `Q(s, a_nominal)` and let it absorb the nonlinearity), storing the
     actor's raw NOMINAL action was arguably already correct, and
     relabeling to `u_executed` (clipped, lossy — many different nominal
     proposals collapse to the same executed label when the shield
     saturates) introduced a genuine NEW inconsistency: TD3's target
     computation (`td3.py`'s `a2 = self.actor_t(s2) + noise`) still
     produces NOMINAL actions for bootstrapping, while the critic is
     simultaneously trained on EXECUTED-action labels for the current
     transition — mixing two different action-coordinate systems within
     the same Bellman residual. **Because attempt 1 was applied before
     every subsequent attempt (2-6), no experiment in the whole
     investigation ever tested the simplest, most standard formulation:
     nominal-action replay + velocity observation present, together.**
  3. Reinforced the single-seed weakness with a sharper framing: "these
     are not six independent Bernoulli failures; most are correlated
     runs on one seed, with additive changes and a shared replay-action
     assumption" — they prove several configurations fail for THAT SEED,
     not population-level falsification. Recommended ≥5 seeds for any
     screening-level negative verdict.
  4. Rated the user's two flagged hypotheses: reward-shaping mismatch
     (§7.2, advisor's real PBRS vs. our continuous exponential bonus) —
     "credible secondary bottleneck," worked through actual numbers
     (median initial goal distance ≈0.28m, exponential term ≈0.015 there
     vs. base -0.282 — confirms it's small but the `-err` term stays
     dense, so "negligible signal" is too strong a claim); observation
     normalization (§7.1) — "plausible contributor but unlikely primary
     cause," noting the raw Cartesian features are already mostly within
     ~[-0.35,0.70], not orders-of-magnitude mismatched, and the same
     representation works fine for the other (working) arms. Also
     flagged an UNLISTED hypothesis: no remaining-time feature in the
     observation despite `max_steps=100` truncation being treated as a
     terminal state, which "could also encourage rushing."
  5. **Concrete recommended experiment** before accepting any stop
     decision: a paired A/B (nominal- vs executed-action replay
     labeling), 5 seeds each, 3000 episodes, stage-0-only training AND
     evaluation (fixing bug #1), richer diagnostics (minimum goal
     distance reached, radial velocity at closest approach, overshoot
     count — not just binary success), with an explicit, falsifiable
     pre-registered success criterion (≥15pp mean success advantage for
     nominal-action arm across most seeds to confirm; all 5 nominal-
     action seeds staying ≤5% through 3000 episodes to refute).
  All of Codex's code citations were independently verified by the
  commander directly (not accepted on faith) — every one checked out.
  **Commander's revised position: Codex's review is correct and
  identifies a genuine, previously-unnoticed gap. Withdrawing the "stop
  this thread" recommendation.** This is exactly the outcome the user's
  insistence on independent review was for. Next: fix the eval-protocol
  bug (cheap, unambiguously correct regardless of the larger question),
  then plan a scoped version of Codex's proposed experiment before
  committing to its full 5-seed/3000-episode/2-arm scale (~10+ hours of
  compute) — check with the user given the size of that commitment.
- 2026-07-18 iter40: independently verified Phase 5m thoroughly (commander
  — this was the largest single dispatch of the session). Reran pytest
  myself: 98/98 (warnings are benign third-party `diffcp` NumPy
  deprecations). Read `differentiable_qp.py` in full: `v1 =
  v1_offset_param + v1_matrix_param@jerk`, objective/constraints exactly
  mirror `solve_safety_qp`'s box-only path including the
  `box_lower - box_offset` shift; `problem.is_dpp()` gated; per-transition
  `_affine_map`/`_trajectory_quantities` loop is correctly necessary
  (these genuinely depend on each transition's own q,v,a — not something
  to vectorize away without touching Phase 2's verified core, which
  wasn't touched); the FINAL solve is a single `compiled.layer(...)` call
  with every parameter batched on the leading axis — genuinely native
  batching, confirmed by reading the call site, not a disguised loop.
  Read the batching-correctness test: uses 4+ genuinely different states
  (not one repeated), compares against individually-solved results —
  this is the test that would catch an indexing bug, and it's real.
  Read the `test_m6_projection_extracts_physical_state_from_actual_layout`
  test (Codex's own addition, beyond what I asked): monkeypatches
  `differentiable_v_exec` to capture what q/v/a get extracted from the
  REAL `env._state()` output (not a hand-built fake) across both
  `obstacles_in_state`/`closest_point_in_state` configs, asserting the
  extraction matches the env's true internal state — directly tests the
  exact risk I flagged (getting the observation slice indices right).
  Confirmed the training log: exactly one checkpoint, ep200, succ=0.00,
  stage=0, intervene=0.198, **wall time 4362.7s (72m43s) for 200
  episodes alone** — the timing smoke test's per-call bound (<2s, which
  passed) is consistent with this: ~7000 actor updates over 200 episodes
  × ~0.6s/call ≈ 4200s, matching the observed total almost exactly, so
  the slowdown is understood, not a mystery regression.
  **Verdict: Phase 5m is a genuine engineering success (correct,
  natively batched, well-tested differentiable QP) but a research
  dead-end for THIS problem** — the one checkpoint obtained shows the
  identical stuck pattern as every other attempt, at ~15-20x the
  computational cost of STE, making further iteration on this specific
  approach impractical without a substantial separate optimization
  effort with no evidence yet that it would help even if made fast.
  **This is the sixth consecutive negative result on the RL-convergence
  thread** (buffer bug, missing-v observation, missing-margin
  observation, hyperparameter sweep, STE, differentiable QP) — including
  the theoretically most-principled fix. Recommending to the user that
  this specific thread be closed for now.
- 2026-07-18 iter39: Phase 5m production differentiable-QP integration is
  complete. Added a cached DPP-compliant `CvxpyLayer` for the exact box-only
  safety QP, with native leading-batch parameters for every transition's
  distinct `box_matrix`/shifted bounds/`v1_matrix`/`v1_offset` (no Python
  loop of individual QP solves), and wired its physical-state-aware action
  projection into TD3 through a generic callback that takes precedence over
  the retained STE option. Forward, finite-difference-gradient, distinct-
  state batching, float32 backward, observation-layout, and TD3 default-path
  regressions pass; full suite **98/98** (314 third-party diffcp NumPy
  deprecation warnings). Repeated batch-64 forward+backward costs
  **0.41-0.44s**, despite the single-item prototype's ~5.9ms. The requested
  400-episode probe was stopped after its first checkpoint because its
  measured cost projected to ~2.4h versus ~12-13min for STE: ep200 was
  **succ 0.00, stage 0, intervention 0.198, collision 0.70**, with checkpoint
  `wall_s=4362.7` (72m43s); total foreground time before stopping about
  **77m48s**, so no ep400 checkpoint/JSON exists. This gives no evidence of
  improvement over STE at ep200 and exposes a severe training-cost regression,
  while the numerical and true-batching integration itself is verified.
- 2026-07-18 iter38: user chose Opus's "better-established alternative"
  (from the iter36 consult) — a differentiable QP layer, replacing the
  STE approximation with the real thing: backpropagate the actor
  gradient through the actual safety QP via implicit differentiation of
  its KKT conditions, rather than approximating with STE.
  Installed `cvxpy`+`cvxpylayers` (new deps, network access confirmed
  working, `cvxpylayers==1.2.0`). Built a standalone feasibility
  prototype (commander, `/tmp` scratchpad, not yet in the tracked repo)
  reusing the EXISTING, already-verified `_affine_map`/
  `_trajectory_quantities` from `safety_qp.py` (no second, independently
  -fallible re-derivation of the trajectory math) to construct the same
  box-only QP (`require_terminal_stop=False`, matching the real-time
  live-control config) in CVXPY, with `v_nom` as the differentiable
  `cp.Parameter`. **Feasibility confirmed**: forward solution matches
  the existing scipy solver to ~2e-7; backward analytic gradient
  matches a finite-difference check to ~4.6e-5. Timing: ~5.9ms per
  single forward+backward call — actually FASTER than the scipy solve.
  **Found a bigger complication than the initial timing test showed**:
  `box_matrix`/`box_offset`/`v1_matrix`/`v1_offset` depend on each
  transition's own `(q,v,a)` state (via the affine map's closure over
  q0,v0,a0), not just on `v_nom` — a real training batch (64 transitions,
  64 different states) needs PER-TRANSITION-varying constraint matrices
  passed as batched `cp.Parameter`s, not a single fixed structure with
  only `v_nom` varying (which is what the quick timing test actually
  measured — an unrealistic best case). `cvxpylayers` supports this via
  matrix-shaped batched parameters, but it's a materially bigger build
  than the initial prototype suggested. Dispatching the production
  module + integration to Codex next with a precise spec, since the
  hard feasibility question (is this even correct/differentiable) is
  now answered yes — what remains is real but well-scoped engineering.
- 2026-07-18 iter37: STE probe (task `bnm6hrzus`) finished. **Fifth
  consecutive negative result**: succ [0.0, 0.0, 0.0, 0.03], intervention
  rate flat ~0.20-0.22, isolated from the failed hyperparameter sweep
  (config confirms `dv_scale_mult=1.0, expl_noise=0.25, gamma=0.98` —
  original defaults). Independently verified (commander) directly from
  the JSON, not the training log's summary line.
  Ran one more diagnostic before stopping to reassess: a fresh 3000-step
  mini-run checking for basic brokenness (NaN/Inf in rewards, Q-value
  sanity) rather than another targeted fix. Reward stream is finite and
  in a plausible range (mean -0.30, [-0.61, 0.07], no NaN/Inf). Q(s,a)
  values are also finite but MONOTONICALLY DECREASING over the 3000
  steps sampled (-0.58 → -3.03 mean, still moving at the last sample,
  not plateaued) — not obviously pathological (a fresh critic learning
  a genuinely negative value scale under consistently negative reward
  looks like this too), but also not a clean bill of health; genuinely
  ambiguous from this short a probe.
  **Five well-motivated, Opus-validated or evidence-grounded
  interventions tried in sequence (replay-buffer consistency fix,
  missing-velocity observation, missing-margin observation,
  hyperparameter sweep, straight-through actor gradient), none resolved
  the convergence failure.** This is now a genuine research-scope
  decision, not a remaining-bug hunt — flagging for the user rather
  than proposing a sixth guess unprompted.
- 2026-07-18 iter36: STE implementation complete and tested (commander).
  Fixed a floating-point-precision test bug (`torch.equal` on
  numerically-but-not-bit-identical STE arithmetic — switched to
  `torch.allclose`) and a buffer-size bug (`learn()` silently no-ops
  below `5*batch` transitions; my first test draft only filled 64 for a
  batch of 32, needed >=160 — raised to 200) during test authoring;
  both caught by the tests themselves failing, not shipped silently.
  New `code/tests/test_td3.py` (first ever test file for this class):
  confirms default (`use_action_ste=False`) feeds q1 the raw actor
  output untouched by a deliberately-offset buffered action (hooked and
  inspected directly, not inferred); confirms the STE construction's
  forward/backward properties in isolation; confirms the flag produces
  a genuinely different actor update when buffered and raw actions
  differ; confirms it doesn't crash and actually updates parameters.
  Full suite: 86/86 (82 + 4 new). Wired `use_action_ste=(arm ==
  "kinodynamic_shield")` into `m6_kinodynamic_shield.py`'s `TD3(...)`
  construction — the OTHER two arms are unaffected (flag stays False
  for them, byte-identical to before). Smoke-tested (4 episodes, runs
  cleanly). Committing now, then launching a probe with STE alone
  (reverting `dv_scale_mult`/`expl_noise`/`gamma` to their ORIGINAL
  defaults, not combined with iter35's failed sweep) — isolates whether
  this specific, more structural fix has any effect, rather than
  confounding it with already-falsified hyperparameter changes.
- 2026-07-18 iter35: the combined hyperparameter sweep (`dv_scale_mult=8,
  expl_noise=0.1, gamma=0.99`, task `b5og1nul4`) finished. **Result:
  still stuck** — succ [0.0, 0.0, 0.0, 0.03] across the 4 checkpoints
  (indistinguishable from noise), collision rate rose to 0.83 (worse
  than baseline, expected given the more aggressive `dv_scale`).
  Confirms Opus's ranked hypothesis list items 1-3 (action scale,
  exploration noise, discount factor) do NOT explain the failure, even
  combined. This is now the FOURTH negative result on this thread
  (buffer bug, missing-v observation, missing-margin observation,
  hyperparameter sweep) — strengthens the case that something more
  structural is wrong, not a tuning issue.
  Separately, the user raised a sharp, independent point mid-investigation
  ("the safety layer can't just restrict, it needs to feed back into
  learning") that turned out to name a real, literature-documented
  mechanism: standard TD3's actor loss `-Q(s, actor(s))` always queries
  Q at the actor's RAW proposed action, but the replay buffer stores the
  ACTUALLY-EXECUTED (shield-corrected) action — in the ~20-30% of steps
  where the shield intervenes, the critic may never have observed the
  raw proposed action actually being executed, so `Q(s, actor(s))` can
  be extrapolating/unreliable there (an "OOD action" problem, cf.
  Fujimoto et al.'s BCQ 2019 framing). Escalated my first proposed fix
  (an auxiliary imitation loss toward the executed action) to Opus
  BEFORE implementing — correctly rejected: on intervened steps the
  executed action IS the shield's generic conservative brake, so
  imitating it risks teaching the actor to copy "safe but useless"
  behavior exactly where task progress matters most. Opus's
  recommendation instead: a **straight-through estimator** on the actor
  loss — forward pass evaluates Q at the buffered executed action
  (on-distribution, where the critic is accurate), backward pass routes
  the gradient to the raw actor output unchanged, giving a locally
  correct gradient without imitating the shield's specific choice.
  Implemented directly (commander, given the precision of the spec and
  that this is the first-ever change to `td3.py`, used across every
  experiment in this project, not just m6): new `use_action_ste: bool =
  False` constructor flag, default OFF (must be byte-identical to
  existing behavior for every other arm/script using this class); when
  True, `actor_action = raw_action + (a_batch - raw_action).detach()`
  replaces the raw actor output in the actor loss only. `a_batch` is
  the SAME buffer-sampled action already used for the critic loss — no
  new data collection needed. Writing tests next (regression: default
  behavior unchanged; gradient-flow: STE forward value exactly equals
  the buffered action while gradients still reach actor parameters)
  before any training probe.
- 2026-07-18 iter34: user chose to systematically investigate the RL-
  convergence problem rather than stop at the shield-is-verified
  checkpoint. Got a prioritized, evidence-linked sweep design from
  Opus (given all iter18-33 evidence, not generic RL folklore) — ranked
  candidates: (1) `dv_scale` — the ONE untuned placeholder the code
  itself flags, directly governs reach-vs-brake authority, top suspect
  given "moves substantially, never settles" is a classic mis-scaled-
  authority signature; (2) `expl_noise=0.25` — carried over unchanged
  from delta-q mode, likely miscalibrated for a rate/velocity action
  interface; (3) `gamma=0.98` — under a ~100-step episode with a
  dwell bonus only paid at the very end, the effective ~50-step horizon
  may undervalue late goal arrival; (4) dwell/goal-radius reward
  reweighting (lowest priority, not in the first combined test). Opus
  recommended ONE combined best-guess run first (not one-at-a-time —
  these are coupled through the same action interface, and isolating
  already-failed single fixes wastes runs at ~15-20min/run), and
  explicitly advised against introducing SAC or another algorithm yet
  (symptom pattern points at action-interface conditioning, not an
  exploration/entropy problem SAC would fix).
  Exposed `dv_scale_mult`/`expl_noise`/`gamma` as new CLI args on
  `m6_kinodynamic_shield.py` (previously hardcoded — `expl_noise=0.25`
  inline, `dv_scale`/`gamma` not exposed at all), threaded through
  `run_one` and recorded in the saved JSON `config`. Smoke-tested (4
  episodes, `--seed-salt 999`, `/tmp` output, not committed) — runs
  cleanly, config confirmed correctly recorded. Launching the real
  combined-config run next: `--dv-scale-mult 8 --expl-noise 0.1
  --gamma 0.99`, 800 episodes, kinodynamic_shield only.
- 2026-07-18 iter33: independently verified Phase 5j (commander) —
  reran pytest myself (82/82); read the full `env.py` instrumentation
  diff: `endpoint_violation` correctly checks RAW pre-clip `q_new`/`v_new`
  on all three paths (accept/fallback/emergency each now capture
  `raw_q_new`/`raw_v_new` before any clipping), `inter_sample_violation`
  correctly uses `kinodynamics.interval_within_limits` with the TRUE
  hard bounds (not the derated ones used internally for the safety
  margin) — the right, strict choice for an audit. Read
  `m6_safety_audit.py` in full: handles the observation-dimension
  mismatch between old checkpoints (pre-v-fix, pre-margin-fix) and the
  current env correctly by recovering each actor's expected `state_dim`
  from its own saved weights and projecting the current (larger)
  observation down to match — the SHIELD and DYNAMICS executing each
  audited step are the CURRENT, fully-fixed code; only the frozen
  actor's input is backward-projected. This is the right design (audits
  today's safety guarantee, not a stale one). **Result confirmed
  genuine and important**: zero endpoint violations, zero inter-sample
  violations, zero no-feasible-brake emergencies across all 6
  checkpoints and 18,000 evaluated steps, while the certified
  fallback/braking path was genuinely exercised on 19-21% of steps (not
  a trivial always-accept result) — Phase 1-4's core safety claim holds
  empirically at this scale, completely independent of whether the RL
  policy learned the task. This is the positive counterpart to the
  unresolved RL-convergence problem: **the shield works; getting TD3 to
  learn well while using it is the open problem, not the shield's
  soundness**. All three parallel Phase 5 threads (3000ep decisive
  probe, margin feature, safety audit) are now complete — writing a
  full Phase 5 synthesis next.
- 2026-07-18 iter32: the decisive 3000-episode WITH-v-fix comparison
  probe (task `b22voj1q3`) finished. **Definitive result**: independently
  checked all 15 checkpoints — max success EVER seen across the entire
  3000-episode run was 6.7% (one lucky episode out of 30 at ep1000,
  never sustained), final succ=0.00, never advanced past curriculum
  stage 0. This closes the question the iter27 checkpoint raised: the
  raw-velocity-observation fix alone is confirmed NOT sufficient, with
  the strongest possible evidence (full historical training budget,
  single seed, completely flat). Combined with iter29-30's inconclusive
  margin-feature probes, three interventions now tried
  (buffer-consistency fix, raw-v observation, velocity-margin
  observation) without resolving the RL convergence problem — this is
  a genuinely unresolved open question, not something a fourth quick
  guess is likely to crack. Committing this result now; Phase 5j's
  safety-guarantee audit (separate, more encouraging finding — see next
  entry) is still finishing.
- 2026-07-18 iter31: Phase 5j safety-guarantee audit measured all 6 committed kinodynamic checkpoints over 30 evaluation episodes each (18,000 stage-0 steps total): zero raw endpoint violations, zero exact inter-sample violations, and zero no-feasible-brake emergencies on every checkpoint; certified fallback remained meaningfully exercised at 19.00%-21.37%, and the full suite stayed green at 82/82.
- 2026-07-18 iter30: independently verified Phase 5i (commander) — read
  `velocity_margin`/`_velocity_margins` in full, confirmed correct
  (bisection over `[0,upper_bound]` calling `braking_witness_jerk`,
  called once per step on the PRE-step state, matches spec exactly);
  reran pytest myself, 82/82. Given the confirmed ~16-17ms/step
  overhead (nearly QP-scale) with NEITHER probe so far showing a clear
  learning benefit, reduced the default `bisection_iters` 10→4 via a
  new `margin_bisection_iters` constructor param (not a safety
  parameter — only affects observation-signal resolution, the
  underlying `braking_witness_jerk` safety checks are unaffected).
  Independently measured: iters=10 → 15.7ms/call, iters=4 → 7.5ms/call,
  iters=2 → 4.9ms/call (diminishing returns below 4, kept 4 as the
  default — coarse 1/16 resolution is plenty for an observation signal
  that isn't safety-critical). 82/82 still green after the change.
  Committing this and checking on the still-running 3000-episode
  decisive probe (v-fix only, no margin feature — that comparison is
  unaffected by any of this iter's changes since it started before
  they landed and Python doesn't hot-reload).
- 2026-07-18 iter29: Phase 5i directional velocity-margin observation implemented with the existing closed-form braking witness (10-round bisection, same derated bounds/horizon, no new QP); full suite 82/82. Paired 100-step timing under the probe's 6-thread settings was 23.460ms/step with margins versus 6.551ms stubbed, **+16.909ms/step** (unexpectedly QP-scale). The salt200 800-episode probe remained stuck at ep800: succ 0.000, stage 0, intervention 0.212 (collision 0.733), versus the prior velocity-only salt0 probe's succ 0.033, stage 0, intervention 0.215 (collision 0.767); no learning improvement at this scale, plus a serious final-block mean-episode wall regression (7.55s versus 0.99s). Throwaway results remain under `/tmp/m6_margin_probe`.
- 2026-07-18 iter28: user redirected me to re-read their ORIGINAL design
  message (start of this conversation, section 八 "你的構想") rather
  than just brute-forcing more compute — good call. That spec listed
  FOUR velocity-mode observation additions: current acceleration (have
  it), safety-filter correction amount (have it), dynamic obstacle
  prediction (separate, already-queued Phase 5 item), and **"正負方向的
  安全速度裕度" (positive/negative direction safety velocity margin)**
  — NEVER implemented. This is materially richer than raw `v` (which I
  added in iter26 with no clear effect): it's a POSITION- and braking-
  capability-aware "how much room before I can no longer guarantee
  stopping," directly relevant to Opus's hypothesis #2 (anticipatory
  braking is hard for a reactive policy without an explicit signal for
  it). Launched three things in parallel per the user's "都實驗" (do
  all of them) instruction: (1) the decisive 3000-episode WITH-v-fix
  comparison probe (task `b22voj1q3`, still valuable regardless of the
  margin work), (2) implementing the velocity-margin observation via a
  cheap bisection over the existing `braking_witness_jerk` (no new QP
  calls) — dispatched to Codex (task `bqgylx0j9`), includes its own
  800-episode validation probe, (3) the Phase 5c-deferred safety-
  guarantee audit (joint-limit/inter-sample violation rates on already-
  trained checkpoints — doesn't need RL to have converged, validates
  Phase 1-4's actual claim independent of task success) — **prompt
  written but DELIBERATELY NOT YET DISPATCHED**, because it also needs
  to touch `env.py` and dispatch (2) is concurrently editing that same
  file; launching both at once risks a merge conflict or one clobbering
  the other's changes. Queued for after (2) lands and commits.
- 2026-07-18 iter27: independently verified the velocity-observation
  fix's 800-episode probe (commander) — **no meaningful improvement**:
  succ 0.000/0.033/0.000/0.033 at ep200/400/600/800, coll 0.667-0.767,
  intervention rate still 0.195-0.216, stage still stuck at 0. This is
  statistically indistinguishable from the pre-fix pattern at this
  sample size (30 eval episodes/checkpoint). The fix (adding `v` to the
  observation) is still objectively correct and worth keeping — a
  policy that cannot perceive its own velocity is a real deficiency
  regardless of whether it turns out to be THE bottleneck — but it does
  not appear to be sufficient alone to unblock learning, at least not
  within 800 episodes. Important caveat before concluding "the fix
  failed": the earlier 3000-episode NO-fix run (iter23) is the strongest
  evidence of a truly flat, non-improving trend — this probe is only
  800 episodes WITH the fix, so it can't yet distinguish "the fix
  doesn't help" from "the fix helps convergence RATE but 800 episodes
  still isn't enough to see it," since the pre-fix 800-episode run
  ALSO looked statistically similar to its own pre-fix 3000-episode
  run's early checkpoints. A fair test needs a 3000-episode WITH-fix
  run to compare against the 3000-episode NO-fix baseline's flat trend.
  **This is now the third hypothesis round (needs-more-time: refuted;
  missing-velocity-observation: inconclusive/likely-insufficient-alone)
  on a debugging thread that has consumed substantial time.** Flagging
  this as a natural checkpoint to consult the user on how much further
  to invest before either running the decisive 3000-episode comparison
  or stepping back to document this as an open limitation and moving to
  other Phase 5 work (the deferred safety metrics from Phase 5c, which
  don't require RL convergence to measure and more directly validate
  Phase 1-4's actual claims).
- 2026-07-18 iter26: implemented the fix directly (commander, small and
  well-understood — no Codex round-trip needed). `env.py`: added
  `self.v / DQ_MAX` (7 floats) to the velocity-mode observation,
  inserted before the existing `self.a` term; `state_dim`'s velocity-mode
  addition is now `16` (was `9`), commented `7 v + 7 a + 2 shield flags`.
  Updated `test_env_velocity_mode.py`'s two affected assertions
  (state_dim delta, zero-state check) to `16`, and added a direct
  assertion that `state[-16:-9] == env.v/DQ_MAX` after a real step — the
  test that actually proves this fix does what it's supposed to, not
  just that the dimension count changed. Full suite: 79/79 green (same
  count, two tests renamed/updated in place rather than duplicated).
  Committing, then launching an 800-episode single-seed validation
  probe (same seed 0, same everything else) — if velocity really was
  the missing piece, this should show CLEAR improvement over the
  pre-fix baseline at the same checkpoint (succ was 0.00-0.03 throughout
  both prior runs), not just "still stuck, marginally different."
- 2026-07-18 iter25: escalated the diagnostic findings to Opus for a
  ranked root-cause call before spending more compute guessing. Opus's
  read of the evidence (moves substantially, 0% collision, ALWAYS times
  out at exactly max_steps, even in noise-free eval) pointed to a POMDP
  gap: velocity mode makes `v_t` genuine dynamic state the policy needs
  to decide when to decelerate, and asked me to check whether `v_t` is
  even IN the observation. **It is not.** Checked `env.py::_state()`
  directly (line 386-389): velocity-mode observation appends `self.a`
  (acceleration, normalized) but never `self.v` (velocity) — **a real
  bug in my own Phase 3 design spec**, not a Codex implementation error;
  Codex built exactly what I specified, I simply forgot to include
  velocity when I wrote that dispatch. My own exploration-noise-
  accumulation hypothesis (iter24-adjacent reasoning) was ranked by
  Opus as a real but minor contributor, not sufficient to explain a
  100%-noise-free-eval failure — correctly deprioritized. Fixing this
  directly: add `self.v` (normalized by `DQ_MAX`) to the velocity-mode
  observation alongside the existing `self.a` term (`state_dim`'s `+9`
  becomes `+16`). This is the highest-confidence fix found so far in
  this debugging arc — dispatching it now with a validation probe.
- 2026-07-18 iter24: stage-0 scripted reachability diagnostic (`m6_diag_scripted_policy.py`) found scripted success/collision/timeout = 38/0/62% over n=100 versus trained-3000ep TD3 = 0/0/100% over n=50; scripted velocity norm median/p90/max = 0.086/0.242/0.501 rad/s (DQ_MAX RMS fraction 1.43/4.03/7.66%) versus trained = 0.201/0.380/0.547 rad/s (3.35/6.33/8.72%), and scripted termination steps mean/median = 90.0/100 overall (successful episodes 73.7/76) versus trained = 100/100. Thus stage 0 is physically achievable at a reasonable 38% even with this simple policy, the trained actor is moving rather than stationary but wanders to timeout, and obstacles cannot explain the failure because stage 0 has none and neither cohort collided.
- 2026-07-18 iter23: the 3000-episode probe (task `b7bq24gs9`) finished.
  **Credit-assignment/"just needs more time" hypothesis REFUTED**: 14
  checkpoints from ep200 to ep3000 all show succ 0.00-0.03, curriculum
  stuck at stage 0 the entire run, intervention rate flat-to-rising
  (0.195→0.216, never declining). This is NOT a training-budget problem
  — something else is genuinely blocking learning, independent of the
  Phase 5d buffer fix (which is confirmed correct but evidently
  insufficient alone) and independent of episode count. Verified
  (commander): read the full log, confirmed the final checkpoint
  (succ=0.00, ep3000) and the flat trend across all 14 points myself,
  not just the summary line. Committed the result under
  `experiments/results/m6_kinodynamic_longrun/` (kept separate from the
  tracked 800×5 comparison set — this is a diagnostic probe, not a
  result to average into the paper-grade numbers).
  Dispatched a DIAGNOSTIC investigation (not a fix attempt) to
  disambiguate remaining hypotheses: is the task physically achievable
  at all under the current placeholder `DDQ_MAX`/`DDDQ_MAX`/`dv_scale`
  within `max_steps` (test via a scripted, non-learning policy), is the
  trained policy exhibiting a degenerate "don't move" local optimum
  (check actual velocity magnitudes reached), or is obstacle-avoidance
  specifically the bottleneck (termination-cause breakdown: collision vs
  timeout vs success).
- 2026-07-18 iter22: launched the credit-assignment-hypothesis probe —
  `kinodynamic_shield`, fixed replay buffer, single seed (`--seed-salt 0`,
  same seed 0 as the earlier runs, now with the fix), **3000 episodes**
  matching the historical M4 budget, `--out
  experiments/results/m6_kinodynamic_longrun` (separate dir, not mixed
  with the committed 800-episode 5-seed results), only process running
  so no oversubscription risk this time (6 threads, machine otherwise
  idle). Task id `b7bq24gs9`, log
  `experiments/logs/m6_kinodynamic_longrun.log`. Projected ~60-70min at
  the post-fix ~1.37s/ep rate. Next: check whether curriculum ever
  advances past stage 0 and whether intervention rate trends down given
  enough training time — this is the actual test of "needs more budget"
  vs. "still broken."
- 2026-07-18 iter21: independently verified iter20's fix (commander) —
  diff is exactly the specified change (`env.py`: `_last_executed_action
  = clip((v_new-v_old)/dv_scale, -1, 1)`, set unconditionally after all
  three branches converge; `m6_kinodynamic_shield.py`: uses it in place
  of raw `a` only for the `kinodynamic_shield` arm, placed once before
  the shared buffer-add call so it covers both warmup and post-warmup
  paths); reran pytest myself, 79/79. Codex's probe (n=1 seed, 400
  episodes, pre- vs post-fix) was honestly reported as inconclusive —
  the fix is real and necessary, but that tiny single-seed comparison
  can't establish whether it's SUFFICIENT alone; correctly did not
  overclaim.
  **Additional hypothesis worth testing before concluding anything is
  "broken"**: velocity-mode control is a strictly harder credit-
  assignment problem than delta-q — the policy's action now influences
  the task-relevant flange position through TWO integration steps
  (u→v→q) instead of one (u→q directly), and `dv_scale`'s conservative
  placeholder value means each step's achievable velocity change is
  small, stretching the effective task horizon further. The historical
  delta-q+shield baseline (`m4v2.log`) itself needed most of its
  3000-episode budget to reach good performance — 800 episodes may
  simply be too short for velocity mode to bootstrap at all, independent
  of the (still worth having fixed) buffer bug. Testing this directly:
  launching a single-seed, 3000-episode `kinodynamic_shield` run (with
  the fix applied, matching the historical M4 episode budget) rather
  than immediately re-running the full 5-seed comparison — cheaper way
  to falsify/support the "just needs more time" hypothesis before
  committing more compute to a possibly-still-broken setup.
- 2026-07-18 iter20: fixed Phase 5c's kinodynamic replay-buffer action-
  consistency bug by exposing the effective executed velocity action and
  storing it for both warmup and learned transitions. The prescribed
  400-episode salt100 probe was **still stuck** at stage 0: ep400 succ 0.00
  and intervention rate 0.214, versus the pre-fix salt0 seed0 checkpoint's
  succ 0.033, stage 0, and intervention rate 0.201. The fix restores data
  integrity, but this short one-seed probe shows no learning improvement and
  indicates another blocker likely remains.
- 2026-07-18 iter19: `kinodynamic_shield` (task `bqd7hcxt1`) finished, all
  5 seeds, independently verified (JSON structure + pytest 78/78).
  Confirmed consistently across ALL 5 seeds (not seed0-specific noise):
  succ 0.00-0.07, collision 0.47-0.73, stuck at curriculum stage 0 the
  entire run. Committed. All three Phase 5b arms now complete — wrote
  the full Phase 5c analysis (`experiments/results/m6_kinodynamic/ANALYSIS.md`),
  organized per §4's four metric categories, with real numbers pulled
  from all three result JSONs. **Found and verified (by reading the
  actual code, not guessing) a real replay-buffer action-consistency bug**
  specific to the `kinodynamic_shield` arm — see the Phase 5 checklist
  entry above and the analysis doc for full detail. This is very likely
  why that arm failed to learn at all, and it should be fixed and
  re-validated before any further seed scale-up (n=8-10 would just burn
  compute on a confounded setup otherwise). Next: dispatch the fix.
- 2026-07-17 iter18: `no_shield` (task `bd0iqcpc3`) finished, independently
  verified — 5 seeds, all 800 episodes, succ range 0.17-0.53 / coll
  0.37-0.63; seed2 never advanced past curriculum stage 0 (legitimate
  training variance, rolling success never hit the 70% advance
  threshold). Committed. `kinodynamic_shield` seed0 finished 800
  episodes in 1099.7s (~1.37s/ep, matches the post-fix estimate) but its
  success rate stayed near 0 (0.00-0.07) and it NEVER advanced past
  curriculum stage 0, unlike `no_shield`/`ape2_shield` which both
  reached stage 1-2 — worth flagging honestly in the Phase 5c writeup as
  a real finding (possibly the velocity-action interface/hyperparameters
  need their own tuning pass, not a like-for-like drop-in with delta-q's
  tuned settings) rather than glossing over it. Collision rate stayed
  high (63-77%) — expected, since the shield only guards joint-limit/
  kinematic safety, not obstacle avoidance, which is still the policy's
  job to learn. `load average` back down to ~11-17, no new
  oversubscription. Continuing to monitor `kinodynamic_shield`'s
  remaining 4 seeds.
- 2026-07-17 iter17: thread-limiting fix confirmed working well.
  `ape2_shield` (task `bzvrh4qk7`) finished cleanly — ~220s/seed (vs.
  thousands of seconds/seed under the earlier oversubscription),
  independently verified (commander): all 5 seeds present, each reached
  final curriculum stage (stage 2), final succ/coll range 0.20-0.33 /
  0.27-0.53 across seeds — meaningful cross-seed variance, exactly the
  kind of spread n=2 would have risked missing. Committed
  `m6_uoar_ape2_shield_salt0.json`. `no_shield` ~90% done (seed 4/5).
  `kinodynamic_shield` now measuring ~1.39s/episode (600 eps in 831.3s)
  — better than even the original smoke-test estimate, ~1.5h projected
  total for all 5 seeds. Both remaining arms progressing normally,
  continuing to monitor.
- 2026-07-17 iter16: user checked in ~4h49m after the iter15 relaunch.
  Found the three parallel processes were badly CPU-oversubscribed —
  `load average 28` on a 16-core machine, each process using ~38 threads
  / 450-620% CPU simultaneously (numpy/scipy/torch each spawning their
  own BLAS/intra-op thread pools with no cap, ×3 concurrent processes).
  Measured real throughput was ~38.7s/episode for `kinodynamic_shield`
  under this contention — ~7x slower than the smoke-tested 5.6s/ep,
  which would have meant ~43h total instead of the planned ~6.2h.
  Progress at kill time: `no_shield` ~48% through (seed 2/5, ep400),
  `ape2_shield` ~65% (seed 3/5, ep200), `kinodynamic_shield` only ~10%
  (seed 0/5, ep400/800) — none had reached a seed boundary with results
  flushed to disk (the script only assembles/writes its JSON after ALL
  seeds of a run finish), so killing lost that in-progress compute, but
  nothing corrupted or silently wrong was ever written. **Lesson: launching
  N heavy ML training processes in parallel without capping each one's
  thread count is a real trap — always set `OMP_NUM_THREADS`/
  `MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` (and check `uptime`/`top`
  shortly after launch) when running more than one such job
  concurrently on a shared machine.** Relaunched all three arms with
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  NUMEXPR_NUM_THREADS=4` (3×4=12 of 16 cores, leaving headroom); 1-min
  load average dropped from 28 to 16 within 30s of relaunch, per-process
  CPU% dropped modestly (~1645%→~1262% combined) — a real but partial
  fix (torch's own intra-op pool doesn't fully respect these env vars in
  every build); will re-measure actual throughput at the next check-in
  before deciding whether further tuning or an episode-count reduction
  is needed. New task ids: `no_shield` `bd0iqcpc3`, `ape2_shield`
  `bzvrh4qk7`, `kinodynamic_shield` `bqd7hcxt1`.
- 2026-07-17 iter15: session was interrupted mid-launch of Phase 5b.
  Root cause: the `no_shield` run had been started with manual
  `nohup ... & disown` inside a Bash call instead of the harness's own
  `run_in_background` tool parameter — it silently died partway through
  (seed3 incomplete, seeds 4 never started, no final JSON) once that
  tool call's session was torn down, despite `nohup`/`disown`. **Lesson:
  always launch long background jobs via the Bash tool's
  `run_in_background: true`, never manual `nohup`/`&`/`disown` — the
  latter does not reliably survive in this sandboxed environment.**
  Confirmed no partial/misleading results were committed (the script
  only writes its final JSON after all arms/seeds finish, so the
  incomplete run left no corrupt tracked artifact — only a stray log
  file and 3 orphaned `.pt` checkpoints, both harmless, both about to be
  overwritten). Relaunched all three arms cleanly, properly
  harness-tracked this time: `no_shield` (task `bwucb5kv0`),
  `ape2_shield` (task `b8jjoyv2z`), `kinodynamic_shield` (task
  `bf7w395o8`), same `--episodes 800 --seeds 5 --seed-salt 0` sizing as
  before, logs at `experiments/logs/m6_no_shield.log`,
  `m6_ape2_shield.log`, `m6_kinodynamic_shield_run.log`. Next iteration:
  wait for completion notifications (or check log tails on a long
  fallback wakeup), then move to Phase 5c analysis.
- 2026-07-17 iter14: verified Phase 5a's `m6_kinodynamic_shield.py`
  (commander) — full read-through, correct per-arm mechanism separation
  (no_shield: plain TD3; ape2_shield: faithful port of m4_shield.py's
  deployment ladder; kinodynamic_shield: safety inside env.step(), no
  APE2 wrapper needed); `pytest` unaffected (78/78). Ran my own smoke
  test covering ALL THREE arms (Codex's smoke only covered
  kinodynamic_shield) at 8 episodes/1 seed each — all ran cleanly,
  timing data collected (see Phase 5a's roadmap entry). Committed the
  script. Sized and launched Phase 5b (800 episodes x 5 seeds, 3
  parallel background processes) — see the Phase 5b sizing note above
  for the reasoning. This will run for hours; subsequent loop iterations
  should just check progress via the log tails and the background-task
  completion notifications, not redo this analysis.
- 2026-07-17 iter13: added `experiments/m6_kinodynamic_shield.py`, a three-arm plain-TD3 / APE2-shield / velocity-mode kinodynamic-shield comparison with M4's unchanged tabletop curriculum and evaluation protocol, disjoint salted per-arm seeds, and Phase-5 safety/efficiency/intervention/timing metrics; a 15-episode, 1-seed kinodynamic-only smoke ran end-to-end and emitted sane JSON in 80.79s (the other two arms were not exercised in this time-bounded smoke), and the full suite remains green at 78 passed.
- 2026-07-17 iter12: independently re-verified iter11's Phase-4 claim
  (commander session, extra scrutiny — this was the phase that already
  caught me proposing an unsound bound once). Reran `pytest` myself: 78
  passed. Hand-derived `chord_deviation_bound`'s closed form myself from
  scratch (`d(τ)=c1τ+0.5a0τ²+(j/6)τ³`, `c1=-0.5a0h-jh²/6`) and confirmed
  it matches the implementation exactly; independently brute-force
  checked the reversal regression case's claimed `2√3/9` value against a
  200k-point dense scan — matched to 9 decimal places. Independently
  re-ran the Cartesian-through-FK adversarial checks myself (not just
  trusting the test suite's assertions) with my own dense-sampling
  script: reversal case ratio **2.427**, multi-joint simultaneous
  reversal ratio **4.296** — both match Codex's reported numbers closely
  and confirm bound > empirical truth in both cases (sound, with
  measured conservatism in the 2-4x range, smaller than the original
  `eps_lin`'s own ~30x — reasonable, not alarming). Confirmed
  `downstream_length_bounds` was factored out of `chord_error_bound`
  without changing its behavior (same R_i used by both terms, as
  required). Phase 4 core is genuinely sound and done; the Phase 5-scale
  shield-comparison experiment is correctly left for Phase 5.
- 2026-07-17 iter11: Phase 4 cubic collision-linearization core landed: exact per-joint deviation bound, unchanged Term A plus additive Term B composition, and velocity-only collision-margin wiring; adversarial validation covers (a)-(c) Cartesian-through-FK and (d) by the explicitly allowed joint-space dense-scan substitute, with Cartesian bound/true ratios 2.425-4.296 and full pytest **78 passed**; the Phase-5-scale shield-comparison experiment remains unchecked.
- 2026-07-17 iter10: before dispatching Phase 4, proposed a chord-error-
  bound inflation approach for cubic joint trajectories, escalated it to
  Opus for a soundness check (per this roadmap's own escalation trigger
  for architecture-level collision-certification decisions) — Opus found
  it UNSOUND via a clean counterexample (folding a first-order reversal
  effect into a second-order formula) and gave a corrected triangle-
  inequality decomposition instead. Full exchange recorded in the Phase 4
  design note above so it isn't re-proposed. Dispatching the corrected
  design to Codex next, with the adversarial validation tests as the
  actual acceptance gate (agent id `ad222dfb467f95545` if this session
  needs to resume that consult).
- 2026-07-17 iter9: independently re-verified iter8's Phase-3b claim
  (commander session, extra scrutiny given this is the first change to
  `env.py`) — diff scope correct (`env.py` + one new test file only);
  reran `pytest` myself twice (no flakiness): **70 passed** both times.
  Read the full `env.py` diff line-by-line: confirmed the `action_mode`
  branch is a pure early-return in `step()` — the pre-existing `delta_q`
  path's code is untouched, `state_dim`/`_state()` additions are strictly
  conditional on `action_mode=="velocity"`, and a dedicated test
  (`test_default_and_explicit_delta_q_modes_remain_identical`) asserts
  `not hasattr(default_env, "v")` — the new attributes aren't even
  created in default mode. Confirmed `_step_velocity` correctly reuses
  the exact same collision/reward helpers as the original `step()`
  (verified the `-5.0` collision penalty convention matches, not
  double-applied — `_reward()` itself has no collision term). Verified
  the accept/fallback/emergency three-tier degradation is genuinely
  reachable and distinct (test file has one scenario per tier, confirmed
  by reading the QP-then-membership-check control flow). Phase 3b
  genuinely done. Dispatching Phase 4 next.
- 2026-07-17 iter8: Phase 3b landed as an opt-in velocity-action mode with short-horizon box-only QP certification, derated terminal-membership gating, deterministic fresh-witness braking fallback, emergency zero-jerk handling, acceleration/shield observations, and intervention reward penalty; full `pytest` confirmed **70 passed** (61 existing + 9 Phase-3b tests).
- 2026-07-17 iter7: Phase 3a landed with shared braking-witness search, first-jerk fallback API, and optional box-only safety QP certification; full `pytest` confirmed **61 passed** (58 existing + 3 Phase-3a tests).
- 2026-07-17 iter6: before starting Phase 3, sanity-checked the shield's
  actual behavior numerically (commander, direct `solve_safety_qp` calls)
  — found the hard `v_N=0,a_N=0` terminal equality at real-time-affordable
  `n_steps=3` suppresses nearly all motion. Escalated the terminal-set
  design to an Opus consult (per this roadmap's own §5 escalation
  trigger for "terminal braking set choice") rather than deciding
  unilaterally; Opus confirmed the diagnosis and the proposed fix
  (terminal SET membership via cheap derated `braking_feasible`, not a
  terminal EQUALITY inside the expensive QP), plus added a last-witness
  fallback for recursive feasibility across replans. Full design recorded
  in the corrected Phase 3 entry above. Dispatching Phase 3a (the small,
  additive `kinodynamics`/`safety_qp` amendments this design needs) next
  — NOT touching `env.py` until 3a is verified.
- 2026-07-17 iter5: independently re-verified iter4's Phase-2 claim
  (commander session) — diff scope correct (`safety_qp.py` +
  `test_safety_qp.py` only, roadmap edits additive), reran `pytest`
  myself: **58 passed**. Read `safety_qp.py` in full: the design is
  smarter than what I specified in the dispatch — it exploits that the
  multi-step trajectory is an exact LINEAR function of the jerk sequence
  (fixed initial condition, LTI system), so the collocation constraints
  fed to SLSQP are exactly affine, not an approximation; the only
  approximation is between-collocation-point extrema, which is exactly
  what the post-hoc `interval_within_limits` certify+retry loop exists to
  catch. Confirmed the retry logic is actually exercised (not just
  happy-path) via `test_failed_exact_certification_retries_with_tightened_margin`'s
  monkeypatch-forced-first-failure test. Ran my own direct solve-time
  benchmark before green-lighting Phase 3 (see Phase 2's perf finding in
  §3 above) — dispatching Phase 3 next with `n_steps=3` per that finding.
- 2026-07-17 iter4: Phase 2 multi-joint N-step SLSQP jerk filter landed
  with collocation constraints, exact Phase-1 interval certification,
  retry-with-tightened-margin fallback, and infeasible-safe returns; added
  four QP tests including a 7-joint solve-time smoke guard; full `pytest`
  confirmed **58 passed** (54 existing + 4 Phase-2 tests).
- 2026-07-17 iter3: independently re-verified iter2's Phase-1 claim
  (commander session, not Codex self-report) — `git diff --stat` scope
  matches the constraint list exactly (kinodynamics.py, test_kinodynamics.py
  new; panda.py/ASSUMPTIONS.md additive only); reran
  `pytest tests -q` myself, confirmed **54 passed**; read `kinodynamics.py`
  and `test_kinodynamics.py` in full, hand-verified the core regression
  test's numbers (q(0)=q(1)=0, true interior max ≈0.385 > q_max=0.3) —
  math checks out, not tautological. Phase 1 genuinely done. Installed
  `scipy` into `code/.venv` for Phase 2 (no solver existed at all before).
  Dispatching Phase 2 to Codex next.
- 2026-07-17 iter2: Phase 1 single-joint jerk dynamics, exact interval
  extrema/limit certification, conservative N-step braking witness, Panda
  accel/jerk placeholders, and 8 regression tests landed; full `pytest`
  confirmed 54/54 green (46 existing + 8 new).
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
