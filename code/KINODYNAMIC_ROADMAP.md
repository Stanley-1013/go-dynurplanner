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
- [ ] wire `grid_td3.py`/`forecast.py` obstacle prediction into the N-step
      safety horizon
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
- [ ] intervention-rate-over-training metric — does the policy learn to
      self-limit, or does it permanently lean on the shield? (data will
      exist once Phase 5b's per-block `train_intervention_rate` history
      is analyzed — not yet analyzed as of this entry)

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
