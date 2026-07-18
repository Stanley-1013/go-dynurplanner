# Phase 5c — M6 three-arm comparison, first-pass analysis

> **2026-07-18 SECOND UPDATE — read this first, supersedes the update
> below it.** After the first update (through Phase 5j) was written, the
> user asked to systematically pursue the RL-convergence question rather
> than stop. Six interventions in total were tried, each with real
> justification, each independently verified, in this order:
> 1. Replay-buffer action-consistency fix (real bug, real fix)
> 2. Missing raw-velocity observation (real POMDP gap) — tested up to a
>    decisive 3000-episode run: max success ever seen was 6.7%, once,
>    never sustained
> 3. Missing safety-velocity-margin observation (from the user's own
>    original design spec) — no improvement, added real per-step cost
> 4. A prioritized, Opus-designed hyperparameter sweep (action scale,
>    exploration noise, discount factor, combined) — no improvement,
>    collision rate got worse
> 5. A straight-through estimator on the actor's gradient (addressing a
>    real, literature-documented extrapolation-error mechanism the user
>    identified: TD3's actor loss queries Q at raw actions the critic may
>    never have seen executed in shield-heavy regions) — no improvement
> 6. **A real differentiable QP layer** (`cvxpylayers`, KKT-based implicit
>    differentiation through the actual safety QP, not an approximation)
>    — the theoretically most-principled fix available, explicitly
>    endorsed by Opus over the imitation-loss alternative. Verified
>    mathematically correct (forward matches the scipy solver to ~2e-7,
>    gradient matches finite-difference to ~4.6e-5) and correctly
>    natively-batched in production. **Still no improvement** at the one
>    training checkpoint obtained (succ=0.00, ep200, identical pattern to
>    every prior attempt) — and ~15-20x more expensive per training step
>    than the STE alternative (72m43s for 200 episodes vs. ~3-6 min),
>    making further iteration on this specific approach impractical
>    without a separate, substantial optimization effort that has no
>    evidence yet of paying off even if completed.
>
> **This is six consecutive negative results, including the most
> theoretically sound one.** The commander's recommendation: close this
> specific thread. The shield (Phase 1-4, Phase 5j) remains a complete,
> independently verified, empirically-validated-at-scale contribution
> regardless of this outcome — the RL-convergence question is real,
> interesting, and unresolved, but has now had a thorough, good-faith,
> expert-informed attempt and should be documented as an honest open
> limitation (matching this project's existing C4-style precedent for
> honest negative results) rather than pursued further via more targeted
> guesses. See `KINODYNAMIC_ROADMAP.md` iter18-40 for the complete
> evidence trail.

> **2026-07-18 FIRST UPDATE.** Everything below this notice
> was written during the investigation and is kept for the full trail,
> but here is where things actually stood after Phase 5c through 5j
> (superseded by the update above, kept for the historical record):
>
> **Phase 1-4 (the kinodynamic certified safety shield itself) is DONE,
> independently verified at every step, and now empirically validated
> at scale.** Phase 5j's safety audit rolled out all 6
> `kinodynamic_shield` checkpoints for 18,000 total steps and found
> **zero joint-limit endpoint violations, zero inter-sample violations,
> zero no-feasible-brake emergencies** — while the certified
> fallback/braking path genuinely fired on 19-21% of steps (not a
> trivial always-pass result). This is the actual claim Phase 1-4 make,
> and it holds. Two real design bugs were caught and fixed BEFORE they
> shipped (a soundness bug in an early collision-bound proposal, an
> over-conservative terminal-braking-set design) — both via Opus second
> opinions before dispatch, not after something broke in production.
>
> **What is NOT done: getting TD3 to learn the task well while using
> velocity-mode actions.** Three well-motivated interventions were
> tried in sequence, each backed by real evidence or a design-spec gap,
> none alone resolved it:
> 1. Fixed a genuine replay-buffer action-consistency bug (the buffer
>    stored the RL's nominal action instead of what the shield actually
>    executed) — real bug, real fix, no measurable learning change.
> 2. Added raw velocity to the observation (it was missing entirely —
>    a real POMDP gap) — tested at 800 AND a decisive 3000 episodes
>    (matching the historical training budget): max success ever seen
>    across all 15 checkpoints of the 3000-episode run was 6.7%, one
>    lucky episode, never sustained. Confirmed insufficient alone.
> 3. Added the "positive/negative safety velocity margin" observation
>    from the user's own original design spec (also missing entirely) —
>    tested at 800 episodes, no clear improvement, plus a real
>    ~16ms/step cost (mitigated to ~7.5ms via a tuned bisection budget,
>    still non-trivial).
>
> The task IS physically achievable at the step budget used (a scripted
> non-learning policy succeeds 38% of the time), obstacle-avoidance is
> ruled out as the blocker (the stuck stage has zero obstacles), and the
> trained policy is not stationary (it moves, sometimes faster than the
> scripted policy) — it just never learns to arrive and stop. This
> looks like a genuine, still-open RL research question (credit
> assignment / exploration statistics / reward shaping specific to a
> double-integrator action space), not a bug waiting to be found. See
> the iter18-33 entries in `KINODYNAMIC_ROADMAP.md`'s status log for the
> full evidence trail before proposing a fourth quick fix — three
> reasonably strong hypotheses are now spent.
>
> **Recommendation**: treat the shield (Phase 1-4) as a complete,
> verified contribution independent of this open RL question. Whether
> and how much further to invest in the RL-convergence problem is a
> scope decision for the user, not something to keep guessing at via
> single-seed probes — it likely needs either a systematic
> hyperparameter/architecture sweep or a different RL algorithm choice,
> which is a different KIND of work than what this loop has been doing.

**Run config**: 800 episodes × 5 seeds per arm, `--seed-salt 0`, tabletop
task, `n_obstacles=3`, `reward_mode=uoar`. Source JSONs:
`m6_uoar_no_shield_salt0.json`, `m6_uoar_ape2_shield_salt0.json`,
`m6_uoar_kinodynamic_shield_salt0.json` (all in this directory, all git-
tracked, all independently verified by the commander — diff scope,
5/5-seeds-present, pytest unaffected — before this analysis was written).

This is a **first pass**, not the paper-final numbers (roadmap Phase 5
targets n=8-10 for that). Read the headline finding below before
scaling up seed count further — there is a concrete bug to fix first.

## Headline finding: the kinodynamic_shield arm did not learn, and there's a likely concrete cause

All three arms, final-episode-block eval, mean ± population stdev across 5 seeds:

| Arm | Success rate | Collision rate | Final curriculum stage (per seed) |
|---|---|---|---|
| `no_shield` | 0.333 ± 0.145 | 0.493 ± 0.100 | [1,1,0,1,1] |
| `ape2_shield` | 0.260 ± 0.049 | 0.460 ± 0.098 | [2,2,2,2,2] |
| `kinodynamic_shield` | **0.027 ± 0.025** | **0.593 ± 0.098** | **[0,0,0,0,0]** |

`kinodynamic_shield` never advanced past curriculum stage 0 (the
*easiest* stage — 0 obstacles/speed per `STAGES[0]=(0,0.0)`) in **any**
of its 5 seeds, and its success rate stayed near zero (0.00–0.07)
throughout training. This isn't seed noise — it's consistent across all
5 seeds independently. It's also not explained by task difficulty: it
never even reached the harder stages `no_shield`/`ape2_shield` were
being evaluated on, yet still collided *more* (0.593 vs. 0.493/0.460)
and succeeded far less.

**Leading hypothesis, verified by reading the code (not speculation):**
`m6_kinodynamic_shield.py` line 237 stores the RL actor's raw nominal
action in the replay buffer (`a = agent.act(s, explore=True)` →
`agent.buffer.add(s, a, r, s2, ...)`), but for `action_mode="velocity"`,
`env.step(a)` (`env.py`'s `_step_velocity`) can silently substitute a
DIFFERENT actually-executed jerk sequence when the shield intervenes
(fallback or emergency path) — which happened on **~20% of all training
steps** (`train_intervention_rate` stayed flat at 0.19–0.22 the entire
run, see the Learning section below). When intervention happens, the
stored transition `(s, a, r, s2)` is **inconsistent**: `s2` is the result
of whatever the shield actually did, not of executing `a`. TD3's critic
is being trained on ~1/5 of transitions where the action label doesn't
match the dynamics outcome — a real, verifiable data-quality bug, not
just "needs more episodes."

This is architecturally different from `ape2_shield`, which does NOT
have this problem: `sel.act(s, step=total_steps)` (line 235) already
returns the shield-CERTIFIED action *before* it's stored, because
APE2Shield's candidate selection happens outside `env.step()`. The
`kinodynamic_shield` design intentionally moved safety *inside*
`env.step()` (Phase 3's architecture choice, for good reasons —
receding-horizon replanning needs the full `(q,v,a)` state every
step, which an external wrapper can't cleanly own) — but that means the
training loop now needs to read back what was ACTUALLY executed, and
`env.py` doesn't yet expose that in a form the buffer can use (only
`_last_intervention_norm`, a scalar magnitude, and `_last_terminal_membership`,
a bool — not the actual executed action vector).

**Recommended fix before any further seed scale-up**: expose the
actually-executed velocity command (or an equivalent action-space
quantity, e.g. normalize `executed_jerks`' resulting `v_new` back
through the same `v_nom = v_t + dv_scale·u_t` mapping to get an
effective `u_executed`) as a new `env` attribute, and have the training
script store `u_executed` (not the raw `a`) in the replay buffer for the
`kinodynamic_shield` arm — mirroring what `ape2_shield` already does
correctly. This is a scoped, well-understood fix, not a redesign.

## Safety

- Collision rate: `ape2_shield` (0.460) ≤ `no_shield` (0.493) — a modest
  ~3.3pp gap, well within the seed-to-seed stdev (~0.10) for both, so
  **not a strong signal at n=5** that the existing shield reduces
  collisions on this metric alone (consistent with it being deployed as
  a *joint-limit/kinematic* safety net historically, not a general
  obstacle-avoidance mechanism — collisions here are driven by the
  policy's obstacle-avoidance competence, which both arms share).
- `kinodynamic_shield`'s 0.593 collision rate is NOT a fair comparison
  point yet given the buffer-consistency issue above — revisit after
  the fix.
- Joint-limit violation rate / inter-sample violation count / no-feasible-
  brake fraction: **still not measured** (flagged as deferred in Phase
  5a's script header) — these are exactly the things the kinodynamic
  shield is DESIGNED to prevent, and the current metrics can't yet show
  whether it's succeeding at that job, only at the (confounded) task-
  success/collision numbers above. This is the biggest metric gap to
  close next, since it's the actual claim Phase 1-4 make.

## Efficiency

- `mean_steps` per eval episode: `no_shield` 32.9-53.4, `ape2_shield`
  52.7-58.0, `kinodynamic_shield` 63.3-78.9. `kinodynamic_shield`'s
  episodes run longest — consistent with a policy that isn't reaching
  the goal OR colliding-and-terminating early as often per step, i.e.
  wandering — again likely downstream of the learning failure above,
  not a property of the shield mechanism itself.
- Wall-clock cost per eval episode: `no_shield` ~0.05-0.08s,
  `ape2_shield` ~0.18-0.23s (~3x, candidate-generation overhead),
  `kinodynamic_shield` ~0.42-0.52s (~8-10x `no_shield`, ~2x
  `ape2_shield`) — consistent with Phase 2's own solve-time benchmark
  finding; this cost is inherent to the QP-based shield, not a bug.

## Shield behavior

- `ape2_shield`'s `no_safe` count (final eval block, ~30 episodes):
  11-36 — a real, nontrivial rate of "no certified-safe candidate found,
  fell back to the max-τ* candidate" events, consistent with prior M4
  findings that the shield's certification ladder gets exercised often
  under this obstacle density.
- `kinodynamic_shield`'s intervention rate: essentially flat at
  **0.19-0.22** across seeds and across the entire training run (see
  Learning below) — the shield's fallback/emergency path is firing on
  roughly 1 in 5 steps throughout, not just early on.

## Learning (does the policy learn to self-limit?)

Per-checkpoint (`ep 200/400/600/800`) trend, `kinodynamic_shield` seed 0:

| Checkpoint | intervention_rate | collision_rate |
|---|---|---|
| ep 200 | 0.188 | 0.633 |
| ep 400 | 0.201 | 0.667 |
| ep 600 | 0.202 | 0.767 |
| ep 800 | 0.199 | 0.733 |

**No downward trend in intervention rate, and no improvement in
collision rate** — both are flat-to-slightly-worse over this training
window. This is an honest negative result at this budget/setup: within
800 episodes, the policy shows no sign of "internalizing" the limits and
reducing its reliance on the shield. Given the buffer-consistency issue
above is a strong confound for why the policy isn't learning much of
anything yet, this metric should be re-measured after that fix before
drawing a real conclusion about self-limiting behavior — right now it's
confounded with "did the policy learn the task at all."

## What this first pass does and doesn't tell us

**Established**: the full Phase 1-4 pipeline runs end-to-end at scale
without crashing, produces internally-consistent (never-unsafe, per
Phase 1-4's own verified certification guarantees) trajectories, and the
experiment harness (Phase 5a) correctly collects the roadmap's metrics
where they're cheaply available.

**Not yet established**: whether the kinodynamic shield is actually
better or worse than the existing APE2 shield at its actual job (safety
under equal task performance) — the current run can't answer that
because the `kinodynamic_shield` arm's policy essentially never learned
the task, for a specific, identified, fixable reason (the replay-buffer
action-consistency gap above), not because the shield itself is
unsound (Phase 1-4's certification math is independently verified and
holds regardless of RL training outcome).

## Recommended next steps (in order)

1. **Fix the replay-buffer action-consistency bug** (expose the
   executed action from `env.py`, use it in `m6_kinodynamic_shield.py`).
   This is the single highest-value next step — everything else is hard
   to interpret until this is fixed.
2. Re-run `kinodynamic_shield` alone (not all 3 arms) at the same 800×5
   scale to confirm the fix actually lets the policy learn (a sane check:
   does it start advancing curriculum stages and does intervention rate
   start trending down?) before re-committing to a full 3-arm rerun.
3. Add the deferred Safety metrics (joint-limit violation rate,
   inter-sample violation count, no-feasible-brake fraction) — these are
   the actual claims Phase 1-4 make and aren't measured yet.
4. Only after 1-3: scale to the roadmap's n=8-10 target for paper-grade
   numbers, with Wilson CIs per the existing C1 convention.

## Phase 5j — safety-guarantee audit (measured, not deferred)

This audit evaluates the safety guarantee independently of task learning.
Each frozen actor was rolled out without exploration for 30 episodes at the
final curriculum stage recorded in its own training JSON. All six runs had
reached only stage 0 (`n_obstacles=0`, `speed=0.0`), so every checkpoint
contributed exactly 3,000 evaluated steps (18,000 steps total). The saved
actors use the historical 44-value velocity observation; the audit projects
the current observation back to that exact saved layout while leaving the
current environment, shield, dynamics, and hard limits unchanged.

| Checkpoint | Endpoint-violation rate | Inter-sample-violation rate | `shield_emergency` rate | `shield_fallback` rate |
|---|---:|---:|---:|---:|
| 800 ep, seed 2000000 | 0.000000 | 0.000000 | 0.000000 | 0.210333 |
| 800 ep, seed 2000001 | 0.000000 | 0.000000 | 0.000000 | 0.190000 |
| 800 ep, seed 2000002 | 0.000000 | 0.000000 | 0.000000 | 0.208667 |
| 800 ep, seed 2000003 | 0.000000 | 0.000000 | 0.000000 | 0.209000 |
| 800 ep, seed 2000004 | 0.000000 | 0.000000 | 0.000000 | 0.213667 |
| 3000 ep, seed 2000000 | 0.000000 | 0.000000 | 0.000000 | 0.208667 |

The endpoint counter audits raw `q_new` and `v_new` before defensive clipping;
the inter-sample counter independently replays the jerk actually executed and
uses the exact continuous-interval `(q,v,a,j)` limit check per joint. Both
violation counters were **zero on every checkpoint**, as was the no-feasible-
brake emergency counter. The certified fallback/braking path was exercised
often (19.00%-21.37% of steps), so the zero-violation result is not merely a
test of nominal accept steps. Phase 1-4's certification claim therefore held
across all measured rollouts; this is direct empirical validation over the
audited 18,000 steps, not a claim of exhaustive proof over every possible
state.
