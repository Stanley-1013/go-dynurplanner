# Advisor Code Catalog — `advisor_code/0801pretrain/`

> Read-only cataloging pass, 2026-07-08. Source directory is gitignored lab
> code (TensorFlow 1.x) — nothing here was executed, only statically read.
> Cross-references: `code/ASSUMPTIONS.md`, `code/godynur/*.py`.

## Group 1 — E3AC.py, E3AC_dual_memory.py, E3AC_dual_memory_Diffusion.py

All three files are near-identical copies of one class (`class E3AC(object)`,
constructor `__init__(self, state_dimension, action_dimension, action_bound)`)
authored by Ying Fengkang (NUS). Network: 1 actor + **5 critics** (hardcoded,
not a generic `K`), each critic loss = `Eta_1 * td_error_avg + Eta_2 *
td_critic_i + Omiga * (Q_i - Q_avg)^2`. Hyperparameters identical across all
three files: `Actor_LR=0.001`, `Critic_LR=0.001`, `Gamma=0.98`, `Tau=0.01`,
`Memory_Size=60000`, `Batch_Size=64`, `Eta_1=0.4`, `Eta_2=0.6`, `Omiga=0.1`.
Actor/critic MLPs are 256→256 dense, ReLU, tanh output on actor.

### E3AC.py
Baseline E3AC: single `self.memory` replay buffer only, no expert data.
`extensive_exploration_strategy(self, state, number_of_exploration)` generates
candidates via **3 noise types x `number_of_exploration` repeats + 1 raw
action**: Gaussian(σ=0.1), Gaussian(σ=0.05 — but the inline comment on that
line still says "variance 0.5", a stale comment/code mismatch), and OU noise
(via `OUNoise`). `evaluate_and_choose_optimal_action(self, state,
action_candidates)` hardcodes `range(10)` and a 10-slot list — i.e. it
silently assumes `number_of_exploration=3` (1 + 3×3 = 10); calling with a
different `number_of_exploration` would either crash or silently ignore extra
candidates. Selection is a flat **argmax over `q_average`** (mean of the 5
critics) — no eta-weighting, no rollout term, at this call site. `save()`
hardcodes path `'model/E3AC_SO/params'` with a comment `# E3AC_foresee0805
# E3AC` suggesting the string was hand-edited per experiment run and not
restored afterward. `__main__` test block uses `state_dim=21`, `action_dim=7`
— stale/inconsistent with the real training state size (see below).

### E3AC_dual_memory.py
Identical to `E3AC.py` plus: `self.expert_memory = np.load('RL_expert_memory.npy')`
(verified via numpy header read: **shape (60000, 48), float32** — 48 =
`2*state_dim + action_dim + 1` solves to **state_dim=20**, confirming the
already-recorded 20-D state independently). `variance_2` corrected to `0.5`
(matches its own comment, unlike plain `E3AC.py`). `train()` adds a growing
expert-sampling schedule: `self.count` starts at 0, `+=1` every 2000 env
steps, capped at 20; batch = `self.count` expert rows (`np.random.choice(
Memory_Size, size=self.count)`) + `Batch_Size - self.count` interaction rows.
Since `Memory_Size` (60000) == expert_memory row count, indexing is exact/safe
here. Expert fraction grows from 0% to a ceiling of 20/64 ≈ 31%. Save/restore
path: `'model/E3AC_foresee_dual_RL0805/params'` (both match, and match a real
entry in `model/`).

**This confirms the ED2 DC-mechanism hypothesis**: two separate memories
(interaction `self.memory` + expert `self.expert_memory`), with a step-counter
schedule that changes the sampling ratio between them over training — this
*is* the paper's data-compensation (DC) design, implemented as a simple linear
step-count schedule rather than anything more elaborate.

### E3AC_dual_memory_Diffusion.py
Line-for-line identical to `E3AC_dual_memory.py` except: (1) loads
`diffusion_expert_memory.npy` instead of `RL_expert_memory.npy`; (2) schedule
cap is 22 instead of 20 (same +1 every 2000 steps; a commented-out alternate
schedule — every 5000 steps, cap 32 — is present but disabled); (3) save path
`'model/E3AC_dual_Diffusion0805/params'` but **restore path is a different
directory**, `'model/E3AC_foresee_dual_Diffusion0805/params'` — a real
save/restore path mismatch within the same file (see Recommended next
actions).

**Diffusion header check** (numpy header read only, no data loaded into a
model): `diffusion_expert_memory.npy` shape is **(63200, 48)**, i.e. more rows
than `Memory_Size=60000`. The code still does `np.random.choice(Memory_Size,
size=self.count)` to index into `self.expert_memory` — so it only ever
samples from the **first 60000 of 63200 rows**, silently ignoring the last
~5% of the file. Not a crash, but a real (minor) bug.

**Reconciliation note:** the "Diffusion" variant does **not** run any
diffusion model at train time — no noise schedule, no denoising loop, no
U-net/MLP diffusion network anywhere in this file or its imports. It is
byte-for-byte the same dual-memory E3AC, just pointed at a different
pre-generated `.npy` expert-trajectory file (presumably produced by an
offline diffusion generator elsewhere, not included in this handoff). If the
paper's ED2 is meant to be "diffusion model generates expert data online
during training," **this code does not implement that** — it consumes a
static file. This matters directly for reconciling `godynur`'s ED2/APE2
assumptions: don't expect to find an online diffusion generator to port: the
lab's actual deliverable for that piece is just the `.npy` file (not present/
readable as source, and out of scope per instructions — only the shape was
inspected).

## Group 2 — rl.py, rl_dual_memory.py, rl_dual_memory_Diffusion.py, rl_dual_memory_Diffusion0718.py, rl_memoryZone2_2step3.py

All five implement plain single-critic, single-actor **DDPG** (`class
DDPG(object)`, `__init__(self, a_dim, s_dim, a_bound)`), 256→256 dense
actor/critic, same hyperparameters throughout: `LR_A=0.001`, `LR_C=0.001`,
`GAMMA=0.98`, `TAU=0.01`, `MEMORY_CAPACITY=60000`, `BATCH_SIZE=64`. None of
these files have `extensive_exploration_strategy` / multi-critic ensembles —
that machinery is E3AC-only. `choose_action` returns one raw actor output;
exploration noise (when used) is added by the *caller* (see Group 4 mains).

### rl.py
Baseline DDPG, single `self.memory` buffer, no expert data (one commented-out
line shows a`self.memory = np.load('diffusion_expert_memory.npy')` was tried
and abandoned here). Save path `'model/DDPG/params'`.

### rl_dual_memory.py
Adds `self.expert_memory = np.load('RL_expert_memory.npy')` and a growing
expert-fraction schedule: `self.count` +=1 every 3000 steps, capped at 10;
`indices1` (expert) = `size=self.count`, `indices2` (interaction) =
`size=BATCH_SIZE-self.count` — same "grow expert fraction from 0" direction
as `E3AC_dual_memory.py`, smaller ceiling (10/64 ≈ 16%). Save path
`'model/DDPG_dual_RL0805/params'` (matches a real `model/` entry). A larger
alternate schedule (every 5000 steps, cap 64) is present but commented out.

### rl_dual_memory_Diffusion.py
Loads `diffusion_expert_memory.npy`. **Schedule direction is inverted** versus
`rl_dual_memory.py`: `self.count` starts at **21** (not 0), grows every 5000
steps to cap 64; but here `indices1` (expert) = `size=BATCH_SIZE-self.count`
and `indices2` (interaction) = `size=self.count` — i.e. `self.count` now
tracks the *interaction* fraction, which grows, while the *expert* fraction
**shrinks** from 43/64 (67%, expert-heavy) at start down to 0/64 by the time
`count` saturates at 64. This is the opposite curriculum from
`rl_dual_memory.py` and both E3AC dual-memory files (which start at 0%
expert and slowly ramp up to a low ceiling). Save path
`'model/DDPG_dual_Diffusion/params'` — no exact match in the `model/` listing
(closest is `DDPG_dual_Diffusion0718`), suggesting this particular
save-path variant was superseded before a final run completed under this name.

### rl_dual_memory_Diffusion0718.py
Also loads `diffusion_expert_memory.npy`, but reverts to the **non-inverted**
schedule direction: `self.count` starts at 12, +=1 every 2000 steps, capped at
32; `indices1` (expert) = `size=self.count` (grows), `indices2` (interaction)
= `size=BATCH_SIZE-self.count` (shrinks) — same direction/labeling as
`rl_dual_memory.py`, just starting from a nonzero floor (12/64 ≈ 19% up to
32/64 = 50%). Save path `'model/DDPG_dual_Diffusion0718/params'` (matches a
real `model/` entry).

**Reconciliation note:** there are genuinely **two different, contradictory
DC-schedule directions** in this codebase for the DDPG family (grow-expert-
from-0 vs. start-expert-heavy-and-decay-to-0), and the *later*-dated file
(`rl_dual_memory_Diffusion.py`, Aug 1 2024) uses the decay-to-0 direction
while the *earlier* one (`..._Diffusion0718.py`, named for Jul 18 but
modified Aug 2 2024) and the plain `rl_dual_memory.py` use the grow-from-0
direction. If reconciling `godynur`'s DC-mechanism design against "the lab's
real schedule," there isn't a single canonical answer — pick whichever
direction matches the paper's prose description of ED2/DC (typically:
start demonstration-heavy, anneal toward self-collected data — which matches
the *decay-to-0* file, i.e. the more mature/later design).

### rl_memoryZone2_2step3.py
**This is not a dynamic-obstacle or two-step-lookahead file at all.** It
implements a second DDPG network under TF scopes `'Actor2'`/`'Critic2'` with a
**reward-sorted, two-zone prioritized replay scheme**:
`memory2 = self.memory` re-ordered by `np.argsort(self.memory[:, 0])`
(column 0 holds the reward — `store_transition` here uses layout
`hstack(([r], s, a, s_))`, reward *first*, unlike every other file's
`hstack((s, a, [r], s_))`). After sorting ascending by reward, indices
`20000..59999` ("Zone A", the higher-reward 2/3 of the buffer) and
`0..19999` ("Zone B", the lower-reward 1/3) are sampled in a ratio that
shifts over training: every 6000 "train steps" (itself only incremented once
`self.pointer > MEMORY_CAPACITY`, i.e. after the buffer is already full),
`self.count` +=1 up to a cap of 21; batch = `64-self.count` from Zone A +
`self.count` from Zone B. There is **no obstacle velocity, no prediction
horizon, no moving-obstacle logic anywhere in this file** — "2step" almost
certainly refers to a two-stage/two-phase *training procedure* (see
`main_DDPG_TMECH.py` below, which pretrains/fills the buffer before calling
`train()`), not to Hart/Waltz/Okhrin-style dynamic obstacle avoidance
(arXiv:2311.16841). Save/restore path is `'model/DDPG/params'` — **identical
to plain `rl.py`'s path**, meaning if this variant were ever trained after
`rl.py`, it would silently overwrite that checkpoint (footgun, though this
looks like an experimental/scratch file rather than a finished deliverable —
no `model/` subdirectory corresponds to "memoryZone2" by name).

**Reconciliation note:** confirms the task's Option B — "2step" = training-
procedure staging, not dynamic-obstacle two-step prediction. Nothing in this
whole codebase (across all 5 `rl*.py`/3 `E3AC*.py` files) references obstacle
velocity or motion prediction; `Franka_Env_Scene2.py`/`..._trajectory_verification.py`
both use fixed AABBs. This is consistent with ASSUMPTIONS.md's existing note
that the handoff is "their STATIC baseline reproduction of URPlanner, no
dynamic obstacles at all" — the dynamic-obstacle extension is squarely the
user's own contribution in `godynur`, not something to backfill from this
codebase.

## Group 3 — Franka_Interface.py, Franka_Interface_DDPG.py, Franka_Interface_trajectory_verification.py, Franka_Env_Scene2_trajectory_verification.py

All of these are **CoppeliaSim virtual-twin verification scripts**, not real-
robot deployment code — they generate a joint-angle trajectory purely from the
math env, then replay it into CoppeliaSim for visualization/proximity
sensing. All three `Franka_Interface*.py` files `import sim` (confirmed via
read; grep also shows no `import simConst` anywhere in these files) and share
the identical connection boilerplate:
```python
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
```
i.e. **localhost, port 19999** — matches the classic V-REP/CoppeliaSim legacy
remote API default port, confirming ASSUMPTIONS.md item 5/9's "CoppeliaSim
confirmed" note with a concrete number. Joint control uses
`sim.simxSetJointPosition` per joint (7 joints, `Franka_joint1..7`), and a
CoppeliaSim `Distance` object handle is read each replay step to track
minimum clearance — this is the demo/video-generation pipeline, not the
training loop.

### Franka_Interface.py
Loads `Franka_Env_Scene2.ArmEnv` + `E3AC_dual_memory.E3AC`, calls `rl.restore()`,
then drives 300 steps using `extensive_exploration_strategy` +
`evaluate_and_choose_optimal_action` (the plain argmax-over-Q version, not the
hybrid one — see Group 4). `env.step()` here is confirmed (again) to return
the 5-tuple `(next_state, reward, done, pose_error, orient_error)`.

### Franka_Interface_DDPG.py
Same shape but loads `rl_dual_memory_Diffusion0718.DDPG`, and action selection
is just `choose_action(state)` + `np.clip(np.random.normal(action, 0.1),
*action_bound)` — no candidate-list/exploration-strategy machinery (DDPG
files never define one).

### Franka_Interface_trajectory_verification.py
Loads `Franka_Env_Scene2_trajectory_verification.ArmEnv` + plain `E3AC.E3AC`.
**This file contains the actual `explore_and_foresee_value(state, raw_action,
horizon, evaluation_criterion)` function matching the paper's eta-annealed
hybrid evaluation** — but note it is only *defined* here and called from a
commented-out block; the live `generate_trajectory()` loop instead calls the
simpler `rl.evaluate_and_choose_optimal_action`. Its `evaluation_criterion in
{'reward','q_value','hybrid'}`; for `'hybrid'`:
```python
eta_1 = np.clip(rl.counter / 200000, *[0, 1])
total_value = foresee_action_value[i] * (1 - eta_1) + q_value[i] * eta_1
```
i.e. eta anneals linearly from 0 (pure one-step-lookahead reward) to 1 (pure
Q-value) over the first 200,000 training steps — **this exactly matches
ASSUMPTIONS.md item 8's `T=2e5` hyperparameter**, and is a direct, concrete
match to the paper's described eta-annealed V_HPE hybrid (Eq 14–18): `horizon`
is a generic parameter but is only ever invoked with `horizon=1` anywhere in
this codebase (see Group 4) — i.e. the "foresee" mechanism is a **one-step
lookahead of the agent's own candidate actions' immediate reward** (simulated
by literally calling `env.step(a)` once per candidate then restoring
`env.arm_info`/`on_goal`/`distance_old`/`orient_old` back to the pre-lookahead
snapshot — cheap because the env is a closed-form math simulator, not
CoppeliaSim), **not** a forecast of future obstacle motion. There is no
obstacle-prediction semantics anywhere in "foresee."

This file's env (`Franka_Env_Scene2_trajectory_verification`) has an
**extended, non-uniform interface**: `reset()`/`reset_test_env()` return a
6-tuple `(s, dist1, dist2, delta_pos, delta_orient, finger_xyz)`, and `step()`
returns an **8-tuple** `(next_state, reward, done, dist1, dist2, delta_pos,
delta_orient, finger_xyz)` — richer than the training env's 5-tuple. This is
not a contradiction of the already-recorded 5-tuple finding (that was for the
*training* env class); it's a different, verification-only class with extra
debug readouts. Concrete bug found: this same file's own `if __name__ ==
'__main__':` block at the bottom still does `next_state, reward, done,
pose_error, orient_error = env.step(action)` (5-value unpack) against a
`step()` that returns 8 values in this same file — running this file's
`__main__` block directly would raise a "too many values to unpack" error.
Also writes `np.save('Scene5_APE2_DC-ED2_3.npy', joint_state_cache)` — the
literal filenames `Scene2_APE2_DC-ED2_1..5.npy` / `Scene5_APE2_DC-ED2_2.npy`
present in the directory are recordings from this exact script, and their
names show the advisor's own team uses "APE2" and "DC-ED2" as their working
names for these mechanisms — strong direct terminological confirmation that
this codebase self-identifies as implementing the paper's APE2/DC-ED2
concepts (not just superficially similar independent inventions).

### Franka_Env_Scene2_trajectory_verification.py obstacle/scene comparison
Grepped `Franka_Env_Scene2.py` for the same box coordinates: **`box1`–`box4`
AABB corner coordinates are byte-identical** between the training env and
this verification env (same 4 boxes, same corners, same `total_link_length =
0.9101`, same 3-segment collision check `link2/3/4`). The only material
difference is the goal-dwell threshold: training env uses `on_goal >= 50`
(matches ASSUMPTIONS.md's recorded value), this verification env uses
`on_goal >= 5` with a comment `# 60` suggesting yet another value was tried.
**Reconciliation note:** `goal_dwell=50` remains correct for training per
ASSUMPTIONS.md; the verification script intentionally uses a much shorter
dwell (5, not 50) purely to make demo trajectory generation faster — this is
not a contradiction, just confirms dwell is hand-tuned per script rather than
a single shared constant.

## Group 4 — main_DDPG_TMECH.py, main_DDPG_pretrain.py, main_E3AC_foresee.py

All three share the same skeleton (`MAX_EPISODES=1000`,
`MAX_EPISODES_STEPS=300`, `np.random.seed(1)`, `tf.set_random_seed(2)`,
`Franka_Env_Scene2.ArmEnv`), differing only in which agent class is imported
and how the action is produced each step.

### main_DDPG_pretrain.py
Imports `rl_dual_memory.DDPG`. Action = `choose_action(state)` + Gaussian
noise (σ=0.1), no candidate-list logic. Defines a `foresee_more(s)` function
(7 hand-enumerated candidates: raw + 4 Gaussian-noised + 2 OU-noised,
evaluated by literally rolling out one env step each and picking max reward)
but **it is defined and never called** in `train()`/`eval()` — dead code
left over from an earlier experiment.

### main_DDPG_TMECH.py
Imports `rl_memoryZone2_2step3.DDPG` (the reward-sorted double-zone replay
variant, see Group 2) — **not** just a venue-label duplicate of
`main_DDPG_pretrain.py`. Adds a `fill_up_memory()` phase that runs full
episodes purely to fill the replay buffer to `MEMORY_CAPACITY` *before*
calling `train()` — necessary because `rl_memoryZone2_2step3.DDPG.learn()`
sorts the *entire* buffer by reward every call and its "train_step" counter
only increments once `self.pointer > MEMORY_CAPACITY`. This is the concrete
evidence that "2step" (Group 2) is a two-phase training procedure
(fill-then-train), reinforcing that reading.

### main_E3AC_foresee.py
Imports plain `E3AC.E3AC` (not dual_memory). **This is the file that actually
invokes the eta-annealed hybrid evaluation live during training**:
```python
action_candidates = rl.extensive_exploration_strategy(state, 3)
optimal_action = explore_and_foresee_value(state=state, raw_action=action_candidates,
                                            horizon=1, evaluation_criterion='hybrid')
```
with the same `explore_and_foresee_value`/`eta_1 = counter/200000` logic
described in Group 3 (defined locally in this file too — copy-pasted, not
imported/shared). Confirms `main_E3AC_foresee.py`'s "foresee" = one-step
lookahead reward rollout blended with critic Q-value via the annealed eta,
which *is* actively used for action selection during training in this
specific entry point (unlike the trajectory-verification file, where the
same function exists but is dormant).

**Reconciliation note (important):** `E3AC.py`'s `save()` hardcodes
`'model/E3AC_SO/params'`, but the `model/` directory that would correspond
to running `main_E3AC_foresee.py` is named `E3AC_foresee0805` in the real
checkpoint listing — the two don't match. This, plus the save/restore path
mismatch already found in `E3AC_dual_memory_Diffusion.py` (Group 1), plus
`rl_memoryZone2_2step3.py` sharing `rl.py`'s save path (Group 2), together
show a consistent pattern: **save/restore paths are hand-edited string
literals inside each class file, toggled per experiment run, and not kept in
sync.** Treat any specific `model/<name>/params` -> `<script>.py` mapping in
this handoff as unreliable without independently checking which script's
`save()` call actually points there at the time of that run.

## Group 5 — seed_test.py, ou_noise.py

### ou_noise.py
`class OUNoise: __init__(self, action_dimension, mu=0, theta=0.05, sigma=0.1)`,
`reset()`, `noise()` — standard Ornstein-Uhlenbeck process:
`dx = theta*(mu - x) + sigma*np.random.randn(len(x)); x += dx`. Confirms the
standard DDPG/E3AC OU exploration noise assumption exactly, with `theta=0.05`
and `sigma=0.1` as the concrete defaults actually used (all call sites in
Groups 1/4 instantiate `OUNoise(action_dim)` with no override, so these
defaults are the real values in play).

### seed_test.py
One-line summary: it's a 6-line scratch script that seeds numpy (`np.random.seed(1)`)
and prints one `np.random.uniform(1, 0.5, [2,2])` call — a throwaway sanity
check of numpy's RNG/seeding behavior, unrelated to the RL pipeline itself.

## Group 6 — model/ subdirectory and rotationconverter dirs

`model/` contains only TensorFlow checkpoint files (no source), under these
experiment-named subdirectories (verified via `ls`): `DDPG_dual_Diffusion0718`,
`DDPG_dual_RL`, `DDPG_dual_RL0805`, `E3AC`, `E3AC_dual_Diffusion`,
`E3AC_dual_Diffusion0805`, `E3AC_dual_RL`, `E3AC_dual_RL0805`,
`E3AC_foresee0805`, `E3AC_foresee_dual_Diffusion0805`,
`E3AC_foresee_dual_RL0805`. This list corroborates which script variants were
actually run to completion (used in Groups 1/2/4 above to cross-check
save/restore path strings) — notably there is **no** `memoryZone2`-named
checkpoint, consistent with that variant reading as experimental/unfinished.

`rotationconverter/` and `rotationconverter-master/` both contain a copy of
`rotation_converter.py` (plus a trivial `setup.py`/`__init__.py` wrapper);
these were already confirmed byte-identical to the top-level
`rotation_converter.py` via diff in a prior pass — no further action needed,
noting only for completeness of this catalog.

## Recommended next actions

- **`E3AC_dual_memory_Diffusion.py`** (Group 1): the lead should personally
  confirm the save-path (`E3AC_dual_Diffusion0805`) vs. restore-path
  (`E3AC_foresee_dual_Diffusion0805`) mismatch before trusting any claim about
  which `model/` checkpoint corresponds to this script's output.
- **`main_E3AC_foresee.py`** (Group 4): this is the single most
  paper-relevant file in the handoff — it's the only entry point where the
  eta-annealed hybrid (APE2-style) evaluation is *actually exercised* during
  training rather than merely defined. If the user's `godynur/ape2.py` is
  meant to reproduce APE2 exactly, this file (plus `E3AC.py`'s
  `extensive_exploration_strategy`) is worth the lead's own direct read
  rather than trusting this summary's paraphrase of the eta/annealing math.
- **`rl_dual_memory_Diffusion.py` vs. `rl_dual_memory_Diffusion0718.py`**
  (Group 2): the two files implement genuinely opposite DC-schedule
  directions (decay-to-zero vs. grow-from-zero expert fraction). Whichever
  direction the lead intends `godynur`'s DC-mechanism to reproduce depends on
  which of these two the advisor considers "the real one" — worth asking
  directly rather than inferring from file-modification dates alone.
