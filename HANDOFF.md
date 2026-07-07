# Handoff note — 2026-07-08, Fable 5 → Sonnet 5

Fable 5's session budget ran out. This note is for whichever session
picks this up next (Sonnet by default; escalate to Opus only at the
specific trigger points listed below — not by default "to be safe").

## Where things actually stand (read this before re-deriving anything)

- **Paper 1 has its four contributions with real numbers.** See
  `paper/PAPER1_PLAN.md` — C1 (measurement, 764 collisions, Wilson CIs),
  C2 (conservativeness lemma, corrected once by a failing test), C3
  (APE2-Shield, succ 0.675/coll 0.24 vs 0.175/0.585 no-shield), C4 (honest
  negative: reward shaping with the same CT machinery does NOT help).
  Don't re-run these from scratch — they're done. What's NOT done: freezing
  a_r at the lab's real value, and the H2 (grid+forecast keystone)
  question, which is currently confounded (see below).
- **H2 (does the grid+forecast keystone help) is UNRESOLVED, not negative.**
  Loop 9's H2 second read found that arms sharing a seed VALUE (not just
  seed offset) produced near-identical failure patterns regardless of the
  aux/grid_mode setting being compared — a real experimental-design bug
  (arm identity was confounded with seed identity), not evidence against
  the keystone. Already fixed (`m5_grid.py --seed-salt`) and relaunched
  with decoupled seeds (`experiments/results/m5_grid_deconf/`) — check
  that job's output before concluding anything about H2.
- **The advisor's real code arrived** (`0801pretrain.zip` →
  `advisor_code/`, gitignored — it's his unpublished code, never commit
  it). Reconciliation against it is IN PROGRESS: `code/ASSUMPTIONS.md`
  §0 has the four verified/wrong findings (FK exact match, geometry
  algorithm identical, but our env's action scale/state dim/reward
  recipe/goal_dwell all differ from his real code). A cataloging subagent
  is/was reading the remaining ~20 files (E3AC variants, dual-memory/ED2,
  CoppeliaSim interface) — check `code/advisor_code_catalog.md` for its
  report before re-reading those files yourself.
- **Deliberate choice made**: the env/reward/action corrections found via
  reconciliation were NOT retroactively applied to existing experiments.
  All of Loop 1-9's results (M1/M3/M4/M5) used our own env defaults
  consistently across every A/B comparison in a given experiment, so the
  *relative* conclusions (shield helps, CT reward doesn't, discrete
  checking is unsound) are NOT invalidated by the discrepancy — only
  *absolute* numbers would differ if we matched the advisor's exact
  recipe. Don't panic and re-run everything; do add an opt-in
  "advisor_faithful=True" mode to `DynArmEnv` before any final
  paper-numbers freeze, and use it for a baseline-reproduction check.

## Specific opus-escalation triggers for THIS project (not generic)

Per `~/.claude/rules/dispatch.md` §5, escalate to opus when stuck on the
same subtask twice, or for genuine judgment calls between defensible
options. Concretely, for this project that means:

1. **If H2 (grid+forecast) still shows high seed variance after the
   deconfounded rerun**, deciding whether to (a) scale up seeds/episodes
   further, (b) redesign the auxiliary task, or (c) drop the keystone and
   ship Paper 1 without it — that three-way fork is a judgment call worth
   an opus second opinion, not a sonnet guess.
2. **When writing Paper 1's actual prose** (not just holding the numbers)
   — framing the honest negative result (C4) persuasively without
   overclaiming is a taste/judgment task, not mechanical. Use
   `/ars-outline` or `/ars-full` at that point, and consider opus for the
   discussion/limitations section specifically.
3. **If the advisor's real E3AC/APE2 implementation (once cataloged)
   turns out to materially conflict with our APE2Shield's candidate
   generation or hybrid-evaluation formula** (not just hyperparameters —
   an actual mechanism difference), deciding whether to rewrite ours to
   match his or keep ours as a documented variant is architecture-level
   and worth escalating.
4. Do NOT escalate for: rerunning experiments, fixing env bugs, writing
   log pages, git hygiene, or reading more advisor files — all sonnet-work.

## Process notes (learned the hard way this session, don't repeat)

- **gitignore a new venv BEFORE creating it**, not after — a timed-out
  `git add -A` once packed 17k venv files into two commits; fixing it
  cost a soft-reset + reflog expire + gc cycle. `.venv*/` is now in
  `.gitignore`; if you make another venv, confirm the pattern still
  matches before your first `git add`.
- **Someone else's unpublished code (advisor, collaborators) never enters
  git**, even privately — `advisor_code/` and `0801pretrain.zip` are
  gitignored; keep it that way for any future code drops from the lab.
- **Long background training runs**: always redirect to a log file under
  `experiments/logs/`, never to `/dev/null` — needed a foreground rerun
  once just to see a crash that `/dev/null` had swallowed.
- **Delegate cataloging/reading-many-files tasks to subagents**; do
  judgment-heavy reconciliation (deciding what a discrepancy MEANS for
  our design) yourself — that split held up well this session and is
  worth keeping.
- Repo: https://github.com/Stanley-1013/go-dynurplanner (private).
  `research-log/01..09_*.html` + `index.html` is the full narrative;
  `code/ASSUMPTIONS.md` is the living reconciliation ledger;
  `paper/PAPER1_PLAN.md` + `paper/references.md` are the writing-stage
  materials. Commit and push often — the user has asked for this
  explicitly twice.
