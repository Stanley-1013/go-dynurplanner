# Paper 1 Plan — working title candidates

1. **"Discrete Collision Checking Is Unsound for Dynamic Manipulator RL:
   Measurement, Certification, and a Continuous-Time Fix"**
2. "Exact Continuous-Time Collision Accounting for Dynamic-Obstacle
   Manipulator Learning" (softer)
3. "Mind the Gap Between Timesteps: Certified Interval Collision Reasoning
   for Manipulator RL in Dynamic Scenes"

Venue: RA-L (primary; 8p + refs, allows the measurement+method shape) →
IROS/ICRA option via RA-L dual track. CoRL if M5/keystone results merit
moving the grid contribution in (else grid goes to Paper 2).

## Thesis (one sentence)

In dynamic scenes, the per-step discrete collision check used by
essentially all manipulator RL pipelines is measurably unsound — worst in
the quasi-static regime everyone trusts — and the URPlanner parameterized
space admits an exact, closed-form continuous-time replacement that is
cheaper than subsampling and yields a certified action shield; reward
shaping with the same machinery, tested honestly, does NOT help — the
value is in measurement and certification, not shaping.

## Contributions (with the numbers we already hold)

- **C1 (measurement / Figure-1).** Exact interval first-contact audit of
  the standard protocol: 764 collisions, 3 seeds, Wilson CIs. Pooled miss
  rate 19.0% [16.4, 21.9]; S=16 subsampling still misses 0.8% while
  costing MORE than the exact check (1.92 vs 1.50 ms/step). Inverted
  speed trend: 45.6% missed at 0.1 m/s vs 13.9% at 2.0 m/s (arm-sweep
  grazing mechanism). Severity: missed contacts median 1.4 cm max-overlap
  but tail to 25.1 cm — missed ≠ mild.
- **C2 (theory).** Conservativeness lemma: chord bound
  ε_lin = ¼(Σᵢ√Rᵢ|δθᵢ|)² (cross-term-correct; the naive per-joint bound is
  disproven by counterexample) + obstacle acceleration + velocity-noise
  terms → certified no-tunneling interval check under bounded model error.
  Empirical property test: certified actions never collide (0 violations).
- **C3 (algorithm).** APE2-Shield: candidate-pool certification at
  near-zero extra cost (shares the root computation with the reward;
  broad-phase prefilter → 1.5 ms/step full audit). Honest fallback ladder
  with counted no-safe residual. H4 numbers: PENDING M4-v2 (v1 showed
  zero certified-action collisions; all residual collisions were provably
  cornered states).
- **C4 (negative result, honest).** D-UOAR-CT reward (objective correction
  to the true continuous-time collision cost): two rounds, mechanism
  diagnosed both times (unbounded TTC spike → bounded form), still trails
  plain UOAR (succ 0.47 vs 0.65, coll 0.40 vs 0.25). Conclusion: use the
  continuous-time machinery for evaluation and certification, not shaping.

## Figures / tables plan

| Item | Source | Status |
|---|---|---|
| Fig 1: miss rate vs speed, by subsampling depth, Wilson CIs (3 panels incl. thickness + wall-clock) | results/m1_hardened/figure1_hardened.png | done (regen at final a_r) |
| Fig 2: tunneling schematic (thin box crossing between endpoints) + τ* breakpoint method illustration | to draw (TikZ) | todo |
| Fig 3: curriculum training curves uoar vs ct (H3, negative) | m3c jsons | data done, plot todo |
| Fig 4: shield on/off — collision decomposition (certified=0 / cornered residual) | m4_uoar.json v2 | pending run |
| Tab 1: per-step cost: exact vs S∈{1..16} | m1h timing | done |
| Tab 2: severity stratification missed vs caught | m1h severity | done |

## Section outline

1. Intro — the trust gap: all dynamic-manipulator RL results rest on
   discrete collision accounting; we measure the gap, then close it.
2. Related work — discrete checking in RL envs; CCD in planning
   (cuRobo swept costs, NeuralSVCD detection-only); safety filters
   (PACS/RAIL reachability, iKinQP QP); URPlanner lineage; DRP as SOTA
   learned planner (CoRL'25) with no continuous-time treatment.
3. Preliminaries — URPlanner parameterized space (Eq.2-13 recap).
4. Continuous-time collision reasoning — τ-linearization, breakpoint
   enumeration exactness argument, swept integral, conservativeness lemma
   (proof in appendix), broad-phase prefilter.
5. Measurement study (C1) — protocol, results, severity; implications for
   benchmark practice.
6. Certified shield (C3) + honest reward negative (C4) — APE2-Shield,
   fallback ladder semantics, H4 results; D-UOAR-CT ablation as the
   "shaping doesn't help" evidence.
7. Discussion & limitations — a_r placeholder → final value from lab;
   AABB world model; constant-velocity interval model (covered by
   inflation); scripted-policy protocol study vs trained-policy audits;
   Panda-only.

## Gaps to close before submission

- [ ] M4-v2 (H4) numbers — running
- [ ] a_r from 師兄 → regenerate M1-hardened + severity (12 min)
- [ ] Fig 2 schematic + Fig 3 plots
- [ ] Optional strengthener: DRP checkpoint audit (M1b, lab machine)
- [ ] Multi-seed bump for M1 (currently 3 seeds — fine for RA-L; 5 nicer)
- [ ] φ_aux exact form from lab code (affects only reproducibility notes)
- [ ] Writing: /ars pipeline AFTER all numbers frozen

## Paper 2 (later): GO-DynURPlanner — grid observation + jointly-trained
occupancy-forecasting auxiliary (H2/M5, keystone), deployment-realistic
sensing, CoppeliaSim verification, HRC scenes. Needs M5 results + GPU-scale
forecaster + real-robot section. Target CoRL/ICRA.
