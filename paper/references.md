# References — Verified Citation Ledger

Consolidated from research loops 2–7 (`research-log/02_deep_research_expansion.html`
through `research-log/07_execution_m0_m1.html`). Every entry below carries the
verification status assigned by the loop that actually checked it — this file
does not re-judge any claim, it only merges and deduplicates.

## Status definitions

- **VERIFIED** — multi-source confirmed (e.g. paper existence, authors, and
  the core numeric claim all corroborated across at least two independent
  checks/loops).
- **CORRECTED** — an earlier-reported number was wrong and has since been
  fixed; the note states what changed.
- **ABSTRACT-ONLY** — the claim rests on abstract-level or single-source
  checking (including cases where a specific number came only from a
  paywalled full-text/HTML source that could not be cross-confirmed). Treat
  as provisional.
- **UNVERIFIED** — single citation, never run through a dedicated
  verification pass in any loop. Do not cite without re-checking.

Status-mapping note: the source logs use their own Chinese-language tags
(已驗證/已更正/全文限定/未能獨立驗證 in Loop 3's ledger, 摘要層級 in Loop 4).
Mapping used here: 已驗證→VERIFIED, 已更正→CORRECTED, 未能獨立驗證→UNVERIFIED,
and both 全文限定 and 摘要層級→ABSTRACT-ONLY (both mean "the precise number
needs re-checking before it goes in the paper," even though one is
technically full-text-sourced and the other abstract-sourced). Where a loop
tagged only the *paper's existence* as verified but left a *specific number*
full-text-only/paywalled, the entry is kept VERIFIED with an inline caveat
naming the unconfirmed figure, rather than downgraded wholesale — downgrading
the whole entry would lose the (verified) fact that the paper is real.
Where two loops disagreed on the same status (rather than one loop simply
extending an earlier check), the stricter (more cautious) status is kept.

---

## (a) Base papers

- Ying et al. (2025). URPlanner: A Universal Paradigm for Collision-Free
  Robotic Motion Planning Based on DRL. arXiv:2505.20175. **[VERIFIED]** —
  the foundation paper; state/reward/collision definitions (Eq.6–10) are the
  base everything else extends.
- Xiao, Xie, Zhang et al. (2026). ENPIRE: Agentic Robot Policy
  Self-Improvement in the Real World. arXiv:2606.19980 (NVIDIA/CMU/UCB).
  **[VERIFIED]** — real, 99% success rate confirmed; explicitly demoted by
  Loop 3's direction verdict to "inspiration, not foundation" (optional
  future-work chapter, not load-bearing).
- Morvan, "train-robot-arm-from-scratch" (GitHub). **[listed-but-not-found
  (check with the lead)]** — not located in any of the five source files
  read for this consolidation.

## (b) Reward shaping & RL theory

- Ng, Harada & Russell (1999). Policy Invariance Under Reward
  Transformations. ICML 1999. **[VERIFIED]** — textbook-level PBRS
  policy-invariance result; grounds every reward-shaping term used.
- Devlin & Kudenko (2012). Dynamic Potential-Based Reward Shaping. AAMAS
  2012. **[VERIFIED]** — proves policy invariance still holds for a
  time-varying potential Φ(s,t); underlies the PBRS trend term.
- Chen et al. (2023). Mitigating Imminent Collision: A TTC-force Reward
  Shaping Approach. AAMAS 2023. **[ABSTRACT-ONLY]** — venue and the
  TTC = d/v_rel concept are multi-source confirmed; the exact formula could
  not be extracted from the paywalled PDF (binary, not text-extractable).
- Wen/Han et al. (2022). RVO-Shaped Rewards for Multi-Robot Navigation.
  arXiv:2203.10229. **[VERIFIED]** — RVO-area + expected-collision-time
  reward design; relied on for the velocity-obstacle reward comparison.
- Xie & Dames (2023). DRL-VO: Navigating Crowded Scenes Using Velocity
  Obstacles. IEEE T-RO, arXiv:2301.06512. **[VERIFIED]** — VO-based active
  avoidance signal, cited as a design comparison point.
- Hart, Waltz & Okhrin (2021). The Impact of Missing Velocity Information in
  Dynamic Obstacle Avoidance based on DRL. arXiv:2112.12465. **[VERIFIED]**
  — core theoretical justification that position-only observation is
  non-Markovian in dynamic settings (frame-stacking/LSTM cannot fully
  substitute for missing velocity).
- Hart, Waltz & Okhrin (2023). Two-Step Dynamic Obstacle Avoidance.
  arXiv:2311.16841. **[VERIFIED]** — same author group; "reward doubling ⇒
  collision halving" finding, used for reward-magnitude design.

## (c) Safety / shields

- Yang, Werner, de Sa & Ames (2025). CBF-RL. arXiv:2510.14959.
  **[VERIFIED]** — Ames group; CBF internalized during training, validated
  on Unitree G1. Caveat: the exact formula previously cited could not be
  confirmed beyond the paywalled full text — do not quote the formula
  without re-checking.
- Long et al. (2025). Neural Configuration-Space Barriers (DR-CBF).
  arXiv:2503.04929. **[VERIFIED]** — xArm6 real robot, cluttered + dynamic
  scenes. Caveat: the specific "100% dynamic" figure is sourced from
  paywalled full text only — recheck before quoting.
- Spalter, Roberts & Hiatt (2024). iKinQP-RL: Towards Online Safety
  Corrections for Robotic Manipulation Policies. arXiv:2409.08233.
  **[VERIFIED]** — "eliminates collision with new obstacles" claim
  confirmed; the specific 0% figure is full-text-sourced only, recheck
  before quoting.
- PACS: From Demonstrations to Safe Deployment, Path-Consistent Safety
  Filtering for Diffusion Policies. arXiv:2511.06385. **[VERIFIED]** —
  full-text verified in Loop 5: 0.20 ms safety step (vs 0.64 ms for CBF),
  ~5 ms trajectory recompute, 1 kHz safety loop, Franka FR3 real robot.
  (Loop 4 had only abstract-level checked this; Loop 5's full-text check
  supersedes and confirms it.)
- RAIL: Reachability-Aided Imitation Learning for Safe Policy Execution.
  arXiv:2409.19190. **[ABSTRACT-ONLY]** — Loop 4 abstract-level check only;
  cited for "pure diffusion policy fails, RAIL succeeds" on Franka Panda;
  never re-verified via full text.
- Latent Safety Filters: Generalizing Safety Beyond Collision-Avoidance via
  Latent-Space Reachability Analysis. arXiv:2502.00935. **[ABSTRACT-ONLY]**
  — Loop 4 abstract-level check only.
- SafeDojo: Safe Reinforcement Learning for VLA via Interactive World
  Model. arXiv:2606.20698. **[ABSTRACT-ONLY]** — Loop 4 abstract-level
  check only; cited as evidence that "safe RL for VLA" is a publishable,
  real-robot-validated (Franka, five tasks) topic.
- Thumm & Althoff (2022). Provably Safe DRL for Robotic Manipulation in
  Human Environments. ICRA 2022, arXiv:2205.06311. **[UNVERIFIED]** — cited
  once in Loop 2 without an individual verification tag; not present in
  Loop 3's ledger or any later loop.
- SafeDiffuser: Safe Planning with Diffusion Probabilistic Models.
  arXiv:2306.00148. **[UNVERIFIED]** — cited once in Loop 2 as a
  speculative combination opportunity (with URPlanner's ED2 module), never
  independently verified.
- sRCBF: Robust CBF for High Relative Degree Systems / Moving Obstacles.
  arXiv:2412.03678. **[UNVERIFIED]** — cited once in Loop 2, never
  re-checked.
- Achiam et al. (2017). Constrained Policy Optimization (CPO). ICML 2017.
  **[UNVERIFIED]** — cited once in Loop 2's safety table without an
  individual verification tag; not in Loop 3's ledger.

## (d) Planners & continuous collision detection (CCD)

- NVIDIA cuRobo — CUDA-Accelerated Robot Motion Generation in Milliseconds
  (NVIDIA Technical Blog). **[VERIFIED]** — Loop 4: 100 ms plan generation
  on AGX Orin, 500 Hz execution, swept-SDF continuous collision term, ~50 ms
  MPC replanning for moving obstacles; cross-confirmed with arXiv:2508.04146.
- Industrial Robot Motion Planning with GPUs: Integration of cuRobo for
  Extended DOF Systems (2025). arXiv:2508.04146. **[VERIFIED]** —
  corroborates the cuRobo blog's capability claims.
- cuRoboV2. arXiv:2603.05493. **[listed-but-not-found (check with the
  lead)]** — not located in any of the five source files read for this
  consolidation.
- Son, Jung & Kim (2025). NeuralSVCD for Efficient Swept Volume Collision
  Detection. arXiv:2509.00499. **[VERIFIED]** — repeatedly re-confirmed
  across Loops 3/4/7: swept volume is used only for collision
  detection/planning, never as an RL reward — the key evidence for the
  identified research gap.
- Deep Reactive Policy (DRP). arXiv:2509.06953. **[VERIFIED]** — full-text
  verified in Loop 5 (cuRobo vs DRP success-rate table: 39.50%/0% vs
  75.50%/66.67% on dynamic scenes); Loop 7 confirms it is CoRL 2025, with
  code + IMPACT weights released (pre-release) at
  github.com/deep-reactive-policy/drp, full release announced for 2026-06.
- Toward Generalist Neural Motion Planners for Robotic Manipulators:
  Challenges and Opportunities (2026). arXiv:2603.24318. **[ABSTRACT-ONLY]**
  — Loop 4's own methodology note states this loop was WebSearch
  abstract-level only; never re-verified via full text in a later loop.

## (e) Competitors

- A Trend-Aware Reinforcement Learning Approach for Adaptive Motion
  Planning of Robotic Manipulators in Dynamic Environments. EAAI 2026,
  ScienceDirect S0952197626005658. **[ABSTRACT-ONLY]** — the paper's
  existence and its general Trend-Learning + Adaptive-Reward-Shaping
  approach were confirmed real in Loop 3, but the full text remains
  paywalled as of Loop 7 (HTTP 403, no arXiv/institutional copy found) — its
  state representation and exact reward formulation are still abstract-level
  only.
- Frozen-occupancy-predictor collaborative-robot avoidance work.
  arXiv:2508.20457. **[VERIFIED]** — Loop 7: confirmed via a direct
  full-text quote ("encoder-decoder and safety critic frozen") that this
  work uses a frozen occupancy predictor as a perception module (not as an
  RL auxiliary task) — closed the keystone-novelty check.
- Transformer Human-Motion Forecasting + Safe RL for Co-Navigation. Front.
  Neurorobot. 2026, PMC12907402. **[CORRECTED]** — an earlier report's
  claim of "+21.6% success / −47.3% collisions" was a sub-agent
  hallucination; the corrected, verified figures are 98.0% vs PPO 90.4%
  (+7.6pp), constraint violations −60% (vs ORCA −83.3%). Also note: the
  paper is about telepresence mobile robots, not manipulators.

## (f) Auto-reward

- Ma et al. (2024). Eureka: Human-Level Reward Design via Coding LLMs.
  ICLR 2024, arXiv:2310.12931. **[VERIFIED]** — 83% win rate vs human
  experts, average +52%, confirmed directly from the abstract text.
- Ma et al. (2024). DrEureka: LM-Guided Sim-to-Real Transfer. RSS 2024,
  arXiv:2406.01967. **[UNVERIFIED]** — cited once in Loop 2's auto-reward
  table without an individual verification tag.
- Wang et al. (2024). Text2Reward: Reward Shaping with Language Models.
  ICLR 2024, arXiv:2309.11489. **[UNVERIFIED]** — cited once in Loop 2,
  never independently re-checked.
- Yu et al. (2023). Language to Rewards for Robotic Skill Synthesis (L2R).
  CoRL 2023, arXiv:2306.08647. **[UNVERIFIED]** — cited once in Loop 2,
  never independently re-checked.
- Sarukkai et al. (2024). Automated Rewards via LLM Progress Functions.
  arXiv:2410.09187. **[VERIFIED]** — count-based progress functions, 20×
  sample-efficiency claim, confirmed in Loop 3's ledger.
- Gao et al. (2026). RF-Agent. arXiv:2602.23876. **[VERIFIED]** —
  LLM + MCTS reward design across 17 tasks, confirmed in Loop 3's ledger.

## (g) Misc empirics

- Chen et al. (2022). DRL Trajectory Planning Under Uncertain Constraints
  (Franka SAC, relative position + velocity). Front. Neurorobot. 16:883562.
  **[VERIFIED]** — SAC reaches 100% safety within 6000 episodes; DDPG at
  10000 episodes still does not guarantee it.
- PL-TD3: Efficient TD3 with PER and LSTM in Dynamic Environments. Sci.
  Rep. 15:18331, PMC12106672. **[UNVERIFIED]** — cited once in Loop 2's
  algorithm table (87–88% success vs 71–73% baseline TD3) without an
  individual verification tag; never re-checked afterward.
- Ahmad, Hussain & Naeem (2024). Trajectory Planning of Robotic Manipulator
  in Dynamic Environment (DDPG, 7-DoF Fetch). arXiv:2403.16652.
  **[ABSTRACT-ONLY]** — paper's existence/authorship confirmed, but the
  specific 71.8% success / 18.5% collision figures come from paywalled
  full-text HTML, single-sourced — recheck before quoting the exact numbers.
- Proactive Dynamic Obstacle Avoidance for Safe HRC (SAC, human hand
  modeled as a moving cylinder, 93% success). Manufacturing Letters,
  S2213846324002359. **[VERIFIED]** — Loop 3's ledger confirms SAC +
  moving-cylinder hand model + generated dataset achieving 93%.
- Bricher & Mueller (2025). Deep-Learning Methods in ISO/TS 15066 HRC
  Safety. arXiv:2511.19094. **[UNVERIFIED]** — cited once in Loop 2 for the
  SSM separation-distance formula, without an individual verification tag;
  never re-checked afterward.
- MDRLAT (velocity-auxiliary-task work, Sensors 2021). **[listed-but-not-found
  (check with the lead)]** — not located in any of the five source files
  read for this consolidation.

---

## Do-not-cite-yet — re-verification queue before paper submission

Every UNVERIFIED and ABSTRACT-ONLY entry above, repeated here as the
worklist to clear before these numbers/claims go into the paper:

1. Chen et al. (2023). TTC-force Reward Shaping. AAMAS 2023. — exact
   formula unconfirmed (paywalled PDF).
2. RAIL. arXiv:2409.19190. — abstract-level only.
3. Latent Safety Filters. arXiv:2502.00935. — abstract-level only.
4. SafeDojo. arXiv:2606.20698. — abstract-level only.
5. Thumm & Althoff (2022). ICRA 2022, arXiv:2205.06311. — single-source,
   untagged.
6. SafeDiffuser. arXiv:2306.00148. — single-source, untagged.
7. sRCBF. arXiv:2412.03678. — single-source, untagged.
8. Achiam et al. (2017). CPO. ICML 2017. — single-source, untagged.
9. Toward Generalist Neural Motion Planners. arXiv:2603.24318. —
   abstract-level only.
10. EAAI 2026 Trend-Aware RL. ScienceDirect S0952197626005658. — full text
    still paywalled; state/reward details unconfirmed.
11. DrEureka. arXiv:2406.01967. — single-source, untagged.
12. Text2Reward. arXiv:2309.11489. — single-source, untagged.
13. L2R. arXiv:2306.08647. — single-source, untagged.
14. PL-TD3. Sci. Rep. 15:18331, PMC12106672. — single-source, untagged.
15. Ahmad, Hussain & Naeem (2024). DDPG Fetch. arXiv:2403.16652. — specific
    percentages are paywalled-full-text single-source.
16. Bricher & Mueller (2025). ISO/TS 15066. arXiv:2511.19094. —
    single-source, untagged.
