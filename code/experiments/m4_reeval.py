"""Re-evaluate saved M4-v2 policies under the active-dodge certified ladder
(no retraining): loads td3_{tag}_seed{n}.pt, runs final-stage eval."""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from godynur.env import DynArmEnv
from godynur.td3 import TD3
import importlib.util
spec = importlib.util.spec_from_file_location("m4", Path(__file__).parent / "m4_shield.py")
m4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m4)

EPISODES = 60
out = {}
for tag, shield in (("shield", True), ("noshield", False)):
    for seed in (0, 1):
        ck = Path(f"experiments/results/m4_shield/td3_{tag}_seed{seed}.pt")
        if not ck.exists():
            print("missing", ck); continue
        env = DynArmEnv(task="tabletop", n_obstacles=3, reward_mode="uoar",
                        seed=20_000 + seed)
        agent = TD3(env.state_dim, env.action_dim,
                    action_scale=env.action_bound[1], seed=seed)
        sd = torch.load(ck, weights_only=True)
        agent.actor.load_state_dict(sd["actor"])
        agent.q1.load_state_dict(sd["q1"]); agent.q2.load_state_dict(sd["q2"])
        env.set_difficulty(*m4.STAGES[-1])
        sel = m4.make_selector(env, agent, shield, seed=777 + seed)
        succ = coll = 0
        for _ in range(EPISODES):
            s = env.reset(); done = False
            while not done:
                a = (m4.shielded_actor_action(env, agent, sel, s) if shield
                     else agent.act(s, explore=False))
                s, _, done = env.step(a)
            if env.last_tau_star is not None: coll += 1
            elif env.on_goal >= env.goal_dwell: succ += 1
        out[f"{tag}_seed{seed}"] = {
            "success": succ / EPISODES, "collision": coll / EPISODES,
            "no_safe": sel.stats["no_safe"], "dodges": sel.stats["scaled"],
        }
        print(f"{tag} seed{seed}: succ {succ/EPISODES:.2f} coll {coll/EPISODES:.2f} "
              f"| no_safe {sel.stats['no_safe']} dodges {sel.stats['scaled']}", flush=True)
json.dump(out, open("experiments/results/m4_shield/m4_reeval_activedodge.json", "w"), indent=2)
