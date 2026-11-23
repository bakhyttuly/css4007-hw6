"""Custom simulator: follow a policy on a GridMDP (Part B seed scheme, 100-step limit)."""
import numpy as np


def simulate_episode(mdp, policy, seed, max_steps=100):
    rng = np.random.default_rng(seed)
    state = mdp.start
    trace, actions, rewards = [state], [], []
    for _ in range(max_steps):
        action = policy[state]
        outs = mdp.transitions(state, action)
        idx = int(rng.choice(len(outs), p=[o[0] for o in outs]))
        _, state, r = outs[idx]
        actions.append(action)
        rewards.append(r)
        trace.append(state)
        if mdp.is_terminal(state):      # no moves and no rewards after termination
            break
    discounted = sum(mdp.gamma ** t * r for t, r in enumerate(rewards))
    return {"trace": trace, "actions": actions, "rewards": rewards,
            "steps": len(rewards), "total_reward": sum(rewards),
            "discounted_return": discounted,
            "outcome": mdp.outcome_name(state)}
