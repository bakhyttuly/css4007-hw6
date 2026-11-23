"""HW6 Part B: FrozenLake simulations (random vs fixed policy, slippery on/off).

Run:  python part_b.py
Outputs go to the outputs/ folder.
"""
import csv
import platform
import re
from pathlib import Path

import gymnasium as gym
import numpy as np

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
GOAL = 15
HOLES = {5, 7, 11, 12}
ACTION_NAMES = {0: "left", 1: "down", 2: "right", 3: "up"}

# Fixed policy (same table as in SUBMISSION.md, written before the experiments).
FIXED_POLICY = {0: 2, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 2, 10: 1, 13: 2, 14: 2}


def random_policy(state, rng):
    return int(rng.integers(4))


def fixed_policy(state, rng):
    return FIXED_POLICY[state]


POLICIES = {"random": random_policy, "fixed": fixed_policy}


def run_episode(env, policy_fn, seed, record_render=False):
    """Run one episode. Returns dict with steps, reward, outcome, trace and (optionally) frames."""
    rng = np.random.default_rng(seed)          # seeds the random action generator
    state, _ = env.reset(seed=seed)            # seeds the environment (slipping)
    trace = [state]
    actions = []
    frames = [env.render()] if record_render else []
    total_reward = 0.0
    steps = 0
    for _ in range(MAX_STEPS):
        action = policy_fn(state, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += reward
        actions.append(action)
        trace.append(state)
        if record_render:
            frames.append(env.render())
        if terminated or truncated:
            break
    # Outcome is decided by the final state, so a terminal on the last allowed step still counts.
    if state == GOAL:
        outcome = "goal"
    elif state in HOLES:
        outcome = "hole"
    else:
        outcome = "time limit"
    return {"steps": steps, "reward": total_reward, "outcome": outcome,
            "trace": trace, "actions": actions, "frames": frames}


def fmt_trace(result):
    parts = [str(result["trace"][0])]
    for a, s in zip(result["actions"], result["trace"][1:]):
        parts.append(f"-{ACTION_NAMES[a]}-> {s}")
    return " ".join(parts)


def main():
    episode_rows = []
    summary_rows = []
    first_success = None
    first_failure = None

    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode="ansi")
        for policy_name, policy_fn in POLICIES.items():
            condition = f"{policy_name}_slippery={slippery}"
            for batch_seed in BATCH_SEEDS:
                counts = {"goal": 0, "hole": 0, "time limit": 0}
                step_list = []
                for i in range(EPISODES_PER_BATCH):
                    episode_seed = 1000 * batch_seed + i
                    res = run_episode(env, policy_fn, episode_seed)
                    counts[res["outcome"]] += 1
                    step_list.append(res["steps"])
                    episode_rows.append([condition, batch_seed, episode_seed, policy_name,
                                         res["steps"], res["reward"], res["outcome"]])
                    info = (condition, batch_seed, episode_seed, res)
                    if slippery and policy_name == "fixed":
                        if res["outcome"] == "goal" and first_success is None:
                            first_success = info
                        if res["outcome"] != "goal" and first_failure is None:
                            first_failure = info
                summary_rows.append([slippery, policy_name, batch_seed,
                                     counts["goal"], counts["hole"], counts["time limit"],
                                     round(100 * counts["goal"] / EPISODES_PER_BATCH, 1),
                                     round(float(np.mean(step_list)), 2)])
        env.close()

    # --- one rendered episode (fixed policy, no slipping, first episode seed of batch 11)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="ansi")
    demo = run_episode(env, fixed_policy, 1000 * BATCH_SEEDS[0] + 0, record_render=True)
    env.close()
    with open(OUT / "example_episode.txt", "w", encoding="utf-8") as f:
        f.write("Example episode: fixed policy, is_slippery=False, seed 11000\n")
        f.write(f"Outcome: {demo['outcome']}, steps: {demo['steps']}, reward: {demo['reward']}\n\n")
        for k, frame in enumerate(demo["frames"]):
            clean = re.sub(r"\x1b\[41m(.)\x1b\[0m", r"[\1]", frame)  # mark agent position as [X]
            f.write(f"--- step {k} ---\n{clean}\n")

    # --- per-episode rows
    with open(OUT / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "batch_seed", "episode_seed", "policy", "steps", "total_reward", "outcome"])
        w.writerows(episode_rows)

    # --- batch summary (csv + markdown table matching SUBMISSION.md columns)
    header = ["Slippery?", "Policy", "Batch seed", "Goals /50", "Holes /50",
              "Time limits /50", "Success %", "Mean steps"]
    with open(OUT / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in summary_rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    (OUT / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- example traces (fixed policy, slippery): one success and one failure
    with open(OUT / "example_traces.txt", "w", encoding="utf-8") as f:
        for label, info in (("Successful trace", first_success), ("Failed trace", first_failure)):
            if info is None:
                f.write(f"{label}: this outcome never occurred in the fixed/slippery runs.\n\n")
                continue
            cond, bseed, eseed, res = info
            f.write(f"{label} ({cond}, batch seed {bseed}, episode seed {eseed}, outcome {res['outcome']}, "
                    f"{res['steps']} steps):\n{fmt_trace(res)}\n\n")

    # --- versions
    with open(OUT / "versions.txt", "w", encoding="utf-8") as f:
        f.write(f"python {platform.python_version()}\n")
        f.write(f"gymnasium {gym.__version__}\n")
        f.write(f"numpy {np.__version__}\n")

    print("\n".join(lines))
    print("\nVersions:", gym.__version__, np.__version__)


if __name__ == "__main__":
    main()
