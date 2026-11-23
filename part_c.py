"""HW6 Part C: value iteration on the paper map and on the extended map.

Run:  python part_c.py      (then python checks.py for the rule checks)
Everything is saved in outputs/.
"""
import csv
import sys
from pathlib import Path

import numpy as np

from gridmdp import GridMDP, load_rules
from solver import extract_policy, q_value, value_iteration
from simulate import simulate_episode

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

ARROW = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
TIE_RULE = ("Tie-breaking rule: an action is tied with the best one if its Q-value is within 1e-9 of the "
            "maximum; among tied actions I choose the first one in the order up, right, down, left.")

# numbers from my paper calculations (Part A)
PAPER_V1 = {(1, 1): -1, (1, 2): -1, (1, 3): 10, (2, 1): -1, (2, 3): -1,
            (3, 1): -1, (3, 2): -1, (3, 3): -1, (3, 4): -1}
PAPER_ICE_V1_Q = {"up": -1.9, "right": -8.2, "down": -1.9, "left": -1.0}
PAPER_V2_Q = {
    (1, 2): {"up": -3.9, "right": 8, "down": -3.9, "left": -1.9},
    (1, 3): {"up": 8, "right": 10, "down": -1.9, "left": -3.9},
    (2, 3): {"up": 5.21, "right": -7.39, "down": -2.71, "left": -0.91},
}
PAPER_V2 = {(1, 2): 8, (1, 3): 10, (2, 3): 5.21}


def value_map(mdp, V):
    lines = []
    for r in range(1, mdp.n_rows + 1):
        cells = []
        for c in range(1, mdp.n_cols + 1):
            s = (r, c)
            if s not in V:
                cells.append("#".rjust(10))
            elif mdp.is_terminal(s):
                cells.append(f"{mdp.tile(s)}={V[s]:.3f}".rjust(10))
            else:
                cells.append(f"{V[s]:.3f}".rjust(10))
        lines.append(f"row {r}: " + "".join(cells))
    return lines


def policy_map(mdp, policy):
    lines = []
    for r in range(1, mdp.n_rows + 1):
        cells = []
        for c in range(1, mdp.n_cols + 1):
            s = (r, c)
            if s in policy:
                cells.append(ARROW[policy[s]].rjust(4))
            elif s in mdp.states:
                cells.append(mdp.tile(s).rjust(4))      # terminal tile letter
            else:
                cells.append("#".rjust(4))
        lines.append(f"row {r}: " + "".join(cells))
    return lines


def save(name, lines):
    text = "\n".join(lines) + "\n"
    (OUT / name).write_text(text, encoding="utf-8")
    print(text)


def fmt_s(s):
    return f"({s[0]},{s[1]})"


def solve_and_report(map_key, title, rules, filename):
    mdp = GridMDP(rules[map_key], rules)
    V, history, iters, delta, converged = value_iteration(
        mdp, rules["solver"]["tolerance"], rules["solver"]["max_iterations"])
    policy, ties = extract_policy(mdp, V)
    lines = [f"=== {title} ===", "Map:"] + ["  " + row for row in mdp.rows]
    lines += ["", f"Discount: {mdp.gamma}, tolerance: {rules['solver']['tolerance']}, "
                  f"iteration cap: {rules['solver']['max_iterations']}",
              f"Iterations used: {iters}", f"Final largest change: {delta:.3e}",
              "Converged: " + ("yes" if converged else "NO - iteration cap reached (nonconvergence)"),
              "", "Final values V*:"] + value_map(mdp, V)
    lines += ["", "Policy (greedy w.r.t. final values):"] + policy_map(mdp, policy)
    lines += ["", TIE_RULE, "Tied actions (only tiles with more than one best action):"]
    tied_any = False
    for s in mdp.states:
        if s in ties and len(ties[s]) > 1:
            tied_any = True
            lines.append(f"  {fmt_s(s)}: {', '.join(ties[s])} -> chose {policy[s]}")
    if not tied_any:
        lines.append("  none")
    lines += ["", "Q-values of all actions at the final values:"]
    for s in mdp.states:
        if not mdp.is_terminal(s):
            qs = ", ".join(f"{a} {q_value(mdp, V, s, a):.4f}" for a in mdp.actions)
            lines.append(f"  {fmt_s(s)} ({mdp.tile(s)}): {qs}")
    save(filename, lines)
    return mdp, V, history, policy, ties, iters, delta, converged


def reconcile_with_paper(mdp, history):
    lines = ["=== Part C solver vs. my paper work (Part A, paper map) ===", ""]
    ok = True
    V1, V2 = history[1], history[2]
    lines.append("V1 map from code:")
    lines += value_map(mdp, V1)
    lines.append("")
    lines.append("V1 for every nonterminal tile, code vs paper:")
    for s, paper in PAPER_V1.items():
        good = abs(V1[s] - paper) < 1e-9
        ok &= good
        lines.append(f"  {fmt_s(s)}: code {V1[s]:.4f}, paper {paper}  {'OK' if good else 'MISMATCH'}")
    lines.append("")
    lines.append("V1 of the ice tile (2,3), all four actions (computed from V0 = 0), code vs paper:")
    for a, paper in PAPER_ICE_V1_Q.items():
        q = q_value(mdp, history[0], (2, 3), a)
        good = abs(q - paper) < 1e-9
        ok &= good
        lines.append(f"  {a}: code {q:.4f}, paper {paper}  {'OK' if good else 'MISMATCH'}")
    lines.append("")
    lines.append("V2 at the three requested tiles (using only V1), code vs paper:")
    for s, qs_paper in PAPER_V2_Q.items():
        lines.append(f"  {fmt_s(s)}:")
        for a, paper in qs_paper.items():
            q = q_value(mdp, history[1], s, a)
            good = abs(q - paper) < 1e-9
            ok &= good
            lines.append(f"    {a}: code {q:.4f}, paper {paper}  {'OK' if good else 'MISMATCH'}")
        good = abs(V2[s] - PAPER_V2[s]) < 1e-9
        ok &= good
        lines.append(f"    V2 = max: code {V2[s]:.4f}, paper {PAPER_V2[s]}  {'OK' if good else 'MISMATCH'}")
    lines += ["", "All numbers match my paper work." if ok else "THERE ARE MISMATCHES - see above."]
    save("part_c_vs_paper.txt", lines)
    return ok


def run_simulations(maps):
    """maps: {name: (mdp, policy, V)}. Returns episode rows and batch summary rows."""
    sim = rules_global["simulation"]
    episodes, summary = [], []
    for name, (mdp, policy, V) in maps.items():
        for batch_seed in sim["batch_seeds"]:
            counts = {"goal": 0, "hole": 0, "time limit": 0}
            steps_list, ret_list = [], []
            for i in range(sim["episodes_per_batch"]):
                seed = 1000 * batch_seed + i
                res = simulate_episode(mdp, policy, seed, sim["max_steps"])
                counts[res["outcome"]] += 1
                steps_list.append(res["steps"])
                ret_list.append(res["discounted_return"])
                path = ">".join(f"{s[0]}{s[1]}" for s in res["trace"])
                episodes.append([name, batch_seed, seed, res["steps"], res["total_reward"],
                                 round(res["discounted_return"], 6), res["outcome"],
                                 int((2, 3) in res["trace"]), path])
            summary.append([name, batch_seed, counts["goal"], counts["hole"], counts["time limit"],
                            round(float(np.mean(steps_list)), 2), round(float(np.mean(ret_list)), 4)])
    return episodes, summary


def main():
    global rules_global
    rules = rules_global = load_rules()
    results = {}
    for key, title, fname, label in (
            ("paper_map", "Paper map (Part A rules)", "part_c_values_paper_map.txt", "paper"),
            ("extended_map", "Extended map (sand D = -2, shallow water W = -5)",
             "part_c_values_extended_map.txt", "extended")):
        results[label] = solve_and_report(key, title, rules, fname)

    mdp_p, V_p, hist_p, pol_p, ties_p, it_p, d_p, conv_p = results["paper"]
    mdp_e, V_e, hist_e, pol_e, ties_e, it_e, d_e, conv_e = results["extended"]

    ok = reconcile_with_paper(mdp_p, hist_p)

    # --- comparison of the two maps
    lines = ["=== Comparison: paper map vs extended map ===", "",
             "tile | letters (paper/ext) | V* paper | V* ext | change | action paper | action ext | action changed?"]
    for s in mdp_p.states:
        if mdp_p.is_terminal(s):
            continue
        ap, ae = pol_p[s], pol_e[s]
        lines.append(f"{fmt_s(s)} | {mdp_p.tile(s)}/{mdp_e.tile(s)} | {V_p[s]:.3f} | {V_e[s]:.3f} | "
                     f"{V_e[s] - V_p[s]:+.3f} | {ap} | {ae} | {'YES' if ap != ae else 'no'}")
    lines += ["", f"Start tile {fmt_s(mdp_p.start)}: V* paper = {V_p[mdp_p.start]:.4f}, "
                  f"V* extended = {V_e[mdp_e.start]:.4f}"]
    save("part_c_comparison.txt", lines)

    # --- simulations (custom simulator, Part B seed scheme)
    maps = {"paper": (mdp_p, pol_p, V_p), "extended": (mdp_e, pol_e, V_e)}
    episodes, summary = run_simulations(maps)
    with open(OUT / "part_c_episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map", "batch_seed", "episode_seed", "steps", "total_reward",
                    "discounted_return", "outcome", "visited_ice", "path (row,col digits)"])
        w.writerows(episodes)
    header = ["Map", "Batch seed", "Goals /50", "Holes /50", "Time limits /50",
              "Mean steps", "Mean discounted return"]
    with open(OUT / "part_c_simulation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary)
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md += ["| " + " | ".join(str(x) for x in row) + " |" for row in summary]
    extra = ["", "Overall (150 episodes per map):"]
    for name, (mdp, policy, V) in maps.items():
        rows = [e for e in episodes if e[0] == name]
        mean_ret = np.mean([e[5] for e in rows])
        ice = [e for e in rows if e[7] == 1]
        ice_holes = sum(1 for e in ice if e[6] == "hole")
        holes = sum(1 for e in rows if e[6] == "hole")
        extra.append(f"- {name}: mean discounted return {mean_ret:.4f} vs V*(start) = {V[mdp.start]:.4f}; "
                     f"episodes that stepped on the ice tile: {len(ice)}; "
                     f"episodes ending in the hole: {holes} (of these {ice_holes} had visited the ice tile)")
    (OUT / "part_c_simulation_summary.md").write_text("\n".join(md + extra) + "\n", encoding="utf-8")
    print("\n".join(md + extra))
    print("\nPaper-map reconciliation:", "OK" if ok else "MISMATCH")
    print(f"Iterations: paper {it_p} (final change {d_p:.2e}), extended {it_e} (final change {d_e:.2e})")


if __name__ == "__main__":
    main()
