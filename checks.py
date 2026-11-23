"""Rule checks for Part C. Run: python checks.py  (output saved in outputs/rule_checks.txt)."""
import sys
from pathlib import Path

from gridmdp import GridMDP, load_rules
from solver import extract_policy, value_iteration
from simulate import simulate_episode

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

rules = load_rules()
paper = GridMDP(rules["paper_map"], rules)
ext = GridMDP(rules["extended_map"], rules)
results = []


def norm(outs):
    return sorted((round(p, 9), s, r) for p, s, r in outs)


def check(name, condition, detail=""):
    results.append((bool(condition), name, detail))


# 1. walls and boundaries: blocked move keeps you in place, reward of the tile where you end
o = paper.transitions((2, 1), "right")
check("Wall: (2,1) moving right into wall (2,2) stays on (2,1), reward -1 (floor)",
      norm(o) == norm([(1.0, (2, 1), -1)]), str(o))
o = paper.transitions((1, 2), "down")
check("Wall: (1,2) mud moving down into wall (2,2) stays on (1,2), reward -3 (mud)",
      norm(o) == norm([(1.0, (1, 2), -3)]), str(o))
o = paper.transitions((1, 1), "up")
check("Boundary: (1,1) moving up leaves the map -> stays, reward -1",
      norm(o) == norm([(1.0, (1, 1), -1)]), str(o))
o = paper.transitions((3, 1), "left")
check("Boundary: (3,1) moving left leaves the map -> stays, reward -1",
      norm(o) == norm([(1.0, (3, 1), -1)]), str(o))
o = ext.transitions((3, 4), "down")
check("Boundary on extended map: (3,4) water moving down stays, reward -5 (water, tile where the move ends)",
      norm(o) == norm([(1.0, (3, 4), -5)]), str(o))

# 2. ice: probabilities sum to 1, exact outcomes, slips into the wall stay on the ice
for a in paper.actions:
    total = sum(p for p, _, _ in paper.transitions((2, 3), a))
    check(f"Ice (2,3), action {a}: probabilities sum to 1", abs(total - 1.0) < 1e-12, f"sum = {total}")
o = paper.transitions((2, 3), "up")
check("Ice (2,3) up: 0.8 -> (1,3), slip left hits wall -> stays on ice (0.1), slip right -> hole (0.1)",
      norm(o) == norm([(0.8, (1, 3), -1), (0.1, (2, 3), -1), (0.1, (2, 4), -10)]), str(o))
o = paper.transitions((2, 3), "left")
check("Ice (2,3) left: 0.8 into wall -> stays on ice, slips up -> (1,3) and down -> (3,3), 0.1 each",
      norm(o) == norm([(0.8, (2, 3), -1), (0.1, (1, 3), -1), (0.1, (3, 3), -1)]), str(o))
o = paper.transitions((2, 3), "right")
check("Ice (2,3) right: 0.8 -> hole (-10), slips up -> (1,3) and down -> (3,3), 0.1 each",
      norm(o) == norm([(0.8, (2, 4), -10), (0.1, (1, 3), -1), (0.1, (3, 3), -1)]), str(o))
check("Slipping depends on the tile you leave: non-ice tiles are deterministic (all tiles but (2,3) have one outcome)",
      all(len(paper.transitions(s, a)) == 1 for s in paper.states if not paper.is_terminal(s)
          and paper.tile(s) != "I" for a in paper.actions))

# 3. floor rewards
o = paper.transitions((1, 1), "right")
check("Mud: entering (1,2) gives -3", norm(o) == norm([(1.0, (1, 2), -3)]), str(o))
o = ext.transitions((3, 1), "right")
check("Sand: entering (3,2) gives -2 (extended map)", norm(o) == norm([(1.0, (3, 2), -2)]), str(o))
o = ext.transitions((3, 3), "right")
check("Water: entering (3,4) gives -5 (extended map)", norm(o) == norm([(1.0, (3, 4), -5)]), str(o))
o = paper.transitions((1, 3), "right")
check("Goal: entering G gives +10", norm(o) == norm([(1.0, (1, 4), 10)]), str(o))
check("Sand and water moves are deterministic (one outcome with probability 1)",
      all(len(ext.transitions(s, a)) == 1 and ext.transitions(s, a)[0][0] == 1.0
          for s in [(3, 2), (3, 4)] for a in ext.actions))
check("Original floor rewards: floor -1, start -1, ice -1, mud -3, hole -10",
      [paper.reward(s) for s in [(1, 1), (3, 1), (2, 3), (1, 2), (2, 4)]] == [-1, -1, -1, -3, -10])

# 4. termination
for name, mdp in (("paper", paper), ("extended", ext)):
    check(f"Terminal ({name} map): no transitions out of G or H",
          all(mdp.transitions(s, a) == [] for s in [(1, 4), (2, 4)] for a in mdp.actions))
    V, hist, it, delta, conv = value_iteration(mdp, rules["solver"]["tolerance"], rules["solver"]["max_iterations"])
    check(f"Terminal ({name} map): value of G and H stays 0 in every round",
          all(h[(1, 4)] == 0 and h[(2, 4)] == 0 for h in hist))
    policy, _ = extract_policy(mdp, V)
    bad, mism = 0, 0
    for i in range(50):
        res = simulate_episode(mdp, policy, 11000 + i)
        if any(mdp.is_terminal(s) for s in res["trace"][:-1]):
            bad += 1                       # a move happened after a terminal state
        manual = sum(0.9 ** t * r for t, r in enumerate(res["rewards"]))
        if abs(manual - res["discounted_return"]) > 1e-12 or len(res["rewards"]) != res["steps"]:
            mism += 1
    check(f"Terminal ({name} map): in 50 simulated episodes no move or reward happens after G/H "
          "(episode ends on first terminal state, rewards = one per move)", bad == 0 and mism == 0,
          f"episodes with moves after terminal: {bad}, return mismatches: {mism}")

lines = ["=== Rule checks (Part C) ==="]
for good, name, detail in results:
    lines.append(("PASS  " if good else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
n_fail = sum(1 for g, _, _ in results if not g)
lines.append("")
lines.append(f"{len(results) - n_fail} of {len(results)} checks passed.")
text = "\n".join(lines) + "\n"
(OUT / "rule_checks.txt").write_text(text, encoding="utf-8")
print(text)
sys.exit(1 if n_fail else 0)
