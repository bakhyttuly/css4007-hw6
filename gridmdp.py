"""Grid World MDP model.

All floor rules (rewards, ice slipping, terminals, walls) are read from data/maps.json,
so the solver never mentions a tile letter. Adding a new floor type only needs a new
entry in that file.
"""
import json
from pathlib import Path

DIRS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}


def load_rules(path=None):
    path = Path(path) if path else Path(__file__).parent / "data" / "maps.json"
    return json.loads(path.read_text(encoding="utf-8"))


class GridMDP:
    """States are (row, col) tuples, numbered from 1 like in the paper work."""

    def __init__(self, rows, rules):
        self.rows = list(rows)
        self.tiles = rules["tiles"]
        self.actions = list(rules["actions"])
        self.gamma = rules["discount"]
        self.n_rows = len(self.rows)
        self.n_cols = len(self.rows[0])
        for row in self.rows:
            if len(row) != self.n_cols:
                raise ValueError("all map rows must have the same length")
            for ch in row:
                if ch not in self.tiles:
                    raise ValueError(f"unknown tile letter {ch!r}")
        self.states = [(r, c)
                       for r in range(1, self.n_rows + 1)
                       for c in range(1, self.n_cols + 1)
                       if self.tiles[self.tile((r, c))].get("traversable", True)]
        self._state_set = set(self.states)
        starts = [s for s in self.states if self.tiles[self.tile(s)]["meaning"] == "start"]
        if len(starts) != 1:
            raise ValueError("map must have exactly one start tile")
        self.start = starts[0]

    def tile(self, s):
        return self.rows[s[0] - 1][s[1] - 1]

    def is_terminal(self, s):
        return bool(self.tiles[self.tile(s)].get("terminal", False))

    def outcome_name(self, s):
        """Name of the outcome of an episode that ended in state s."""
        t = self.tiles[self.tile(s)]
        if t.get("terminal"):
            return "goal" if t["meaning"] == "goal" else "hole"
        return "time limit"

    def move(self, s, direction):
        """Where a physical move in `direction` ends. Walls and map edges keep you in place."""
        dr, dc = DIRS[direction]
        nxt = (s[0] + dr, s[1] + dc)
        return nxt if nxt in self._state_set else s

    def reward(self, s):
        """Reward for a move that ends on state s (also when the move was blocked)."""
        return self.tiles[self.tile(s)]["reward"]

    def transitions(self, s, action):
        """List of (probability, next_state, reward). Empty for terminal states."""
        if self.is_terminal(s):
            return []
        t = self.tiles[self.tile(s)]
        if t.get("movement") == "slippery":
            outcomes = [(t["intended"], action)]
            outcomes += [(t["perpendicular_each"], d) for d in PERPENDICULAR[action]]
        else:
            outcomes = [(1.0, action)]
        result = []
        for p, direction in outcomes:
            nxt = self.move(s, direction)
            result.append((p, nxt, self.reward(nxt)))
        return result
