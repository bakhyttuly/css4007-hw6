"""Value iteration written from scratch (no libraries). Works for any GridMDP."""


def q_value(mdp, V, s, a):
    """Q(s,a) = sum over s' of P(s'|s,a) * (R + gamma * V(s'))."""
    return sum(p * (r + mdp.gamma * V[s2]) for p, s2, r in mdp.transitions(s, a))


def value_iteration(mdp, tol, max_iter):
    """Returns (V, history, iterations, final_delta, converged).

    history[k] is V_k (history[0] is V0 = all zeros). Every round uses only the
    previous round's values.
    """
    V = {s: 0.0 for s in mdp.states}
    history = [dict(V)]
    delta = float("inf")
    for k in range(1, max_iter + 1):
        new_V = {}
        for s in mdp.states:
            if mdp.is_terminal(s):
                new_V[s] = 0.0
            else:
                new_V[s] = max(q_value(mdp, V, s, a) for a in mdp.actions)
        delta = max(abs(new_V[s] - V[s]) for s in mdp.states)
        V = new_V
        history.append(dict(V))
        if delta < tol:
            return V, history, k, delta, True
    return V, history, max_iter, delta, False


def extract_policy(mdp, V, eps=1e-9):
    """Greedy policy from values.

    Tie-breaking rule: an action is tied with the best one if its Q-value is within
    eps (1e-9) of the maximum; among tied actions the first one in the order
    up, right, down, left is chosen. `ties` lists all tied actions for each state.
    """
    policy, ties = {}, {}
    for s in mdp.states:
        if mdp.is_terminal(s):
            continue
        qs = {a: q_value(mdp, V, s, a) for a in mdp.actions}
        best = max(qs.values())
        tied = [a for a in mdp.actions if best - qs[a] <= eps]
        policy[s] = tied[0]
        ties[s] = tied
    return policy, ties
