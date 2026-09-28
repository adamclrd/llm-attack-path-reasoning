"""
Core logic shared by the generator, the runner and the analysis.

Nothing in this file calls a model or touches the network. Everything here is
deterministic, so it can be unit-tested and it is what the paper's correctness
claims rest on.
"""

from collections import deque


# ------------------------------------------------------------------ basics

def applicable(vuln, state):
    """A vulnerability can be used only if every precondition is in the state."""
    return set(vuln["pre"]).issubset(state)


def missing_predicates(vuln, state):
    """The preconditions that are not satisfied. This is what the C2 verifier
    hands back to the model instead of a bare rejection."""
    return sorted(set(vuln["pre"]) - set(state))


def verify_chain(chain, scenario):
    """Walks a proposed chain step by step against the accumulating state.

    Returns success, the count of valid and invalid steps, the first step that
    failed, and the steps that appear in no minimal chain (superfluous steps).
    """
    state = set(scenario["init"])
    valid = invalid = 0
    first_failure = None

    for vid in chain:
        v = scenario["vulns"].get(vid)
        if v is None:                          # identifier the model invented
            invalid += 1
            first_failure = first_failure or vid
            continue
        if applicable(v, state):
            state |= set(v["eff"])
            valid += 1
        else:
            invalid += 1
            first_failure = first_failure or vid

    success = set(scenario["goal"]).issubset(state) and invalid == 0
    return {
        "success": success,
        "valid_steps": valid,
        "invalid_steps": invalid,
        "first_failure": first_failure,
    }


# ------------------------------------------------------------------ search

def bfs_shortest(scenario, cap=400000):
    """Breadth-first expansion of the reachable state space.

    Returns (shortest chain or None, number of state expansions). Returns
    ("ABORTED", n) if the cap is hit, so a silent failure cannot be mistaken
    for "no solution exists".
    """
    init = frozenset(scenario["init"])
    goal = set(scenario["goal"])
    if goal.issubset(init):
        return [], 0

    queue = deque([(init, [])])
    seen = {init}
    expansions = 0

    while queue:
        state, path = queue.popleft()
        expansions += 1
        if expansions > cap:
            return "ABORTED", expansions
        for vid, v in scenario["vulns"].items():
            if not applicable(v, state):
                continue
            nxt = frozenset(state | set(v["eff"]))
            if nxt in seen:
                continue
            newpath = path + [vid]
            if goal.issubset(nxt):
                return newpath, expansions
            seen.add(nxt)
            queue.append((nxt, newpath))
    return None, expansions


def count_shortest_solutions(scenario, depth, state_cap=200000):
    """Counts how many distinct action sequences of length `depth` reach the goal.

    Layer by layer, keeping for each reachable state the number of sequences
    that produce it. States are deduplicated, sequences are not, so the count
    is a count of chains rather than of states.

    Returns (count, exact) where exact is False if the state cap was hit, in
    which case the count is a lower bound.
    """
    init = frozenset(scenario["init"])
    goal = set(scenario["goal"])
    layer = {init: 1}
    total = 0
    exact = True

    for _ in range(depth):
        nxt_layer = {}
        for state, ways in layer.items():
            for vid, v in scenario["vulns"].items():
                if not applicable(v, state):
                    continue
                nxt = frozenset(state | set(v["eff"]))
                if nxt == state:               # effects already held, no progress
                    continue
                if goal.issubset(nxt):
                    total += ways
                    continue
                nxt_layer[nxt] = nxt_layer.get(nxt, 0) + ways
            if len(nxt_layer) > state_cap:
                exact = False
                break
        if not exact:
            break
        layer = nxt_layer
        if not layer:
            break

    return total, exact


def branching_along(scenario, chain):
    """Mean and maximum number of applicable vulnerabilities at each state
    visited along a chain. This is the effective branching factor: two
    instances with the same distractor count can still differ on it, so it is
    the variable that should be reported next to the distractor count."""
    state = set(scenario["init"])
    counts = []
    for vid in chain:
        counts.append(sum(1 for v in scenario["vulns"].values()
                          if applicable(v, state)))
        v = scenario["vulns"].get(vid)
        if v and applicable(v, state):
            state |= set(v["eff"])
    if not counts:
        return 0.0, 0
    return sum(counts) / len(counts), max(counts)


def superfluous_steps(chain, minimal_chains):
    """Steps of a produced chain whose identifier appears in no minimal chain.

    Matches the definition given in Appendix C of the paper: a step that is
    merely out of order is an invalid transition, not a superfluous step.
    """
    useful = set()
    for m in minimal_chains:
        useful |= set(m)
    return sum(1 for vid in chain if vid not in useful)


def path_accuracy(chain, minimal_chains):
    """One minus the smallest normalised edit distance to any minimal chain.

    An alternative but valid minimal chain therefore scores 1 rather than
    being penalised for differing from the reference chain.
    """
    if not minimal_chains:
        return 0.0
    if not chain:
        return 0.0
    best = 1.0
    for m in minimal_chains:
        d = _levenshtein(chain, m) / max(len(chain), len(m))
        best = min(best, d)
    return 1.0 - best


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ------------------------------------------------------------------ baselines

def random_baseline(scenario, budget, rng):
    """Picks uniformly among the applicable vulnerabilities until the budget
    runs out. Every transition is valid by construction, so this baseline
    isolates the value of choosing well from the value of choosing legally."""
    state = set(scenario["init"])
    chain = []
    for _ in range(budget):
        options = [vid for vid, v in scenario["vulns"].items()
                   if applicable(v, state)]
        if not options:
            break
        vid = rng.choice(options)
        chain.append(vid)
        state |= set(scenario["vulns"][vid]["eff"])
        if set(scenario["goal"]).issubset(state):
            break
    return chain
