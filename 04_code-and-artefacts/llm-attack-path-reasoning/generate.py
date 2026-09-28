"""
Scenario generator, second version.

What changed compared to build_hard.py, and why:

1. Several independent instances per cell instead of one. The first version
   produced exactly one instance per (depth, distractor count) cell, so a cell
   rate rested on repeated calls against a single logical structure. Here the
   number of instances per cell is a command-line argument and every instance
   is drawn with its own seed.

2. The instances are structurally different, not just relabelled. In the first
   version the reference chain of a given depth was a fixed list, so the three
   distractor levels at that depth shared one logical structure and the whole
   study contained four distinct structures. Here the chain is drawn by random
   walk over a grammar of step kinds, and hosts, ports, services, versions and
   assets are drawn from pools, so two instances of the same depth differ in
   shape and not only in tokens.

3. All four distractor types are actually instantiated. In the first version
   the inert quota was computed as a remainder that came out at zero for every
   level, so the set contained no T1 distractor at all while the paper reported
   a composition dominated by T1. The proportions are now explicit arguments
   and the realised composition is written into each instance.

4. The near-miss family no longer contains functional duplicates. The first
   version widened its pool by renaming, but the rename was a no-op on the
   effects, so at the two higher levels several near-miss distractors had
   identical preconditions and effects. Near-misses are now derived from the
   reference steps of the instance itself, so each one is a near-miss of that
   instance and they are checked for duplication before being kept.

5. Each instance records the quantities the analysis needs: the number of
   shortest solutions, the effective branching factor along the reference
   path, the realised per-type distractor counts and the BFS expansion count.

Usage:
    python generate.py --instances 8 --depths 4 5 6 7 --k 20 40 60 --out scenarios.json
    python generate.py --instances 8 --identifiers real --out scenarios_real.json
"""

import argparse
import json
import random

from engine import bfs_shortest, count_shortest_solutions, branching_along


# ------------------------------------------------------------------ pools

SERVICES = ["svc_alpha", "svc_beta", "svc_gamma", "svc_delta", "svc_epsilon",
            "svc_zeta", "svc_eta", "svc_theta"]
KERNELS = ["kern", "libc", "drv_net", "drv_blk"]
PORTS = [80, 443, 8080, 8443, 9000, 9090, 5000, 3000]
VERSIONS = ["1.0", "1.2", "2.1", "2.4", "3.0", "3.7", "4.2", "5.4"]
ACCOUNTS = ["acct_svc", "acct_ops", "acct_back", "acct_repl", "acct_task"]
FLAGS = ["debug_on", "world_writable", "no_sandbox", "legacy_auth"]
ALLOC = [None]        # current host allocator, used by _plausible_substitute
ASSETS = ["asset_x", "asset_y", "asset_z", "asset_w", "asset_v", "asset_u"]


# ------------------------------------------------------------------ chain grammar
#
# The attacker's position is a small record: which hosts carry a shell, which
# carry root, which credentials are held. A step kind is applicable when that
# record satisfies its requirement, and it advances the record. Drawing a chain
# is a random walk over this grammar, which is what makes two instances of the
# same depth structurally different rather than relabelled.


def make_allocator(rng):
    """Hands out host names from one shared pool.

    Every family, reference steps and distractors alike, draws from the same
    range. An earlier version drew reference hosts from h01-h89 and distractor
    hosts from h90-h99, which made the host number a perfect predictor of
    distractor status: the range was a surface cue of exactly the kind the
    identifier substitution of Section 4.4 is meant to remove.
    """
    pool = ["h%02d" % i for i in range(1, 100)]
    rng.shuffle(pool)
    it = iter(pool)

    def alloc():
        return next(it)
    return alloc


def build_chain(depth, rng, alloc):
    """Returns (static_facts, steps) with exactly `depth` steps, the last one
    being the exfiltration step. Raises ValueError if the walk cannot reach the
    requested depth, in which case the caller retries with another seed."""
    used_hosts = set()
    entry = alloc()
    used_hosts.add(entry)

    static = []
    steps = []
    shells = set()
    roots = set()
    creds = {}                                   # account -> host it is valid on

    # step 1 is always initial access: nothing else can start from no capability
    port = rng.choice(PORTS)
    svc = rng.choice(SERVICES)
    ver = rng.choice(VERSIONS)
    static += [f"reachable(external,{entry},{port})", f"runs({entry},{svc},{ver})"]
    steps.append(("access",
                  [f"reachable(external,{entry},{port})", f"runs({entry},{svc},{ver})"],
                  [f"hasShell({entry},attacker)"]))
    shells.add(entry)
    cur = entry

    while len(steps) < depth - 1:
        options = []
        if cur in shells and cur not in roots:
            options += ["escalate_kernel", "escalate_config"]
        if roots:
            options.append("credential")
        if creds:
            options.append("lateral_cred")
        if shells:
            options.append("lateral_trust")

        # The exfiltration step reads a capability on the host that holds the
        # asset, so the step just before it has to produce a host-scoped
        # capability. A credential step produces hasCred(attacker,...), which
        # names no host, and would leave the asset attached to the attacker
        # rather than to a machine.
        if len(steps) == depth - 2:
            options = [o for o in options if o != "credential"]

        if not options:
            raise ValueError("walk stuck")

        kind = rng.choice(options)

        if kind == "escalate_kernel":
            k, kv = rng.choice(KERNELS), rng.choice(VERSIONS)
            static.append(f"runs({cur},{k},{kv})")
            steps.append((kind, [f"hasShell({cur},attacker)", f"runs({cur},{k},{kv})"],
                          [f"hasPriv({cur},attacker,root)"]))
            roots.add(cur)

        elif kind == "escalate_config":
            fl = rng.choice(FLAGS)
            static.append(f"config({cur},{fl})")
            steps.append((kind, [f"hasShell({cur},attacker)", f"config({cur},{fl})"],
                          [f"hasPriv({cur},attacker,root)"]))
            roots.add(cur)

        elif kind == "credential":
            src = rng.choice(sorted(roots))
            acct = rng.choice([a for a in ACCOUNTS if a not in creds]
                              or ACCOUNTS)
            tgt = alloc()
            used_hosts.add(tgt)
            steps.append((kind, [f"hasPriv({src},attacker,root)"],
                          [f"hasCred(attacker,{acct})",
                           f"credValidOn({acct},{tgt})"]))
            creds[acct] = tgt

        elif kind == "lateral_cred":
            acct = rng.choice(sorted(creds))
            tgt = creds[acct]
            steps.append((kind, [f"hasCred(attacker,{acct})",
                                 f"credValidOn({acct},{tgt})"],
                          [f"hasShell({tgt},attacker)"]))
            shells.add(tgt)
            cur = tgt

        elif kind == "lateral_trust":
            src = rng.choice(sorted(shells))
            tgt = alloc()
            used_hosts.add(tgt)
            static.append(f"trusts({tgt},{src})")
            steps.append((kind, [f"hasShell({src},attacker)", f"trusts({tgt},{src})"],
                          [f"hasShell({tgt},attacker)"]))
            shells.add(tgt)
            cur = tgt

    # final step: exfiltrate from a host the walk actually reached
    last_eff = steps[-1][2][0]
    host = last_eff.split("(")[1].split(",")[0]
    asset = "asset_x"
    static.append(f"dataAt({host},{asset})")
    steps.append(("exfil", [last_eff, f"dataAt({host},{asset})"],
                  [f"exfiltrated({asset})"]))

    return static, steps, sorted(used_hosts)


# ------------------------------------------------------------------ distractors

def _plausible_substitute(value, rng):
    """Returns a different but equally plausible value of the same kind.

    The point is that the substitute must not look synthetic. An earlier
    version appended a suffix such as "_alt", which made every altered atom
    recognisable on sight: the suffix was a perfect predictor of distractor
    status, so a model could discard the whole near-miss family by matching a
    string rather than by comparing preconditions. That is the same kind of
    surface cue as the identifier leak of Section 4.8.1, and it has to be
    removed for the same reason.
    """
    if value.isdigit():
        pool = [str(p) for p in PORTS]
    elif value in VERSIONS:
        pool = VERSIONS
    elif value in SERVICES:
        pool = SERVICES
    elif value in KERNELS:
        pool = KERNELS
    elif value in ACCOUNTS:
        pool = ACCOUNTS
    elif value in FLAGS:
        pool = FLAGS
    elif value in ("root", "admin", "user"):
        pool = ["root", "admin", "user"]
    elif value in ASSETS:
        pool = ASSETS
    elif len(value) == 3 and value[0] == "h" and value[1:].isdigit():
        return ALLOC[0]()
    else:
        pool = None
    if not pool:
        return None
    other = [x for x in pool if x != value]
    return rng.choice(other) if other else None


def make_inert(n, rng, alloc, idx):
    """T1. Preconditions that no sequence can satisfy.

    They name a capability on a host that appears nowhere else, or an account
    that is never granted. Nothing marks them as artificial: whether they are
    reachable can only be decided by checking what the other vulnerabilities
    produce, which is the single applicability check the typology calls for.
    """
    out = []
    for i in range(n):
        ghost = alloc()
        target = alloc()
        pick = rng.random()
        if pick < 0.4:
            pre = [f"hasPriv({ghost},attacker,root)"]
        elif pick < 0.7:
            pre = [f"hasCred(attacker,{rng.choice(ACCOUNTS)})",
                   f"credValidOn({rng.choice(ACCOUNTS)},{ghost})"]
        else:
            pre = [f"hasShell({ghost},attacker)",
                   f"runs({ghost},{rng.choice(SERVICES)},{rng.choice(VERSIONS)})"]
        out.append((f"D-INE-{idx}-{i:02d}", pre,
                    [f"hasShell({target},attacker)"], "unsat"))
    return out


def make_nearmiss(n, rng, steps, alloc, idx):
    """T2. Each one is derived from a reference step of this instance, so it is
    a near-miss of the chain the model actually has to find.

    Two sub-families, in roughly equal parts:
      unsatisfiable  one precondition atom is replaced by a different but
                     equally plausible value of the same kind, which nothing
                     produces, so the step can never fire
      wrong effect   the preconditions match a reference step exactly, but the
                     effect lands on a host that is not on the path, so the
                     chain silently breaks two or three steps later
    """
    out = []
    seen = set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        kind, pre, eff = steps[rng.randrange(len(steps))]
        if rng.random() < 0.5 and pre:                       # unsatisfiable
            j = rng.randrange(len(pre))
            head, args = pre[j].split("(", 1)
            args = args.rstrip(")").split(",")
            slot = rng.randrange(len(args))
            sub = _plausible_substitute(args[slot], rng)
            if sub is None:
                continue
            args[slot] = sub
            new_pre = list(pre)
            new_pre[j] = f"{head}({','.join(args)})"
            new_eff = list(eff)
            family = "unsat"
        else:                                                 # wrong effect
            ghost = alloc()
            new_pre = list(pre)
            new_eff = []
            for e in eff:
                head, args = e.split("(", 1)
                args = args.rstrip(")").split(",")
                if args and args[0].startswith("h"):
                    args[0] = ghost
                new_eff.append(f"{head}({','.join(args)})")
            if new_eff == list(eff):
                continue
            family = "applicable"
        sig = (tuple(sorted(new_pre)), tuple(sorted(new_eff)))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((f"D-NM-{idx}-{len(out):02d}", new_pre, new_eff, family))
    return out


def make_decoys(n, rng, steps, idx):
    """T3. Applicable, with an effect that resembles the goal without being it:
    exfiltration of a different asset."""
    out = []
    caps = [e for _, _, eff in steps for e in eff]
    for i in range(n):
        entry = caps[i % len(caps)]
        asset = ASSETS[1 + (i % (len(ASSETS) - 1))]           # never asset_x
        out.append((f"D-DEC-{idx}-{i:02d}", [entry],
                    [f"exfiltrated({asset})"], "applicable"))
    return out


def make_deadends(n_vulns, branch_len, rng, steps, alloc, idx):
    """T4. Branches that hang off a capability produced early in the chain and
    really are applicable one after another, but whose effects never contribute
    to the goal. Only the absence of progress distinguishes them, so they are
    the family that cannot be filtered by inspecting one vulnerability."""
    out = []
    entries = [e for _, _, eff in steps[:max(1, len(steps) // 2)] for e in eff]
    b = 0
    while len(out) < n_vulns:
        entry = entries[b % len(entries)]
        for i in range(branch_len):
            if len(out) >= n_vulns:
                break
            cap = f"hasShell({alloc()},attacker)"
            out.append((f"D-DEAD-{idx}-{b}-{i}", [entry], [cap],
                        "applicable"))
            entry = cap
        b += 1
    return out


# ------------------------------------------------------------------ identifiers

def anonymise(scenario, rng):
    """Replaces every identifier with an opaque token drawn in random order.

    Without this the identifiers leak the answer. A label such as V-REF-3
    states both that the vulnerability is on the reference chain and where it
    sits in it, and a shared D- prefix states that a vulnerability is a
    distractor, so the chain can be reproduced by matching on names instead of
    reasoning over preconditions and effects. This is the leak documented in
    Section 4.8.1 of the paper. The role of each vulnerability is kept in the
    scenario file for the analysis and is never shown in the prompt.
    """
    ids = list(scenario["vulns"].keys())
    tokens = [f"VULN-{i:03d}" for i in range(1, len(ids) + 1)]
    rng.shuffle(tokens)
    mapping = dict(zip(ids, tokens))

    roles = {mapping[old]: scenario["vulns"][old]["role"] for old in ids}
    scenario["vulns"] = {
        mapping[old]: {"pre": v["pre"], "eff": v["eff"]}
        for old, v in scenario["vulns"].items()
    }
    scenario["roles"] = roles
    scenario["reference_chain"] = [mapping[x] for x in scenario["reference_chain"]]
    if isinstance(scenario.get("bfs_solution"), list):
        scenario["bfs_solution"] = [mapping[x] for x in scenario["bfs_solution"]]
    # present them in token order so that position carries no information either
    scenario["vulns"] = dict(sorted(scenario["vulns"].items()))
    return scenario


def keep_real(scenario, rng):
    """The other identifier regime: plausible published identifiers, same
    logical structure. Used for the identifier-substitution control, which
    needs two instantiations of one structure rather than two structures."""
    ids = list(scenario["vulns"].keys())
    pool = [f"CVE-20{rng.randint(15, 24)}-{rng.randint(1000, 29999)}"
            for _ in ids]
    seen, uniq = set(), []
    for c in pool:
        while c in seen:
            c = f"CVE-20{rng.randint(15, 24)}-{rng.randint(1000, 29999)}"
        seen.add(c)
        uniq.append(c)
    mapping = dict(zip(ids, uniq))
    roles = {mapping[old]: scenario["vulns"][old]["role"] for old in ids}
    scenario["vulns"] = {mapping[o]: {"pre": v["pre"], "eff": v["eff"]}
                         for o, v in scenario["vulns"].items()}
    scenario["roles"] = roles
    scenario["reference_chain"] = [mapping[x] for x in scenario["reference_chain"]]
    if isinstance(scenario.get("bfs_solution"), list):
        scenario["bfs_solution"] = [mapping[x] for x in scenario["bfs_solution"]]
    scenario["vulns"] = dict(sorted(scenario["vulns"].items()))
    return scenario


# ------------------------------------------------------------------ assembly

def build_instance(depth, k, seed, mix, branch_len, identifiers):
    """Builds one instance and verifies it. Returns None if the draw failed the
    depth check, so that the caller can retry with the next seed rather than
    silently keeping an instance whose true depth is not the announced one."""
    rng = random.Random(seed)
    alloc = make_allocator(rng)
    ALLOC[0] = alloc

    for _ in range(40):
        try:
            static, steps, hosts = build_chain(depth, rng, alloc)
            break
        except ValueError:
            continue
    else:
        return None

    vulns = {}
    ref_ids = []
    for i, (kind, pre, eff) in enumerate(steps, 1):
        vid = f"V-REF-{i}"
        vulns[vid] = {"pre": pre, "eff": eff, "role": "chain", "kind": kind}
        ref_ids.append(vid)

    idx = seed % 1000
    n_t4 = int(round(k * mix["T4"]))
    n_t2 = int(round(k * mix["T2"]))
    n_t3 = int(round(k * mix["T3"]))
    n_t1 = max(0, k - n_t4 - n_t2 - n_t3)

    families = [
        ("T1-inert", make_inert(n_t1, rng, alloc, idx)),
        ("T2-nearmiss", make_nearmiss(n_t2, rng, steps, alloc, idx)),
        ("T3-decoy", make_decoys(n_t3, rng, steps, idx)),
        ("T4-deadend", make_deadends(n_t4, branch_len, rng, steps, alloc, idx)),
    ]
    unsat_ids = []
    for role, items in families:
        for vid, pre, eff, family in items:
            vulns[vid] = {"pre": pre, "eff": eff, "role": role}
            if family == "unsat":
                unsat_ids.append(vid)

    scenario = {
        "id": f"d{depth}_k{k}_i{seed}",
        "depth": depth,
        "nominal_k": k,
        "distractors": len(vulns) - len(ref_ids),
        "init": sorted(set(static)),
        "goal": ["exfiltrated(asset_x)"],
        "vulns": vulns,
        "reference_chain": ref_ids,
    }

    # A distractor labelled non-applicable must really be unreachable. The
    # label is checked against the monotone closure rather than trusted, so a
    # T1 that some other distractor happens to enable causes the instance to
    # be redrawn instead of being reported under the wrong type.
    reach = set(scenario["init"])
    changed = True
    while changed:
        changed = False
        for v in vulns.values():
            if set(v["pre"]) <= reach and not set(v["eff"]) <= reach:
                reach |= set(v["eff"])
                changed = True
    for vid in unsat_ids:
        if set(vulns[vid]["pre"]) <= reach:
            return None

    # depth check: a distractor may have opened a shorter route to the goal,
    # which would silently make the instance easier than its label claims
    found, expansions = bfs_shortest(scenario)
    if found in (None, "ABORTED") or len(found) != depth:
        return None
    scenario["bfs_solution"] = found
    scenario["bfs_expansions"] = expansions

    n_min, exact = count_shortest_solutions(scenario, depth)
    scenario["minimal_chains"] = n_min
    scenario["minimal_chains_exact"] = exact

    mean_b, max_b = branching_along(scenario, ref_ids)
    scenario["branching_mean"] = round(mean_b, 2)
    scenario["branching_max"] = max_b

    counts = {}
    for v in vulns.values():
        counts[v["role"]] = counts.get(v["role"], 0) + 1
    scenario["type_counts"] = counts

    for v in scenario["vulns"].values():
        v.pop("kind", None)

    if identifiers == "neutral":
        scenario = anonymise(scenario, rng)
    else:
        scenario = keep_real(scenario, rng)
    return scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8,
                    help="independently generated instances per cell")
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--k", type=int, nargs="+", default=[20, 30, 40, 50, 60],
                    help="nominal distractor counts; intermediate levels are "
                         "worth having, the first study jumped from 20 to 40")
    ap.add_argument("--mix", default="0.25,0.30,0.15,0.30",
                    help="T1,T2,T3,T4 proportions")
    ap.add_argument("--branch-len", type=int, default=3)
    ap.add_argument("--identifiers", choices=["neutral", "real"],
                    default="neutral")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="scenarios.json")
    args = ap.parse_args()

    p = [float(x) for x in args.mix.split(",")]
    mix = {"T1": p[0], "T2": p[1], "T3": p[2], "T4": p[3]}

    scenarios = []
    rejected = 0
    seed = args.seed
    for depth in args.depths:
        for k in args.k:
            made = 0
            while made < args.instances:
                s = build_instance(depth, k, seed, mix, args.branch_len,
                                   args.identifiers)
                seed += 1
                if s is None:
                    rejected += 1
                    continue
                scenarios.append(s)
                made += 1

    with open(args.out, "w") as f:
        json.dump(scenarios, f, indent=1)

    print(f"{len(scenarios)} instances written to {args.out} "
          f"({rejected} draws rejected by the depth check)\n")
    print(f"{'cell':>12} {'n':>3} {'vulns':>6} {'branching':>10} "
          f"{'min chains':>11}  realised mix")
    cells = {}
    for s in scenarios:
        cells.setdefault((s["depth"], s["nominal_k"]), []).append(s)
    for (d, k), group in sorted(cells.items()):
        nv = sum(len(s["vulns"]) for s in group) / len(group)
        br = sum(s["branching_mean"] for s in group) / len(group)
        mc = sum(s["minimal_chains"] for s in group) / len(group)
        tc = {}
        for s in group:
            for r, c in s["type_counts"].items():
                tc[r] = tc.get(r, 0) + c
        tot = sum(c for r, c in tc.items() if r != "chain")
        mixs = " ".join(f"{r.split('-')[0]}={100*c/tot:.0f}%"
                        for r, c in sorted(tc.items()) if r != "chain")
        print(f"  d={d} k={k:<3} {len(group):>3} {nv:>6.0f} {br:>10.1f} "
              f"{mc:>11.1f}  {mixs}")


if __name__ == "__main__":
    main()
