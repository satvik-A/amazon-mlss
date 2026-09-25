"""Follow-up: does context-aware calibration (rank-conditional probs + entity-level P(no match))
fix expected-F decoding? Reuses the world generator from decode_sim.py."""
import numpy as np, itertools, importlib.util, sys
spec = importlib.util.spec_from_file_location('d', 'decode_sim.py'); src = open('decode_sim.py').read()
src = src.split("count_p = [0.35")[0]; d = {}; exec(src, d)
make_world, pav_fit, f05, pb = d['make_world'], d['pav_fit'], d['f05'], d['pb']

def ranks(s): r = np.empty(len(s), int); r[np.argsort(-s)] = np.arange(len(s)); return r

def fit_ctx(ents):
    S = np.concatenate([e[3] for e in ents]); L = np.concatenate([e[2] for e in ents])
    R = np.concatenate([np.minimum(ranks(e[3]), 2) for e in ents])
    # margin feature: score minus best *other* score in the entity (competition)
    M = np.concatenate([e[3] - np.array([np.max(np.delete(e[3], i)) if len(e[3]) > 1 else -9 for i in range(len(e[3]))]) for e in ents])
    pooled = pav_fit(S, L)
    by_rank = {r: pav_fit(S[R == r], L[R == r]) for r in range(3)}
    top = np.array([e[3].max() if len(e[3]) else -9 for e in ents]); has = np.array([len(e[0]) > 0 for e in ents])
    p_has = pav_fit(top, has)
    return pooled, by_rank, p_has

def probs_rank(s, by_rank):
    r = np.minimum(ranks(s), 2); return np.array([by_rank[ri](np.array([si]))[0] for si, ri in zip(s, r)])

def ef(p, p0=None):
    o = np.argsort(-p); ps = p[o]; n = len(ps)
    if n == 0: return []
    full = pb(ps); p_none = full[0]
    best_k, best = 0, (p0 if p0 is not None else p_none)
    for k in range(1, n+1):
        A, B = pb(ps[:k]), pb(ps[k:])
        a = np.arange(len(A))[:, None]; b = np.arange(len(B))[None, :]
        F = np.where(a > 0, 1.25*a/(0.25*(a+b)+k), 0.0)
        e = (A[:, None]*B[None, :]*F).sum()
        if p0 is not None:   # condition DP on "at least one true", weight by entity head
            e = (1 - p0) * e / max(1 - p_none, 1e-6)
        if e > best: best, best_k = e, k
    return list(o[:best_k])

def rank2(p, t1, t2):
    o = np.argsort(-p)
    if len(p) == 0 or p[o[0]] <= t1: return []
    return [o[0]] + [i for i in o[1:] if p[i] > t2]

def run(name, **kw):
    cal, ev = make_world(4000, **kw), make_world(4000, **kw)
    pooled, by_rank, p_has = fit_ctx(cal)
    def sc(ents, dec): return np.mean([f05({int(c[i]) for i in dec(s)}, t) for t, c, _, s in ents])
    grid = np.round(np.arange(0.2, 0.96, 0.05), 3)
    rk = lambda s: probs_rank(s, by_rank)
    t1, t2 = max(itertools.product(grid, grid), key=lambda tt: sc(cal, lambda s: rank2(rk(s), *tt)))
    res = {
        'EF-DP, pooled calibration':            sc(ev, lambda s: ef(pooled(s))),
        'EF-DP, rank-conditional calibration':  sc(ev, lambda s: ef(rk(s))),
        'EF-DP, rank-cond + entity has-match head': sc(ev, lambda s: ef(rk(s), 1 - p_has(np.array([s.max()]))[0] if len(s) else None)),
        f'rank thresholds on rank-cond probs (t1={t1},t2={t2})': sc(ev, lambda s: rank2(rk(s), t1, t2)),
    }
    print(f'\n== {name} ==')
    for k, v in res.items(): print(f'  {k:55s} {v:.4f}')

cp = [0.35, 0.40, 0.15, 0.07, 0.03]
run('easy',   mu_p=3.0, mu_n=-3.5, mu_h=0.0, count_p=cp)
run('medium', mu_p=2.0, mu_n=-3.0, mu_h=1.0, count_p=cp)
run('hard',   mu_p=1.5, mu_n=-2.5, mu_h=1.3, count_p=cp, chain_rate=0.5)
