"""Simulate per-entity macro F0.5 decoding strategies on a synthetic ER world.
Labels are generated first (with within-entity correlation), scores come from a noisy
classifier, probabilities from isotonic calibration on a held-out half -> honest test of
whether expected-F decoding (which assumes independence) beats tuned thresholds."""
import numpy as np, itertools
rng = np.random.default_rng(7)

def f05(pred, truth):
    if not truth and not pred: return 1.0
    if not truth or not pred: return 0.0
    tp = len(pred & truth)
    if tp == 0: return 0.0
    P, R = tp/len(pred), tp/len(truth)
    return 1.25*P*R/(0.25*P+R)

def make_world(n_ent, mu_p, mu_n, mu_h, count_p, block_recall=0.95, lam_neg=10, chain_rate=0.3):
    ents = []
    for i in range(n_ent):
        k = rng.choice(len(count_p), p=count_p)
        d = rng.normal(0, 0.7)                       # entity-level difficulty (breaks independence)
        truth = set(range(k))
        cands, labels, scores = [], [], []
        for j in range(k):
            if rng.random() < block_recall:
                cands.append(j); labels.append(1); scores.append(rng.normal(mu_p + d, 1.2))
        nn = rng.poisson(lam_neg); nh = rng.poisson(3) if rng.random() < chain_rate else 0
        for j in range(nn):
            cands.append(100+j); labels.append(0); scores.append(rng.normal(mu_n + d, 1.5))
        for j in range(nh):
            cands.append(200+j); labels.append(0); scores.append(rng.normal(mu_h + d, 1.2))
        ents.append((truth, np.array(cands), np.array(labels), np.array(scores)))
    return ents

def pav_fit(x, y):
    o = np.argsort(x); x, y = x[o], y[o].astype(float)
    vals, wts, right = [], [], []
    for xi, yi in zip(x, y):
        vals.append(yi); wts.append(1.0); right.append(xi)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            v2, w2, r2 = vals.pop(), wts.pop(), right.pop()
            v1, w1, _ = vals.pop(), wts.pop(), right.pop()
            vals.append((v1*w1+v2*w2)/(w1+w2)); wts.append(w1+w2); right.append(r2)
    right, vals = np.array(right), np.clip(np.array(vals), 1e-4, 1-1e-4)
    return lambda s: vals[np.minimum(np.searchsorted(right, s), len(vals)-1)]

def pb(ps):                                           # Poisson-binomial pmf
    d = np.array([1.0])
    for p in ps: d = np.convolve(d, [1-p, p])
    return d

def ef_decode(p):
    o = np.argsort(-p); ps = p[o]; n = len(ps)
    best_k, best = 0, np.prod(1-ps)                   # E[F(empty)] = P(no true among candidates)
    for k in range(1, n+1):
        A, B = pb(ps[:k]), pb(ps[k:])
        a = np.arange(len(A))[:, None]; b = np.arange(len(B))[None, :]
        F = np.where(a > 0, 1.25*a/(0.25*(a+b)+k), 0.0)
        e = (A[:, None]*B[None, :]*F).sum()
        if e > best: best, best_k = e, k
    return set(o[:best_k].tolist())

def run(name, **kw):
    count_p = kw.pop('count_p')
    cal, ev = make_world(6000, count_p=count_p, **kw), make_world(6000, count_p=count_p, **kw)
    iso = pav_fit(np.concatenate([e[3] for e in cal]), np.concatenate([e[2] for e in cal]))
    def probs(ents): return [(t, c, iso(s)) for t, c, _, s in ents]
    C, E = probs(cal), probs(ev)
    def score(ents, dec): return np.mean([f05({int(c[i]) for i in dec(p)}, t) for t, c, p in ents])
    glob = lambda t: (lambda p: np.where(p > t)[0])
    def rank2(t1, t2):
        def d(p):
            o = np.argsort(-p)
            if len(p) == 0 or p[o[0]] <= t1: return []
            return [o[0]] + [i for i in o[1:] if p[i] > t2]
        return d
    grid = np.round(np.arange(0.2, 0.96, 0.025), 3)
    tg = max(grid, key=lambda t: score(C, glob(t)))
    t1, t2 = max(itertools.product(grid, grid), key=lambda tt: score(C, rank2(*tt)))
    ef = lambda p: list(np.where(np.isin(np.arange(len(p)), list(ef_decode(p))))[0]) if len(p) else []
    oracle = np.mean([f05({int(x) for x in c[l == 1]} if True else set(), t) for t, c, l, _ in ev])
    res = {'global t=0.5': score(E, glob(0.5)), f'global tuned (t={tg})': score(E, glob(tg)),
           f'rank-dependent (t1={t1}, t2={t2})': score(E, rank2(t1, t2)), 'expected-F DP (untuned)': score(E, ef),
           'oracle (blocking ceiling)': oracle}
    print(f'\n== {name} ==')
    for k, v in res.items(): print(f'  {k:40s} {v:.4f}')

count_p = [0.35, 0.40, 0.15, 0.07, 0.03]
run('easy   (clean separation)', mu_p=3.0, mu_n=-3.5, mu_h=0.0, count_p=count_p)
run('medium (chains overlap)',   mu_p=2.0, mu_n=-3.0, mu_h=1.0, count_p=count_p)
run('hard   (noisy, many chains)', mu_p=1.5, mu_n=-2.5, mu_h=1.3, count_p=count_p, chain_rate=0.5)
run('medium, 60% singletons',    mu_p=2.0, mu_n=-3.0, mu_h=1.0, count_p=[0.6, 0.25, 0.1, 0.04, 0.01])

print('\n== Marginal threshold to ADD one more candidate, given m certain-true already predicted ==')
for m in range(0, 5):
    # truth = m known trues (+ candidate if true). predict m vs m+1.
    if m == 0:
        print('  m=0 (first pick, entity otherwise singleton): p > 0.500'); continue
    fa = lambda P, R: 1.25*P*R/(0.25*P+R)
    add_T, add_F = 1.0, fa(m/(m+1), 1.0)
    no_T, no_F = fa(1.0, m/(m+1)), 1.0
    p = (no_F - add_F)/((add_T - add_F) - (no_T - no_F))
    print(f'  m={m}: add iff p > {p:.3f}   (wrong add: 1.00->{add_F:.3f}, skip true: 1.00->{no_T:.3f})')
