import polars as pl, re, unicodedata, collections
from rapidfuzz.distance import Levenshtein
o = pl.read_parquet("research/eda/cache/lookalike_sample.parquet").filter((pl.col("ns")>=80)&(pl.col("as_")>=80))
def nums(a): return [n.lstrip("0") or "0" for n in re.findall(r"\d+", a or "")]
def rel(x, y):
    if x == y: return "equal"
    if y.endswith(x) or x.endswith(y): return "suffix-trunc"
    if y.startswith(x) or x.startswith(y): return "prefix-trunc"
    if Levenshtein.distance(x, y) == 1: return "1-edit"
    if sorted(x) == sorted(y): return "digit-swap"
    try:
        d = abs(int(x)-int(y)); return f"delta<=25" if d <= 25 else "other"
    except: return "other"
def first_num(a):   # house number = first number token
    m = nums(a); return m[0] if m else None
res = collections.Counter(); ex = collections.defaultdict(list)
for r in o.iter_rows(named=True):
    x, y = first_num(r["a1"]), first_num(r["a2"])
    k = "R-missing" if y is None else ("S1-missing" if x is None else rel(x, y))
    res[(r["y"], k)] += 1
    if len(ex[(r["y"],k)]) < 3: ex[(r["y"],k)].append(f"{r['a1'][:60]}  ||  {r['a2'][:60]}")
tot = {y: sum(v for (yy,_),v in res.items() if yy==y) for y in (0,1)}
print("first-number relation (share within label) -- NOTE first number may be a unit/plot/order-shuffled")
for k in sorted({k for _,k in res}):
    print(f"  {k:14s} pos {res[(1,k)]/tot[1]:.3f}   neg {res[(0,k)]/tot[0]:.3f}")
for key in [(1,"delta<=25"),(1,"other"),(0,"suffix-trunc"),(0,"equal")]:
    print(key, *ex[key], sep="\n    ")
# number-SET relation: is there any shared number at all (order-free)?
def setrel(a1,a2):
    A, B = set(nums(a1)), set(nums(a2))
    if not B: return "R-no-numbers"
    if A == B: return "set-equal"
    if B <= A: return "R-subset"
    if A & B: return "overlap"
    return "disjoint"
c = collections.Counter((r["y"], setrel(r["a1"], r["a2"])) for r in o.iter_rows(named=True))
print("\nnumber-set relation:"); [print(f"  {k:12s} pos {c[(1,k)]/tot[1]:.3f}  neg {c[(0,k)]/tot[0]:.3f}") for k in ["set-equal","R-subset","overlap","disjoint","R-no-numbers"]]
# extra name tokens
LEG = set("inc llc ltd limited pvt private corp corporation co company lp llp pllc pc the and of group".split())
def toks(s):
    s = unicodedata.normalize("NFKD", s or ""); s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return set(re.findall(r"[a-z]+", s.lower()))
add = {0: collections.Counter(), 1: collections.Counter()}; nadd = collections.Counter()
for r in o.iter_rows(named=True):
    e = toks(r["n2"]) - toks(r["n1"]); nadd[(r["y"], min(len(e - LEG),3))] += 1
    for t in e: add[r["y"]][t] += 1
print("\n# non-legal extra tokens in R name (0/1/2/3+):", {y:[round(nadd[(y,k)]/tot[y],3) for k in range(4)] for y in (0,1)})
print("top extra tokens POS:", add[1].most_common(30)); print("top extra tokens NEG:", add[0].most_common(30))
