"""Quick decision-layer tests on saved held-out level-2 predictions (er-stack4: stack_pred_B2 / stack_pred_C).
Protocol: B2 (out-of-sample stack scores) is split by S1 hash into B2a (fit, test 1b only) and B2b (tune every threshold);
C reports. Decision = rank-threshold -> source caps -> one_owner. Gate: combined C gain >= +0.002.
  1. sibling rescue: anchors = accepted pairs with p >= 0.9; same-source candidates vs anchors (token_set_ratio on address / name,
     number-set equality). (a) rule p = max(p, 0.9) if addr >= 95 and numbers equal; (b) LightGBM on B2a with the sibling features.
  2. rank-threshold per {US, India} x {S2, S3}.
  3. conditioned top-1: rank-1 accepted at p >= t1_low when rev_margin > -m (the S1 is within m of the record's best S1), else t1.
  4. combination of the passes that helped on B2b.
Usage: research/guard.sh 10 .venv/bin/python research/quick4.py [in dir with stack_pred_*.parquet + level1_B/C.parquet]"""
import sys, re, itertools, time, glob, os, subprocess
REF = "__REF__"
if os.path.exists("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
else:
    sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from ber.metrics import macro_f05
from ber.decide import source_caps
from ber import model as M

T0 = time.time()
_f = lambda n: sorted(glob.glob(f"/kaggle/input/**/{n}", recursive=True))
if os.path.exists("/kaggle"):
    IN_SP, IN_L1, CACHE, OUT = os.path.dirname(_f("stack_pred_B2.parquet")[0]), os.path.dirname(_f("level1_B.parquet")[0]), os.path.dirname(_f("train_s1.parquet")[0]), "/kaggle/working"
else:
    IN_SP = IN_L1 = sys.argv[1] if len(sys.argv) > 1 else "kaggle/comp4/in"; CACHE, OUT = "research/eda/cache", "research"
say = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

# ---- data --------------------------------------------------------------------------------------------------------------
def load(nm, l1):
    d = pl.read_parquet(f"{IN_SP}/stack_pred_{nm}.parquet")
    rv = pl.read_parquet(f"{IN_L1}/level1_{l1}.parquet", columns=["s1", "r", "rev_margin"])
    return d.join(rv, on=["s1", "r"], how="left").rename({"p2": "p"})
B2, C = load("B2", "B"), load("C", "C")
S1 = pl.read_parquet(f"{CACHE}/train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
B2, C = (x.join(S1, on="s1", how="left").with_columns(pl.col("r").str.slice(0, 2).alias("src")) for x in (B2, C))
gt_all = pl.read_parquet(f"{CACHE}/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")) \
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "") \
           .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
hb = pl.col("s1").hash(29) % 2
B2a, B2b = B2.filter(hb == 0), B2.filter(hb == 1)
say(f"B2 {B2.height} rows / {B2['s1'].n_unique()} S1 (B2a {B2a['s1'].n_unique()}, B2b {B2b['s1'].n_unique()}); C {C.height} / {C['s1'].n_unique()} S1; "
    f"rev_margin joined {C['rev_margin'].is_not_null().mean():.3f}")

def F(sel, d, per_country=False):
    ids = d["s1"].unique(); m = macro_f05(sel.select("s1", "r"), gt_all.filter(pl.col("s1").is_in(ids.implode())), ids)
    out = dict(F=m["f05"], P=m["pair_precision"], R=m["pair_recall"])
    if per_country:
        pc = m["per"].join(S1, on="s1").group_by("country").agg(pl.col("f").mean()).sort("country")
        out.update({c: f for c, f in pc.iter_rows()})
    return out

# ---- decision: generic thresholds (column t1 / t2 per row) --------------------------------------------------------------
def decide(d, t1, t2, p="p", t1low=None, m=None):
    """t1/t2: floats or polars expressions (per row). t1low/m: conditioned top-1 (test 3)."""
    x = d.sort("r").with_columns(pl.col(p).rank("ordinal", descending=True).over("s1").alias("_rk"))
    T1 = t1 if isinstance(t1, pl.Expr) else pl.lit(t1)
    T2 = t2 if isinstance(t2, pl.Expr) else pl.lit(t2)
    if t1low is not None:
        T1 = pl.when(pl.col("rev_margin").fill_null(-9) > -m).then(pl.min_horizontal(pl.lit(t1low), T1)).otherwise(T1)
    sel = x.filter(((pl.col("_rk") == 1) & (pl.col(p) > T1)) | ((pl.col("_rk") > 1) & (pl.col(p) > T2))).select("s1", "r")
    sel = source_caps(sel, d.select("s1", "r", pl.col(p).alias("p")))
    return M.one_owner(sel.join(d.select("s1", "r", pl.col(p).alias("p")), on=["s1", "r"]), log=lambda *a: None)

GRID = [(a, b) for a in np.arange(0.30, 0.96, 0.05) for b in np.arange(0.30, 0.96, 0.05)]
def tune(d, p="p"):
    return max(((a, b, F(decide(d, a, b, p), d)["F"]) for a, b in GRID), key=lambda z: z[2])

rows = []
def report(name, selC, fB, cfg):
    r = F(selC, C, per_country=True); r.update(test=name, F_B2b=fB, cfg=cfg); rows.append(r)
    base = rows[0]["F"]
    say(f"{name:44s} C {r['F']:.4f} ({r['F']-base:+.4f})  P {r['P']:.4f} R {r['R']:.4f}  US {r.get('US', 0):.4f} India {r.get('India', 0):.4f}  | B2b {fB:.4f}  {cfg}")

# ---- baseline ------------------------------------------------------------------------------------------------------------
a0, b0, f0 = tune(B2b)
report("baseline: shared rank-threshold", decide(C, a0, b0), f0, dict(t1=round(a0, 2), t2=round(b0, 2)))

# ---- 1. sibling rescue ---------------------------------------------------------------------------------------------------
R = pl.concat([pl.read_parquet(f"{CACHE}/train_s{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]) \
      .rename({"entity_id": "r", "business_name": "nm", "business_address": "ad"})
NUM = re.compile(r"\d+")
def sib_feats(d, a1, b1):
    acc = decide(d, a1, b1).join(d.select("s1", "r", "p", "src"), on=["s1", "r"]).filter(pl.col("p") >= 0.9)
    anc = acc.join(R, on="r").select("s1", "src", pl.col("r").alias("ra"), pl.col("nm").fill_null("").alias("an"), pl.col("ad").fill_null("").alias("aa"))
    anc = anc.group_by("s1", "src").agg("ra", "an", "aa")
    x = d.join(R, on="r", how="left").join(anc, on=["s1", "src"], how="left")
    out = np.full((x.height, 4), np.nan, dtype=np.float32)
    for i, (r, nm, ad, ra, an, aa) in enumerate(x.select("r", "nm", "ad", "ra", "an", "aa").iter_rows()):
        if not ra:
            continue
        nm, ad = (nm or "").lower(), (ad or "").lower(); ns = set(NUM.findall(ad))
        bs = bn = -1.0; eq = 0; k = 0
        for r2, n2, a2 in zip(ra, an, aa):
            if r2 == r:
                continue
            k += 1; a2 = a2.lower()
            bs = max(bs, fuzz.token_set_ratio(ad, a2)); bn = max(bn, fuzz.token_set_ratio(nm, n2.lower()))
            eq |= int(bool(ns) and ns == set(NUM.findall(a2)))
        if k:
            out[i] = (bs, bn, eq, k)
    return x.drop("nm", "ad", "ra", "an", "aa").with_columns(*[pl.Series(c, out[:, j]) for j, c in enumerate(["sib_addr_sim", "sib_name_sim", "sib_num_eq", "n_anchors"])])
say("sibling features ...")
B2a, B2b, C = (sib_feats(x, a0, b0) for x in (B2a, B2b, C))
say(f"sibling features done; C rows with an anchor of the same source: {C['n_anchors'].is_not_nan().mean():.3f}")
rule = lambda d: d.with_columns(pl.when((pl.col("sib_addr_sim") >= 95) & (pl.col("sib_num_eq") == 1)).then(pl.max_horizontal("p", pl.lit(0.9))).otherwise(pl.col("p")).alias("p_rule"))
B2b, C = rule(B2b), rule(C)
n_raised = C.filter(pl.col("p_rule") > pl.col("p")); say(f"rule raises {n_raised.height} C rows, of which true {n_raised['y'].sum()}")
a1, b1, f1 = tune(B2b, "p_rule")
report("1a sibling rule (addr>=95 & numbers equal)", decide(C, a1, b1, "p_rule"), f1, dict(t1=round(a1, 2), t2=round(b1, 2)))
feats = ["p", "p1", "noaddr_r", "sib_addr_sim", "sib_name_sim", "sib_num_eq", "n_anchors"]
gbm = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, verbose=-1, num_threads=8),
                lgb.Dataset(M.X(B2a, feats), B2a["y"].to_numpy()), 400)
B2b, C = (x.with_columns(pl.Series("p_sib", gbm.predict(M.X(x, feats)))) for x in (B2b, C))
a2, b2, f2 = tune(B2b, "p_sib")
report("1b sibling LightGBM (B2a)", decide(C, a2, b2, "p_sib"), f2, dict(t1=round(a2, 2), t2=round(b2, 2)))
conf = sel = decide(C, a2, b2, "p_sib"); say(f"one-owner check: {sel.height} pairs, records with 2+ S1s {sel.group_by('r').len().filter(pl.col('len') > 1).height}")

# ---- 2. per country x source thresholds (coordinate ascent from the shared pair) ------------------------------------------
def grp_thr(tab):
    e1 = e2 = None
    for (c, s), (t1, t2) in tab.items():
        cond = (pl.col("country") == c) & (pl.col("src") == s)
        e1 = (pl.when(cond).then(t1) if e1 is None else e1.when(cond).then(t1))
        e2 = (pl.when(cond).then(t2) if e2 is None else e2.when(cond).then(t2))
    return e1.otherwise(a0), e2.otherwise(b0)
def tune_groups(d, p="p", base=(a0, b0)):
    tab = {(c, s): base for c in ("US", "India") for s in ("S2", "S3")}
    for _ in range(2):
        for g in list(tab):
            ga = [(a, b) for a in np.arange(tab[g][0] - 0.2, tab[g][0] + 0.21, 0.05) for b in np.arange(tab[g][1] - 0.2, tab[g][1] + 0.21, 0.05) if 0.05 < a < 0.99 and 0.05 < b < 0.99]
            best = max(((a, b, F(decide(d, *grp_thr({**tab, g: (a, b)}), p), d)["F"]) for a, b in ga), key=lambda z: z[2])
            tab[g] = (best[0], best[1])
    return tab, F(decide(d, *grp_thr(tab), p), d)["F"]
say("per country x source tuning ...")
tab, f3 = tune_groups(B2b)
report("2  rank-threshold per country x source", decide(C, *grp_thr(tab)), f3, {f"{c}-{s}": (round(a, 2), round(b, 2)) for (c, s), (a, b) in tab.items()})

# ---- 3. conditioned top-1 --------------------------------------------------------------------------------------------------
best3 = max(((tl, m, F(decide(B2b, a0, b0, t1low=tl, m=m), B2b)["F"]) for tl in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75) for m in (0.2, 0.35, 0.5)), key=lambda z: z[2])
report("3  conditioned top-1 (rev_margin > -m)", decide(C, a0, b0, t1low=best3[0], m=best3[1]), best3[2], dict(t1_low=best3[0], m=best3[1], t1=round(a0, 2)))

# ---- 4. combination of whatever helped on B2b ---------------------------------------------------------------------------
fb = {r["test"]: r["F_B2b"] for r in rows}; base_b = fb["baseline: shared rank-threshold"]
use_sib = max(fb["1a sibling rule (addr>=95 & numbers equal)"], fb["1b sibling LightGBM (B2a)"]) > base_b + 1e-4
pcol = ("p_sib" if fb["1b sibling LightGBM (B2a)"] >= fb["1a sibling rule (addr>=95 & numbers equal)"] else "p_rule") if use_sib else "p"
use_grp = fb["2  rank-threshold per country x source"] > base_b + 1e-4
use_top = fb["3  conditioned top-1 (rev_margin > -m)"] > base_b + 1e-4
say(f"combination: score {pcol}, per-group thresholds {use_grp}, conditioned top-1 {use_top}")
if use_grp:
    tab4, _ = tune_groups(B2b, pcol, base=tune(B2b, pcol)[:2]); t1e, t2e = grp_thr(tab4)
else:
    a4, b4, _ = tune(B2b, pcol); t1e, t2e = a4, b4
if use_top:
    b3 = max(((tl, m, F(decide(B2b, t1e, t2e, pcol, tl, m), B2b)["F"]) for tl in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75) for m in (0.2, 0.35, 0.5)), key=lambda z: z[2])
    selC, f4 = decide(C, t1e, t2e, pcol, b3[0], b3[1]), b3[2]
else:
    selC, f4 = decide(C, t1e, t2e, pcol), F(decide(B2b, t1e, t2e, pcol), B2b)["F"]
report("4  combined", selC, f4, dict(score=pcol, groups=use_grp, top1=use_top))
d = rows[-1]["F"] - rows[0]["F"]
say(f"COMBINED gain on C {d:+.4f} -> {'ACCEPT' if d >= 0.002 else 'REJECT (gate +0.002)'}")
pl.DataFrame([{k: (str(v) if k == "cfg" else v) for k, v in r.items()} for r in rows]).write_csv(f"{OUT}/quick4_results.csv")
