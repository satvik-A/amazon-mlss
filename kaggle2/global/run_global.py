"""[account 2 · CPU · internet] Global (transductive) assignment on a FULL train country.
Held-out F0.5 so far scores each S1 alone on a sample, so it cannot see competition for records: a no-address copy named
"Star Agro Ltd" is ~8% likely for each of a dozen "Star Agro" S1s, and each of them rejects it, although the record
almost always belongs to one of them. Here every S1 of the country is scored (like the test set), then decision rules
that let S1s compete for records are compared with labels on held-out S1s (tuned on B, reported on C).
Inputs: er2-cands2-train-<country> (pool), er2-matcher-full2 (matcher.txt, head.txt, decision.json), er-bundle."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
CTRY = "__CTRY__"
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", os.path.dirname(find("polars-1.44.2*.whl")[0]),
                "polars==1.44.2", "rapidfuzz==3.14.6", "lightgbm==4.6.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.metrics import macro_f05
from ber.decide import rank_threshold, source_caps
from ber import model as M
T0 = time.time(); WD = "/kaggle/working"
REP = open(f"{WD}/results_global_{CTRY}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
POOL = find(f"pool_train_{CTRY}.parquet")[0]; MD = os.path.dirname(find("decision.json")[0])
cfg = json.load(open(f"{MD}/decision.json")); m = lgb.Booster(model_file=f"{MD}/matcher.txt"); fc = cfg["features"]
log(f"ref {REF}; {CTRY}; pool {POOL}; model {cfg.get('ref')} rule {cfg['rule']} t1 {cfg['t1']:.2f} t2 {cfg['t2']:.2f} policy {cfg['policy']}")
NZ = Normalizer(ART)

# ---- 1. score every S1 of the country (same path as the submission job) ------------------------------------------
S = pl.read_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == CTRY)
st = M.claim_stats(POOL); nf = M.s1_name_freq(S, NZ)
SN = pl.concat([NZ.transform(S.slice(i, 1_000_000)) for i in range(0, S.height, 1_000_000)])
Rall = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == CTRY).collect()
RN = pl.concat([NZ.transform(Rall.slice(i, 1_000_000)) for i in range(0, Rall.height, 1_000_000)])
log(f"normalised {S.height} S1 + {Rall.height} records  {time.time()-T0:.0f}s")
NSH = 10; out = []
for k in range(NSH):
    pool = pl.scan_parquet(POOL).filter(pl.col("s1").hash(3) % NSH == k).collect(engine="streaming").drop("y", strict=False).join(st, on="r", how="left").join(nf, on="s1", how="left")
    ids = pool["s1"].unique()
    F = M.pool_features(pool, None, None, NZ, QN=SN.filter(pl.col("entity_id").is_in(ids.implode())),
                        RN=RN.filter(pl.col("entity_id").is_in(pool["r"].unique().implode())))
    del pool
    F = F.filter(M.cutoff_mask(int(cfg["policy"]["cutoff"])))
    F = F.with_columns(pl.Series("p", m.predict(M.X(F, fc))))
    out.append(F.select("s1", "r", "p", "noaddr_r"))
    log(f"shard {k}: {F.height} rows  {time.time()-T0:.0f}s")
    del F
D = pl.concat(out); del out, SN, RN
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S["entity_id"].implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
D = D.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
D.write_parquet(f"{WD}/scored_{CTRY}.parquet")
log(f"scored {D.height} rows ({D.height/S.height:.1f}/S1); positives in candidates {D['y'].sum()} of {gt.height}  {time.time()-T0:.0f}s")

# ---- 2. decision rules with competition, tuned on B, reported on C --------------------------------------------------
sB = S.filter(pl.col("entity_id").hash(11) % 100 >= 60).filter(pl.col("entity_id").hash(11) % 100 < 90)["entity_id"]
sB = sB.sample(min(60_000, len(sB)), seed=5)                      # tuning subset (speed)
sC = S.filter(pl.col("entity_id").hash(11) % 100 >= 90)["entity_id"]
gB = gt.filter(pl.col("s1").is_in(sB.implode())); gC = gt.filter(pl.col("s1").is_in(sC.implode()))
def f05(sel, s_ids, g):
    return macro_f05(sel.filter(pl.col("s1").is_in(s_ids.implode())).select("s1", "r"), g, s_ids)
def decide(d, col, t1, t2):
    x = d.with_columns(pl.col(col).alias("_q"))
    return source_caps(rank_threshold(x, t1, t2, p="_q"), x, p="_q").join(x.select("s1", "r", "_q"), on=["s1", "r"]).rename({"_q": "p"})
def one_owner(sel):
    return sel.sort(["r", "p", "s1"], descending=[False, True, False]).unique(subset=["r"], keep="first", maintain_order=True)
e = 1e-6
odds = pl.col("p").clip(0.0, 1 - e) / (1 - pl.col("p")).clip(e, 1.0)
variants = {"p": D}
for kk in (0.05, 0.2, 0.5, 1.0, 2.0):             # q = odds_s / (k + sum_u odds_u): k = prior weight of "no owner"
    variants[f"q_k{kk}"] = D.with_columns((odds / (kk + odds.sum().over("r"))).alias("p"))
grid = [(a, b) for a in np.arange(0.3, 0.96, 0.05) for b in np.arange(0.3, 0.96, 0.05)]
res = []
def closure(s_ids):
    """S1s evaluated + every S1 that competes with them for a record (full candidate lists: rank_threshold ranks within S1)"""
    rs = D.filter(pl.col("s1").is_in(s_ids.implode()))["r"].unique()
    return D.filter(pl.col("r").is_in(rs.implode()))["s1"].unique()
cB, cC = closure(sB), closure(sC)
log(f"tuning closure: B {len(sB)} S1 -> {len(cB)} with competitors; C {len(sC)} -> {len(cC)}")
for name, V in variants.items():
    VB, VC = V.filter(pl.col("s1").is_in(cB.implode())), V.filter(pl.col("s1").is_in(cC.implode()))
    for owner in (False, True):
        def score(t1, t2, s_ids, g):
            full = VB if s_ids is sB else VC
            sel = decide(full, "p", t1, t2)
            if owner: sel = one_owner(sel)
            return f05(sel, s_ids, g)
        if name == "p" and not owner:
            cand = [(cfg["t1"], cfg["t2"])]           # the shipped rule as the baseline
        else:
            coarse = [(a, b) for a, b in grid if round(a * 20) % 2 == 0 and round(b * 20) % 2 == 0]
            cand = coarse
        best = max(((a, b, score(a, b, sB, gB)["f05"]) for a, b in cand), key=lambda x: x[2])
        a, b = best[0], best[1]
        if len(cand) > 1:                              # local refinement
            best = max(((x, y, score(x, y, sB, gB)["f05"]) for x in (a - 0.05, a, a + 0.05) for y in (b - 0.05, b, b + 0.05) if 0.05 < x < 0.99 and 0.05 < y < 0.99), key=lambda z: z[2])
        mC = score(best[0], best[1], sC, gC)
        res.append(dict(variant=name, one_owner=owner, t1=round(best[0], 2), t2=round(best[1], 2), F_B=round(best[2], 5), F_C=round(mC["f05"], 5),
                        singleton_C=round(mC["singleton_f"], 5), P_C=round(mC["pair_precision"], 5), R_C=round(mC["pair_recall"], 5)))
        log(json.dumps(res[-1]) + f"  {time.time()-T0:.0f}s")
R_ = pl.DataFrame(res)
log(R_.sort("F_B", descending=True))
b0 = R_.filter((pl.col("variant") == "p") & ~pl.col("one_owner"))["F_C"][0]
bb = R_.sort("F_B", descending=True).row(0, named=True)
log(f"BASELINE (shipped rule, per S1) C {b0:.5f}; BEST on B: {bb['variant']} one_owner={bb['one_owner']} -> C {bb['F_C']:.5f} ({bb['F_C']-b0:+.5f})")
R_.write_csv(f"{WD}/global_{CTRY}.csv")
log(f"DONE {time.time()-T0:.0f}s")
