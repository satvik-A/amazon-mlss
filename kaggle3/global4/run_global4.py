"""[account 3 · CPU · pass 4] Full-country level-1 scoring of a TRAIN country with the pass-4 matcher (every S1, like the
test set), then competition + sibling features (ber.compete) for the S1s of B and C (hash(11) % 100 >= 60).
Output feats_<country>.parquet [s1, r, y, p1, COMP, SIB] for rows with p1 > 0.0005 -> joined locally with the stack's
out-of-sample scores (stack_pred_B2 / C) to train the competition layer.
Inputs: er3-cands4-train-<country> (pool), er3-matcher4 dataset (matcher.txt, head.txt, decision.json), er-bundle."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"; CTRY = "__CTRY__"
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber import model as M
from ber.compete import comp_features, sib_features, COMP, SIB
T0 = time.time(); WD = "/kaggle/working"
REP = open(f"{WD}/results_global4_{CTRY}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
POOL = find(f"pool_train_{CTRY}.parquet")[0]; MD = os.path.dirname(find("decision.json")[0])
cfg = json.load(open(f"{MD}/decision.json")); m = lgb.Booster(model_file=f"{MD}/matcher.txt"); fc = cfg["features"]
log(f"ref {REF}; {CTRY}; pool {POOL}; policy {cfg['policy']}")
NZ = Normalizer(ART)
S = pl.read_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == CTRY)
st = M.claim_stats(POOL); nf = M.s1_name_freq(S, NZ)
SN = pl.concat([NZ.transform(S.slice(i, 1_000_000)) for i in range(0, S.height, 1_000_000)])
Rall = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == CTRY).collect()
RN = pl.concat([NZ.transform(Rall.slice(i, 1_000_000)) for i in range(0, Rall.height, 1_000_000)])
log(f"normalised {S.height} S1 + {Rall.height} records  {time.time()-T0:.0f}s")
NSH = 16; out = []
for k in range(NSH):
    pool = pl.scan_parquet(POOL).filter(pl.col("s1").hash(3) % NSH == k).collect(engine="streaming").drop("y", strict=False).join(st, on="r", how="left").join(nf, on="s1", how="left")
    F = M.pool_features(pool, None, None, NZ, QN=SN.filter(pl.col("entity_id").is_in(pool["s1"].unique().implode())),
                        RN=RN.filter(pl.col("entity_id").is_in(pool["r"].unique().implode())))
    del pool
    F = M.apply_policy(F, cfg["policy"])
    F = F.with_columns(pl.Series("p1", m.predict(M.X(F, fc))))
    out.append(F.select("s1", "r", "p1")); log(f"shard {k}: {F.height} rows  {time.time()-T0:.0f}s"); del F
D = pl.concat(out); del out, SN, RN
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S["entity_id"].implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
D = D.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
log(f"scored {D.height} rows ({D.height/S.height:.1f}/S1); positives in candidates {D['y'].sum()} of {gt.height} ({D['y'].sum()/gt.height:.4f})  {time.time()-T0:.0f}s")
D = comp_features(D)                                                        # over ALL S1s (competitors)
D = D.filter((pl.col("s1").hash(11) % 100 >= 60) & (pl.col("p1") > 0.0005))
D = sib_features(D, Rall)
D.select("s1", "r", "y", "p1", *COMP, *SIB).write_parquet(f"{WD}/feats_{CTRY}.parquet")
log(f"wrote feats_{CTRY}.parquet {D.height} rows (B + C S1s, p1 > 0.0005)  DONE {time.time()-T0:.0f}s")
