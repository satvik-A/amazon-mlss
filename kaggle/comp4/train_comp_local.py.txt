"""[local] Competition + sibling layer (level 3) for pass 4. Inputs (kaggle/comp4/in/): stack_pred_B2/C.parquet (er-stack4:
out-of-sample stacked p2, level-1 p1, noaddr_r, y) + feats_<country>.parquet (er3-global4-*: COMP + SIB from full-country
level-1 scores). Trains on half of B2, tunes the rank-threshold on the other half, reports on C vs the stack alone.
Writes kaggle/comp4/out/comp.txt + comp.json (gate: C delta >= +0.001)."""
import glob, json, os, sys
sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, polars as pl, lightgbm as lgb
from ber.metrics import macro_f05
from ber import model as M
from ber.compete import COMP, SIB
I, O = "kaggle/comp4/in", "kaggle/comp4/out"; os.makedirs(O, exist_ok=True)
F = pl.concat([pl.read_parquet(f) for f in glob.glob(f"{I}/feats_*.parquet")]).drop("y", "p1")
def load(nm):
    d = pl.read_parquet(f"{I}/stack_pred_{nm}.parquet").join(F, on=["s1", "r"], how="left")
    print(nm, d.height, "rows; with competition features", d["n_r"].is_not_null().mean())
    return d
B2, C = load("B2"), load("C")
gt_all = pl.read_parquet("research/eda/cache/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")) \
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
ORPH = pl.read_parquet("research/eda/cache/train_s2.parquet", columns=["entity_id"]).vstack(pl.read_parquet("research/eda/cache/train_s3.parquet", columns=["entity_id"])) \
         .filter(~pl.col("entity_id").is_in(gt_all["r"].implode())).rename({"entity_id": "r"})
W_ORPH = float(os.environ.get("W_ORPH", 2.0))   # test has ~2x the orphan records per S1 of train (label-free count, 2026-09-27)
def f05(sel, d, w=1.0):
    """macro F0.5; w > 1 counts a false positive on an orphan record w times (test-like orphan density)."""
    ids = d["s1"].unique(); g = gt_all.filter(pl.col("s1").is_in(ids.implode()))
    m = macro_f05(sel.select("s1", "r"), g, ids)
    if w == 1.0: return m
    fo = sel.join(ORPH, on="r", how="semi").group_by("s1").len().rename({"len": "fo"})
    per = m["per"].join(fo, on="s1", how="left").with_columns(pl.col("fo").fill_null(0))
    npw = pl.col("np") + (w - 1) * pl.col("fo")
    f = pl.when((pl.col("nt") == 0) & (npw == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0).otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + npw))
    return dict(m, f05=per.select(f.mean()).item())
def dec(d, col, t1, t2):
    x = d.with_columns(pl.col(col).alias("p"))
    sel = M.decide(x, dict(rule="rank-threshold", t1=t1, t2=t2), None).join(x.select("s1", "r", "p"), on=["s1", "r"])
    return M.one_owner(sel, log=lambda *a: None)
def tune(d, col, w=1.0):
    g = [(a, b) for a in np.arange(0.3, 0.96, 0.05) for b in np.arange(0.3, 0.96, 0.05)]
    return max(((a, b, f05(dec(d, col, a, b), d, w)["f05"]) for a, b in g), key=lambda z: z[2])
h = pl.col("s1").hash(17) % 2
Tr, Tu = B2.filter(h == 0), B2.filter(h == 1)
feats = ["p2", "p1", "noaddr_r", *COMP, *SIB]
m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8,
                   bagging_freq=1, verbose=-1, num_threads=8), lgb.Dataset(M.X(Tr, feats), Tr["y"].to_numpy()), 2000,
              valid_sets=[lgb.Dataset(M.X(Tu, feats), Tu["y"].to_numpy())], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
print("gain", sorted(zip(m.feature_importance("gain").round(), feats), reverse=True)[:10])
Tu = Tu.with_columns(pl.Series("p3", m.predict(M.X(Tu, feats)))); C = C.with_columns(pl.Series("p3", m.predict(M.X(C, feats))))
res = {}
for col in ("p2", "p3"):
    for w in (1.0, W_ORPH):
        a, b, fb = tune(Tu, col, w); sel = dec(C, col, a, b); r = f05(sel, C)
        k = col if w == 1.0 else f"{col}_w"
        res[k] = dict(t1=float(a), t2=float(b), F_tune=fb, C=r["f05"], C_w=f05(sel, C, W_ORPH)["f05"], P=r["pair_precision"], R=r["pair_recall"])
        print(k, res[k])
json.dump(dict(t1=res["p2_w"]["t1"], t2=res["p2_w"]["t2"], t1_unweighted=res["p2"]["t1"], t2_unweighted=res["p2"]["t2"], w_orph=W_ORPH),
          open(f"{O}/stack_thr_w.json", "w"), indent=1)
d = res["p3"]["C"] - res["p2"]["C"]
print(f"DELTA competition layer minus stack on C: {d:+.5f} -> {'ACCEPT' if d >= 0.001 else 'REJECT'}")
m.save_model(f"{O}/comp.txt")
json.dump(dict(features=feats, t1=res["p3_w"]["t1"], t2=res["p3_w"]["t2"], t1_unweighted=res["p3"]["t1"], t2_unweighted=res["p3"]["t2"], w_orph=W_ORPH, C=res["p3"]["C"], C_stack=res["p2"]["C"], delta_C=d), open(f"{O}/comp.json", "w"), indent=1)
