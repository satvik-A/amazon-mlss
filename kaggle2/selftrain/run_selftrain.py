"""[account 2 · CPU · internet] Does self-training help a country WITHOUT labels (France)? Simulated with US as the
"unlabelled" country: teacher = matcher trained on India only; it pseudo-labels US (non-C S1s); a student is trained on
India + US pseudo-labels; thresholds are always tuned on India B (as they would be for France). Everything is scored on
US held-out C (true labels). Reference: in-domain US model (the ceiling self-training could reach).
Inputs: er2-cands2-train-us, er2-cands2-train-india (pass-2 pools), er-bundle."""
import glob, os, subprocess, sys, time
REF = "__REF__"
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", os.path.dirname(find("polars-1.44.2*.whl")[0]),
                "polars==1.44.2", "rapidfuzz==3.14.6", "lightgbm==4.6.0"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.metrics import macro_f05
from ber.decide import rank_threshold
from ber import model as M

T0 = time.time(); WD = "/kaggle/working"; N_S1 = 60_000
REP = open(f"{WD}/results_selftrain.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
NZ = Normalizer(ART); S1ALL = pl.read_parquet(f"{IN}/train_s1.parquet")

def load(ctry):
    p = find(f"pool_train_{ctry}.parquet")[0]; st = M.claim_stats(p)
    ids = pl.scan_parquet(p).select(pl.col("s1").unique()).collect(engine="streaming")["s1"]
    thr = int(1_000_000 * min(1.0, N_S1 / len(ids)))
    P = pl.scan_parquet(p).filter((pl.col("s1").hash(7) % 1_000_000) < thr).collect(engine="streaming").drop("y", strict=False).join(st, on="r", how="left")
    P = P.join(M.s1_name_freq(S1ALL.filter(pl.col("country") == ctry), NZ), on="s1", how="left")
    S1ids = P["s1"].unique()
    s1 = S1ALL.filter(pl.col("entity_id").is_in(S1ids.implode()))
    R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("entity_id").is_in(P["r"].unique().implode())).collect()
    gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S1ids.implode())) \
           .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
           .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
    P = P.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    F = M.pool_features(P, s1, R, NZ).with_columns((pl.col("s1").hash(11) % 100).alias("h"))
    log(f"{ctry}: {S1ids.len()} S1, {F.height} rows, positives {F['y'].sum()}  {time.time()-T0:.0f}s")
    return F, gt

FI, gI = load("India"); FU, gU = load("US")
fc = M.feature_columns(FI)
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
IA, IB = FI.filter(pl.col("h") < 60), FI.filter((pl.col("h") >= 60) & (pl.col("h") < 90))
UA, UB, UC = FU.filter(pl.col("h") < 60), FU.filter((pl.col("h") >= 60) & (pl.col("h") < 90)), FU.filter(pl.col("h") >= 90)
UN = FU.filter(pl.col("h") < 90)   # "unlabelled" US rows usable for self-training (C never touched)
sub = lambda g_, d: g_.filter(pl.col("s1").is_in(d["s1"].unique().implode()))
gIB, gUB, gUC = sub(gI, IB), sub(gU, UB), sub(gU, UC)

def fit(X, y, w=None, rounds=1500):
    return lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=fc), rounds)

def tune(Bv, gB):
    sB = Bv["s1"].unique(); ev = lambda a, b: macro_f05(rank_threshold(Bv, a, b), gB, sB)["f05"]
    t1, t2, _ = max(((a, b, ev(a, b)) for a in np.arange(0.3, 0.96, 0.1) for b in np.arange(0.3, 0.96, 0.1)), key=lambda x: x[2])
    t1, t2, _ = max(((a, b, ev(a, b)) for a in (t1 - 0.05, t1, t1 + 0.05) for b in (t2 - 0.05, t2, t2 + 0.05) if 0.05 < a < 0.99 and 0.05 < b < 0.99), key=lambda x: x[2])
    return t1, t2

def report(name, m, tB="India"):
    Bv = (IB if tB == "India" else UB).with_columns(pl.Series("p", m.predict(M.X(IB if tB == "India" else UB, fc))))
    t1, t2 = tune(Bv, gIB if tB == "India" else gUB)
    C = UC.with_columns(pl.Series("p", m.predict(M.X(UC, fc))))
    mm = macro_f05(rank_threshold(C, t1, t2), gUC, UC["s1"].unique())
    log(f"  {name:48s} US C F0.5 {mm['f05']:.4f}  P {mm['pair_precision']:.4f} R {mm['pair_recall']:.4f}  (t1 {t1:.2f} t2 {t2:.2f} tuned on {tB} B)")
    return C

log("=== reference ===")
mI = fit(M.X(IA, fc), IA["y"].to_numpy()); log(f"teacher (India A) trained {time.time()-T0:.0f}s")
report("transfer: India-only model", mI)
report("transfer: India-only model, thresholds on US B (oracle thr)", mI, "US")
mU = fit(M.X(UA, fc), UA["y"].to_numpy())
report("in-domain: US A model, US B thresholds (ceiling)", mU, "US")

log("=== self-training (US pseudo-labels from the India teacher, thresholds on India B) ===")
pU = UN.with_columns(pl.Series("p", mI.predict(M.X(UN, fc))))
acc = lambda d: (d["y"] == d["yp"]).mean()
XI, yI = M.X(IA, fc), IA["y"].to_numpy()
for rnd, (hi, lo) in enumerate([(0.9, 0.1), (0.95, 0.05), (0.8, 0.2)]):
    # one owner: a record can be positive for at most its best S1
    d = pU.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("_rk"))
    d = d.with_columns(pl.when((pl.col("p") >= hi) & (pl.col("_rk") == 1)).then(1).when(pl.col("p") <= lo).then(0).otherwise(None).alias("yp")).drop_nulls("yp")
    log(f" hi {hi} lo {lo}: pseudo rows {d.height} ({d.height/pU.height:.3f} of US), pos {int(d['yp'].sum())}, pseudo-label accuracy {acc(d):.4f}, "
        f"pos precision {d.filter(pl.col('yp') == 1)['y'].mean():.4f}")
    m = fit(np.vstack([XI, M.X(d, fc)]), np.concatenate([yI, d["yp"].to_numpy()]))
    report(f"student India+US-pseudo (hi {hi}, lo {lo})", m)
    if rnd == 0:   # second round from the student
        p2 = UN.with_columns(pl.Series("p", m.predict(M.X(UN, fc)))).with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("_rk"))
        d2 = p2.with_columns(pl.when((pl.col("p") >= hi) & (pl.col("_rk") == 1)).then(1).when(pl.col("p") <= lo).then(0).otherwise(None).alias("yp")).drop_nulls("yp")
        m2 = fit(np.vstack([XI, M.X(d2, fc)]), np.concatenate([yI, d2["yp"].to_numpy()]))
        log(f"  round 2 pseudo rows {d2.height}, accuracy {acc(d2):.4f}")
        report(f"student round 2 (hi {hi}, lo {lo})", m2)
# soft distillation: all US rows, teacher probability as the label (cross-entropy)
ps = dict(params, objective="cross_entropy")
ms = lgb.train(ps, lgb.Dataset(np.vstack([XI, M.X(pU, fc)]), np.concatenate([yI, pU["p"].to_numpy()]), feature_name=fc), 1500)
report("student soft labels (cross_entropy on teacher p)", ms)
log(f"\nDONE {time.time()-T0:.0f}s")
