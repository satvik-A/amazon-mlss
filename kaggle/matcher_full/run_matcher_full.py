"""Matcher on FULL-mode train pools (same candidate definition as test). Split blending by S1 hash: A 60 / B 30 / C 10.
Chooses the decision rule and the blocking policy on B; reports everything on held-out C.
Writes matcher.txt, head.txt, decision.json (consumed by the submission job)."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.normalize import Normalizer
from ber.metrics import macro_f05
from ber.decide import rank_threshold
from ber import model as M

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
N_S1 = int(os.environ.get("N_S1", 3000 if LOCAL else 120_000))  # sampled S1 per country
REP = open(f"{WD}/results_matcher_full.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet")[0])
POOLS = sorted(glob.glob("../cands_full/pool_train_*.parquet")) if LOCAL else find("pool_train_*.parquet")
FRONTS = sorted(glob.glob("../cands_full/frontier_train_*.csv")) if LOCAL else find("frontier_train_*.csv")
log(f"ref {REF}; pools {POOLS}; N_S1/country {N_S1}")
NZ = Normalizer(ART)

# ---- sample S1s per country (deterministic hash), features --------------------------------------------------------
parts = []
for p in POOLS:
    P = pl.read_parquet(p)
    ids = P["s1"].unique().sort()
    ids = ids.filter((ids.hash(7) % 1_000_000) < int(1_000_000 * min(1.0, N_S1 / len(ids))))
    parts.append(P.filter(pl.col("s1").is_in(ids.implode())))
    log(f"{os.path.basename(p)}: {P.height} rows, {P['s1'].n_unique()} S1 -> sample {parts[-1]['s1'].n_unique()} S1 / {parts[-1].height} rows")
    del P
pool = pl.concat(parts, how="vertical_relaxed"); del parts
S1ids = pool["s1"].unique()
s1 = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("entity_id").is_in(S1ids.implode())).collect()
R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("entity_id").is_in(pool["r"].unique().implode())).collect()
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S1ids.implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
log(f"pool {pool.height} rows; positives in pool {pool['y'].sum()} / {gt.height} = {pool['y'].sum()/gt.height:.4f}")
F = M.pool_features(pool, s1, R, NZ).join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
del pool; fc = M.feature_columns(F); log(f"features {len(fc)}  {time.time()-T0:.0f}s")

F = F.with_columns((pl.col("s1").hash(11) % 100).alias("h"))
A, Bv, Cv = F.filter(pl.col("h") < 60), F.filter((pl.col("h") >= 60) & (pl.col("h") < 90)), F.filter(pl.col("h") >= 90)
del F
params = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
dA = lgb.Dataset(M.X(A, fc), A["y"].to_numpy(), feature_name=fc, free_raw_data=True)
dB = lgb.Dataset(M.X(Bv, fc), Bv["y"].to_numpy(), reference=dA)
m = lgb.train(params, dA, 4000, valid_sets=[dB], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
m.save_model(f"{WD}/matcher.txt"); del dA, dB, A
Bv = Bv.with_columns(pl.Series("p", m.predict(M.X(Bv, fc)))); Cv = Cv.with_columns(pl.Series("p", m.predict(M.X(Cv, fc))))
log("top features (gain):", [(f, int(g)) for f, g in sorted(zip(fc, m.feature_importance("gain")), key=lambda x: -x[1])[:25]])
sub = lambda g_, d: g_.filter(pl.col("s1").is_in(d["s1"].unique().implode()))
gB, gC = sub(gt, Bv), sub(gt, Cv); sB, sC = Bv["s1"].unique(), Cv["s1"].unique()
f05 = lambda sel, g_, s_: macro_f05(sel, g_, s_)

# ---- decision rules (tuned on B, reported on C) ----------------------------------------------------------------------
grid = [(a, b) for a in np.arange(0.2, 0.95, 0.05) for b in np.arange(0.2, 0.95, 0.05)]
t1, t2, _ = max(((a, b, f05(rank_threshold(Bv, a, b), gB, sB)["f05"]) for a, b in grid), key=lambda x: x[2])
gam = max(((g, f05(M.efd(Bv, g), gB, sB)["f05"]) for g in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)), key=lambda x: x[1])[0]
# has-head: trained on half of B (out-of-sample p), rules compared on the other half of B, reported on C
HB = M.s1_agg(Bv).join(gB.group_by("s1").len().rename({"len": "nt"}), on="s1", how="left").with_columns((pl.col("nt").fill_null(0) > 0).cast(pl.Int8).alias("y1"))
hp = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=os.cpu_count())
head = lgb.train(hp, lgb.Dataset(M.X(HB, M.HEAD_FEATURES), HB["y1"].to_numpy()), 400); head.save_model(f"{WD}/head.txt")
cfg0 = dict(t1=float(t1), t2=float(t2), gamma=float(gam))
log(f"\nrank thresholds t1={t1:.2f} t2={t2:.2f}; expected-F gamma={gam}")
log("=== decision rules (no extra pruning) ===")
resB, resC = {}, {}
for rule in M.RULES:
    cfg = dict(cfg0, rule=rule)
    resB[rule] = f05(M.decide(Bv, cfg, head), gB, sB)["f05"]   # has-head is in-sample on B: C is the honest number
    mm = f05(M.decide(Cv, cfg, head), gC, sC); resC[rule] = mm
    log(f"  {rule:26s} B {resB[rule]:.4f} | C {mm['f05']:.4f}  singleton {mm['singleton_f']:.4f}  non-singleton {mm['nonsingleton_f']:.4f}  P {mm['pair_precision']:.4f} R {mm['pair_recall']:.4f}")
rule = max((r for r in M.RULES if r != "expected-F + has-head"), key=lambda r: resB[r])
log(f"rule chosen on B (has-head excluded from choice: in-sample on B): {rule}; has-head on C for reference {resC['expected-F + has-head']['f05']:.4f}")
ceil = f05(Cv.filter(pl.col("y") == 1).select("s1", "r"), gC, sC); log(f"ceiling on C (perfect decisions on the pool): {ceil['f05']:.4f}")
per = resC[rule]["per"].join(s1.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
log(per.group_by("country").agg(pl.col("f").mean(), pl.len()))

# ---- blocking policy: end-to-end F per policy (Pareto points of the full-mode frontiers) --------------------------
fr = pl.concat([pl.read_csv(f) for f in FRONTS]).group_by(["alpha", "G", "beta"]).agg(pl.col("cands").mean(), pl.col("complete").mean()).sort("cands")
best, pts = -1.0, []
for r in fr.iter_rows(named=True):
    if r["complete"] > best + 1e-4: pts.append(r); best = r["complete"]
pts.append(dict(alpha=0.0, G=99, beta=-1.0, cands=None, complete=None))  # no pruning (internal pool)
rows = []
for r in pts:
    pol = dict(alpha=r["alpha"], G=int(r["G"]), gate=False, beta=r["beta"])
    kb, kc = M.prune_pool(Bv, pol), M.prune_pool(Cv, pol)
    cfg = dict(cfg0, rule=rule)
    rows.append(dict(**pol, cands_B=kb.height / len(sB), F_B=f05(M.decide(kb, cfg, head), gB, sB)["f05"],
                     cands_C=kc.height / len(sC), F_C=f05(M.decide(kc, cfg, head), gC, sC)["f05"]))
E = pl.DataFrame(rows).sort("cands_B"); log("\n=== END-TO-END per blocking policy ==="); log(E)
fmax = E["F_B"].max()
pick = E.filter(pl.col("F_B") >= fmax - 0.0005).sort("cands_B").row(0, named=True)   # smallest set within 0.0005 of best
log(f"policy chosen on B (smallest candidate set within 0.0005 of the best F): {pick}")
dec = dict(cfg0, rule=rule, policy=dict(alpha=pick["alpha"], G=pick["G"], gate=False, beta=pick["beta"]), features=fc,
           head_features=M.HEAD_FEATURES, ref=REF, holdout_C=dict(F=pick["F_C"], cands=pick["cands_C"]))
json.dump(dec, open(f"{WD}/decision.json", "w"), indent=1)
log(json.dumps(dec)[:600]); log(f"\nDONE {time.time()-T0:.0f}s")
