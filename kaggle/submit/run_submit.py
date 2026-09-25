"""Submission: FULL-mode test pools -> blocking policy (candidate_pairs) -> matcher -> decision rule (matching_results).
Inputs: er-cands-test-* (pool_test_<country>.parquet), er-matcher-full (matcher.txt, head.txt, decision.json), artifacts, data."""
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
from ber.io import write_submission
from ber import model as M
from ber.artifacts import pseudo_pairs

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_submit.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("test_s1.parquet")[0])
ART = "../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet")[0])
POOLS = sorted(glob.glob("../cands_full/pool_test_*.parquet")) if LOCAL else find("pool_test_*.parquet")
MD = "../matcher_full" if LOCAL else os.path.dirname(find("decision.json")[0])
cfg = json.load(open(f"{MD}/decision.json")); m = lgb.Booster(model_file=f"{MD}/matcher.txt")
head = lgb.Booster(model_file=f"{MD}/head.txt") if os.path.exists(f"{MD}/head.txt") else None
fc = cfg["features"]
# optional level 2 (er-stack + cross-encoder scores on the test uncertain band from er-xenc-score)
SJ = find("decision_stack.json") if not LOCAL else []
stack = lgb.Booster(model_file=os.path.join(os.path.dirname(SJ[0]), "stack.txt")) if SJ else None
if SJ:
    cfg2 = json.load(open(SJ[0])); log(f"LEVEL 2 active: delta on C {cfg2.get('delta_C')}")
log(f"ref {REF}; model ref {cfg.get('ref')}; rule {cfg['rule']}; policy {cfg['policy']}; holdout {cfg.get('holdout_C')}")
NZ = Normalizer(ART)
s1_all = pl.read_parquet(f"{IN}/test_s1.parquet")
valid_r = pl.concat([pl.read_parquet(f"{IN}/test_s2.parquet", columns=["entity_id"]), pl.read_parquet(f"{IN}/test_s3.parquet", columns=["entity_id"])])["entity_id"]
cands, matches = [], []
for p in POOLS:
    pool = pl.read_parquet(p); ctry = os.path.basename(p)[len("pool_test_"):-len(".parquet")]
    kept = M.prune_pool(pool, cfg["policy"]); del pool
    S = s1_all.filter(pl.col("country") == ctry)
    R = pl.concat([pl.scan_parquet([f"{IN}/test_s2.parquet", f"{IN}/test_s3.parquet"]).filter(pl.col("entity_id").is_in(kept["r"].unique().implode())).collect()])
    F = M.pool_features(kept, S, R, NZ)
    F = F.with_columns(pl.Series("p", m.predict(M.X(F, fc))))
    F.write_parquet(f"{WD}/level1_test_{ctry}.parquet")   # for er-xenc-score (uncertain band) and level 2
    xf = find(f"xenc_test_{ctry}.parquet") if not LOCAL else []
    if stack is not None and xf:
        F = F.join(pl.read_parquet(xf[0]), on=["s1", "r"], how="left")
        F = F.with_columns(pl.Series("p", stack.predict(M.X(F, cfg2["stack_features"]))))
        sel = M.decide(F, cfg2, head); log(f"  {ctry}: level-2 decisions ({F[cfg2['stack_features'][-1]].is_not_null().sum()} pairs with cross-encoder scores)")
    else:
        sel = M.decide(F, cfg, head)
    per = sel.group_by("s1").len()
    log(f"{ctry}: S1 {S.height}  candidates {kept.height} ({kept.height/S.height:.2f}/S1)  predicted matches {sel.height} "
        f"({sel.height/S.height:.2f}/S1)  S1 predicted empty {1 - per.height/S.height:.4f}  mean p {F['p'].mean():.4f}  {time.time()-T0:.0f}s")
    # label-free transfer check: pseudo-pairs (>=99% precise on US/India train) that are candidates -> share predicted
    pp = pseudo_pairs(NZ.transform(S), NZ.transform(R)).join(kept.select("s1", "r"), on=["s1", "r"], how="semi")
    hit = pp.join(sel.select("s1", "r"), on=["s1", "r"], how="semi").height
    pS = F.join(pp, on=["s1", "r"], how="semi")["p"]
    log(f"  {ctry} pseudo-pairs in candidates {pp.height} ({pp['s1'].n_unique()/S.height:.3f} of S1): predicted as match {hit/max(pp.height,1):.4f}, "
        f"mean p {pS.mean() if pS.len() else float('nan'):.4f}, p<0.5 share {(pS < 0.5).mean() if pS.len() else float('nan'):.4f}")
    cands.append(kept.select("s1", "r")); matches.append(sel.select("s1", "r"))
    del F, kept, R
C = pl.concat(cands); Mt = pl.concat(matches)
write_submission(Mt, C, s1_all["entity_id"], valid_r, f"{WD}/matching_results.tsv", f"{WD}/candidate_pairs.tsv")
log(f"total: candidates/S1 {C.height/s1_all.height:.2f}, matches/S1 {Mt.height/s1_all.height:.2f}, S1 with no match {1 - Mt['s1'].n_unique()/s1_all.height:.4f}")
log(f"wrote matching_results.tsv + candidate_pairs.tsv  DONE {time.time()-T0:.0f}s")
