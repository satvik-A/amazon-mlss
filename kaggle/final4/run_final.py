"""[pass 4] Final submission from saved level-1 test scores (no feature recomputation):
level1_test_<c>_<k>.parquet (er-l1test4-*) + xenc_test_<c>_<k>.parquet (er-xenc-score-test4-*) -> level-2 stack (er-stack4,
used only if it passed its gate on holdout C) -> decision rule -> matching_results.tsv + candidate_pairs.tsv (validator-safe).
Also writes the level-1 submission (matches_l1_*) to l1/ as a fallback, and the level-2 uncertain band for the level-3 judge."""
import glob, json, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath("../../code/business_entity_resolution/src"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm==4.6.0",
                    f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, lightgbm as lgb
from ber.io import write_submission
from ber import model as M
T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"
REP = open(f"{WD}/results_final.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = os.path.dirname(find("test_s1.parquet")[0])
MD = os.path.dirname(find("decision.json")[0])
cfg = json.load(open(f"{MD}/decision.json"))
head = lgb.Booster(model_file=f"{MD}/head.txt") if os.path.exists(f"{MD}/head.txt") else None
SJ = find("decision_stack.json")
stack, cfg2 = None, None
if SJ:
    cfg2 = json.load(open(SJ[0]))
    if cfg2.get("delta_C", 0) >= 0.002:
        stack = lgb.Booster(model_file=os.path.join(os.path.dirname(SJ[0]), "stack.txt"))
        log(f"LEVEL 2 active: delta on C {cfg2['delta_C']:+.4f}")
    else:
        log(f"level 2 REJECTED by its gate (delta on C {cfg2.get('delta_C')}) -> level 1")
# pass 4: optional competition + sibling layer (trained offline on full-country train scores; dataset er-comp4)
CJ = find("comp.json")
comp, cfg3 = None, None
if CJ and stack is not None:
    cfg3 = json.load(open(CJ[0])); comp = lgb.Booster(model_file=os.path.join(os.path.dirname(CJ[0]), "comp.txt"))
    log(f"COMPETITION layer active: features {cfg3['features']}; t1 {cfg3['t1']} t2 {cfg3['t2']}; C {cfg3.get('C')}")
from ber.compete import comp_features, sib_features
by_ctry = {}
s1_all = pl.read_parquet(f"{IN}/test_s1.parquet")
valid_r = pl.concat([pl.read_parquet(f"{IN}/test_s2.parquet", columns=["entity_id"]), pl.read_parquet(f"{IN}/test_s3.parquet", columns=["entity_id"])])["entity_id"]
xf = {os.path.basename(f): f for f in find("xenc_test_*.parquet")}
cands = [pl.read_parquet(f) for f in find("cands_*.parquet")]
l1 = [pl.read_parquet(f) for f in find("matches_l1_*.parquet")]
log(f"inputs: {len(find('level1_test_*.parquet'))} level-1 files, {len(xf)} cross-encoder files, {len(cands)} candidate files")
matches, n_x = [], 0
for f in find("level1_test_*.parquet"):
    F = pl.read_parquet(f)
    tag = os.path.basename(f)[len("level1_test_"):]
    if stack is not None:
        x = xf.get(f"xenc_test_{tag}")
        assert x is not None, f"missing cross-encoder scores for {tag}"   # never mix stacked and unstacked shards silently
        F = F.join(pl.read_parquet(x), on=["s1", "r"], how="left"); n_x += 1
        F = F.with_columns(pl.col("p").alias("p1"), pl.Series("p", stack.predict(M.X(F, cfg2["stack_features"]))))
        sel = M.decide(F, cfg2, head)
        if comp is not None:   # decided after all shards of the country are in (records compete across shards)
            by_ctry.setdefault(tag.rsplit("_", 1)[0], []).append(F.select("s1", "r", "p1", pl.col("p").alias("p2"), "noaddr_r"))
            sel = sel.head(0)
        bs = F.filter((pl.col("p") > 0.01) & (pl.col("p") < 0.99))["s1"].unique()
        F.filter(pl.col("s1").is_in(bs.implode())).select("s1", "r", "p").write_parquet(f"{WD}/level2_band_{tag}")
    else:
        sel = M.decide(F, cfg, head)
    matches.append(sel.select("s1", "r").join(F.select("s1", "r", "p"), on=["s1", "r"], how="left"))
    log(f"{tag}: rows {F.height} S1 {F['s1'].n_unique()} matches {sel.height}  {time.time()-T0:.0f}s")
if comp is not None:
    for ctry, parts in by_ctry.items():
        D = comp_features(pl.concat(parts)); n0 = D.height
        D = D.filter(pl.col("p1") > 0.0005)
        Rc = pl.scan_parquet([f"{IN}/test_s2.parquet", f"{IN}/test_s3.parquet"]).filter(pl.col("country") == ctry) \
               .select("entity_id", "business_name", "business_address").collect()
        D = sib_features(D, Rc); del Rc
        D = D.with_columns(pl.Series("p", comp.predict(M.X(D, cfg3["features"]))))
        sel = M.decide(D, dict(rule="rank-threshold", t1=cfg3["t1"], t2=cfg3["t2"]), None)
        matches.append(sel.select("s1", "r").join(D.select("s1", "r", "p"), on=["s1", "r"], how="left"))
        log(f"{ctry}: competition layer on {D.height} of {n0} rows -> {sel.height} matches  {time.time()-T0:.0f}s")
        del D
C = pl.concat(cands)
Rraw = pl.scan_parquet([f"{IN}/test_s2.parquet", f"{IN}/test_s3.parquet"]).filter(pl.col("country") == "France").select("entity_id", "business_address", "country").collect()
Mt = M.one_owner(M.street_filter(pl.concat(matches), s1_all, Rraw, log=log), log=log); del Rraw
write_submission(Mt, C, s1_all["entity_id"], valid_r, f"{WD}/matching_results.tsv", f"{WD}/candidate_pairs.tsv")
log(f"{'STACKED' if stack is not None else 'LEVEL-1'} submission: shards {len(matches)} (with cross-encoder {n_x}); candidates/S1 {C.height/s1_all.height:.2f}, "
    f"matches/S1 {Mt.height/s1_all.height:.2f}, S1 with no match {1 - Mt['s1'].n_unique()/s1_all.height:.4f}")
os.makedirs(f"{WD}/l1", exist_ok=True)
L1 = pl.concat(l1)
write_submission(L1, C, s1_all["entity_id"], valid_r, f"{WD}/l1/matching_results.tsv", f"{WD}/l1/candidate_pairs.tsv")
log(f"level-1 fallback in l1/: matches/S1 {L1.height/s1_all.height:.2f}")
log(f"DONE {time.time()-T0:.0f}s")
