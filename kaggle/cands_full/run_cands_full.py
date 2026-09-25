"""Full candidate generation for ONE (split, country) in FULL mode (every S1 of the country queried).
Writes the internal pool (pre-pruning, annotated) and, for train, the full-mode pruning frontier on ALL S1s.
Config: CFG below is filled in by make_jobs.py (Kaggle uploads only the code file); locally: env SPLIT / COUNTRY."""
import glob, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
CFG = {"split": "__SPLIT__", "country": "__COUNTRY__", "ref": "__REF__"}
if CFG["split"].startswith("__"):
    CFG = {"split": os.environ.get("SPLIT", "train"), "country": os.environ.get("COUNTRY", "US"), "ref": "local"}
LOCAL = not os.path.exists("/kaggle")
if LOCAL:
    sys.path.insert(0, os.path.abspath(f"{HERE}/../../code/business_entity_resolution/src"))
elif glob.glob("/kaggle/input/**/er-src/**/ber/__init__.py", recursive=True):   # account 2: no internet -> offline bundle
    _W = os.path.dirname(glob.glob("/kaggle/input/**/polars-1.44.2*.whl", recursive=True)[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", _W, "polars==1.44.2", "rapidfuzz==3.14.6", "lightgbm==4.6.0"], check=True)
    sys.path.insert(0, os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/er-src/**/ber/__init__.py", recursive=True)[0])))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"git+https://github.com/satvik-A/amazon-mlss.git@{CFG['ref']}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl
from ber.normalize import Normalizer
from ber.pipeline import candidates
from ber import blocking as B

T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"; SPLIT, CTRY = CFG["split"], CFG["country"]
TAG = f"{SPLIT}_{CTRY}"
REP = open(f"{WD}/results_{TAG}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
find = lambda f: glob.glob(f"/kaggle/input/**/{f}", recursive=True)[0]
IN = f"{HERE}/../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet"))
ART = f"{HERE}/../artifacts/kout/artifacts" if LOCAL else os.path.dirname(find("indic_lexicon.parquet"))
NZ = Normalizer(ART)
log(f"{TAG}: ref {CFG['ref']}  country synonyms {'yes' if NZ.addr_syn_c is not None else 'no'}")
S = pl.scan_parquet(f"{IN}/{SPLIT}_s1.parquet").filter(pl.col("country") == CTRY).collect()
R = pl.scan_parquet([f"{IN}/{SPLIT}_s2.parquet", f"{IN}/{SPLIT}_s3.parquet"]).filter(pl.col("country") == CTRY).collect()
if LOCAL:
    S = S.head(5000); R = R.head(150000)
log(f"S1 {S.height}  R {R.height}  {time.time()-T0:.0f}s")
ann = candidates(S, R, S, NZ, log=log)
keep = ["s1", "r", "sc", "prk", "xrk", *[f"a{k}" for k in B.ARMS], "exp", "gid", "rel", "rev_margin"]
ann = ann.select(keep)
SWEEP = os.environ.get("SWEEP", "0") == "1"   # group/beta pruning frontier (known lossy; expensive at 100M+ rows)
if SPLIT == "train":
    gt = pl.read_parquet(f"{IN}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(S["entity_id"].implode())) \
           .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
           .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
    ann = ann.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
    comp = gt.join(ann.filter(pl.col("y") == 1).select("s1", "r", pl.lit(True).alias("k")), on=["s1", "r"], how="left").group_by("s1").agg(pl.col("k").fill_null(False).all())["k"].mean()
    log(f"internal pool {ann.height/S.height:.1f}/S1; pair recall {ann['y'].sum()/gt.height:.4f}; S1 complete {comp:.4f}")
if SPLIT == "train" and SWEEP:
    A = ann.select(pl.col("s1").alias("id"), pl.col("r").alias("id_r"), "sc", "gid", "rel", "a4", "rev_margin")
    rows = []
    for alpha in (0.0, 0.5, 0.7, 0.8):
        for G in (2, 3, 5, 99):
            for beta in (None, 0.5, 0.25, 0.0):
                k = B.prune(A, alpha=alpha, G=G, beta=beta).rename({"id": "s1", "id_r": "r"}).with_columns(pl.lit(True).alias("k"))
                h = gt.join(k, on=["s1", "r"], how="left").with_columns(pl.col("k").fill_null(False))
                rows.append(dict(alpha=alpha, G=G, beta=-1 if beta is None else beta, cands=k.height / S.height,
                                 recall=h["k"].mean(), complete=h.group_by("s1").agg(pl.col("k").all())["k"].mean()))
    F = pl.DataFrame(rows).sort("cands"); F.write_csv(f"{WD}/frontier_{TAG}.csv")
    best, par = -1, []
    for r in F.iter_rows(named=True):
        if r["complete"] > best + 1e-4: par.append(r); best = r["complete"]
    log("FULL-mode Pareto frontier:"); log(pl.DataFrame(par))
ann.write_parquet(f"{WD}/pool_{TAG}.parquet")
log(f"wrote pool_{TAG}.parquet {ann.height} rows  DONE {time.time()-T0:.0f}s")
