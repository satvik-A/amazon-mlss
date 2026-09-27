"""Re-decide final4 with other rank thresholds from its saved level-2 band scores (no Kaggle rerun).
final4 saved level2_band_<shard>.parquet = every row (s1, r, p) of each S1 that has any p in (0.01, 0.99). S1s outside the band
decide identically for any thresholds in (0.01, 0.99), so their final4 pairs are kept (p treated as 1.0 for one_owner).
Band S1s: rank-threshold + source caps -> France street filter -> one_owner over everything -> write_submission.
Usage: python research/revariant.py <name> '<json overrides: {"all": {"t1":..,"t2":..}, "France": {...}, "US": {...}, "India": {...}}>'
       name "check" with {} must reproduce final4 exactly."""
import sys, json, glob, os
sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from ber import model as M
from ber.decide import rank_threshold, source_caps

name, ov = sys.argv[1], json.loads(sys.argv[2])
BASE = dict(t1=0.8, t2=0.75)   # decision_stack.json of er-stack4
CACHE = "research/eda/cache"; OUT = f"kaggle/revariants/{name}"; os.makedirs(OUT, exist_ok=True)
thr = lambda c: {**BASE, **ov.get("all", {}), **ov.get(c, {})}

F4 = pl.read_csv("kaggle/final4/sub/matching_results.tsv", separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8}) \
       .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
band_s1, sel = [], []
for f in sorted(glob.glob("kaggle/final4/band/level2_band_*.parquet")):
    c = os.path.basename(f)[len("level2_band_"):].rsplit("_", 1)[0]
    d = pl.read_parquet(f); t = thr(c)
    band_s1.append(d.select("s1").unique())
    s = source_caps(rank_threshold(d, t["t1"], t["t2"]), d).join(d.select("s1", "r", "p"), on=["s1", "r"])
    sel.append(s)
band_s1 = pl.concat(band_s1).unique()
keep = F4.join(band_s1, on="s1", how="anti").with_columns(pl.lit(1.0).alias("p"))
S1 = pl.read_parquet(f"{CACHE}/test_s1.parquet")
R = pl.concat([pl.read_parquet(f"{CACHE}/test_s{k}.parquet", columns=["entity_id", "business_address", "country"]) for k in (2, 3)])
banded = M.street_filter(pl.concat(sel), S1, R, log=lambda *a: None)
Mt = M.one_owner(pl.concat([keep.select("s1", "r", "p"), banded.select("s1", "r", "p")]), log=lambda *a: None)
# same file format as write_submission: one row per S1, comma-joined ids, empty string when none
out = S1.select(pl.col("entity_id").alias("source1_entity_id")).join(
    Mt.group_by("s1").agg(pl.col("r").sort().str.join(",").alias("matched_entity_ids")).rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left") \
    .with_columns(pl.col("matched_entity_ids").fill_null(""))
out.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
ctry = S1.select(pl.col("entity_id").alias("s1"), "country")
a = F4.select("s1", "r"); b = Mt.select("s1", "r")
same = a.join(b, on=["s1", "r"]).height
print(f"{name}: pairs {b.height} (final4 {a.height}); shared {same}; only-final4 {a.height-same}, only-new {b.height-same}; band S1 {band_s1.height}")
print(b.join(ctry, on="s1").group_by("country").len().sort("country").rows(), "vs final4", a.join(ctry, on="s1").group_by("country").len().sort("country").rows())
