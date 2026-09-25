import polars as pl, sys; sys.path.insert(0, "research/eda"); from load import *
rows, pairs = gt()
s1 = {sp: read(sp, 1) for sp in ("train", "test")}
print("GT rows", rows.height, "unique S1", rows["source1_entity_id"].n_unique(), "train S1", s1["train"].height)
print("in GT but not S1:", rows.filter(~pl.col("source1_entity_id").is_in(s1["train"]["entity_id"].implode())).height)
nm = rows.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element()!="")).list.len().alias("n"))
print("singleton rate:", (nm["n"]==0).mean())
print(nm["n"].value_counts().sort("n").head(15))
pairs = pairs.with_columns(pl.col("r").str.slice(0,2).alias("src"))
per = pairs.group_by(["s1","src"]).agg(pl.len().alias("k"))
print("matches per S1 per source:\n", per.group_by(["src","k"]).len().sort(["src","k"]))
# H1 exclusivity
dup = pairs.group_by("r").len().filter(pl.col("len")>1)
print("S2/S3 IDs under >1 S1:", dup.height, "of", pairs["r"].n_unique())
for s in (2,3):
    df = read("train", s)
    print(f"train S{s}: {df.height} rows; matched to some S1: {pairs.filter(pl.col('src')==f'S{s}')['r'].n_unique()}  -> orphan rate {1 - pairs.filter(pl.col('src')==f'S{s}')['r'].n_unique()/df.height:.3f}")
for sp in ("train","test"):
    for s in (1,2,3):
        df = read(sp, s); print(sp, s, df.height, df["country"].value_counts().sort("country").rows())
