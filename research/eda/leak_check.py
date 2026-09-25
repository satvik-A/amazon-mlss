"""Do entity-id numbers or file row order leak cluster membership?"""
import polars as pl, numpy as np
C = "research/eda/cache"
pairs = pl.read_parquet(f"{C}/gt_pairs.parquet")
row = {}
for s in ("train_s2", "train_s3"):
    d = pl.read_parquet(f"{C}/{s}.parquet", columns=["entity_id"]).with_row_index("row")
    row[s[-2:].upper()] = d
R = pl.concat([row["S2"].with_columns(pl.lit("S2").alias("src")), row["S3"].with_columns(pl.lit("S3").alias("src"))])
s1 = pl.read_parquet(f"{C}/train_s1.parquet", columns=["entity_id"]).with_row_index("s1row")
p = pairs.join(R, left_on="r", right_on="entity_id").join(s1, left_on="s1", right_on="entity_id") \
         .with_columns(pl.col("r").str.slice(3).cast(pl.Int64).alias("rid_num"), pl.col("s1").str.slice(3).cast(pl.Int64).alias("s1_num"))
# 1) row-order: within one S1's matches in the SAME source file, how far apart are their rows vs random pairs?
g = p.group_by(["s1", "src"]).agg(pl.col("row").sort().diff().drop_nulls().alias("d")).explode("d").drop_nulls("d")
n2 = row["S2"].height
print("median |row gap| between copies of the same entity (same file):", g["d"].median(), " | random expectation ~", n2 // 3)
print("share of copies within 10 rows of a sibling:", (g["d"] <= 10).mean())
# 2) id numbers: correlation of S1 id number with its matches' id numbers; and sibling id gaps
print("corr(S1 id num, R id num):", np.corrcoef(p["s1_num"].to_numpy(), p["rid_num"].to_numpy())[0, 1])
gi = p.group_by("s1").agg(pl.col("rid_num").sort().diff().drop_nulls().alias("d")).explode("d").drop_nulls("d")
print("median id gap between siblings:", gi["d"].median(), " | id range ~", p["rid_num"].max())
# 3) S1 row order vs R row order
print("corr(S1 row, R row):", np.corrcoef(p["s1row"].to_numpy(), p["row"].to_numpy())[0, 1])
# 4) does the id number encode source or country? (id ranges per source)
print(R.with_columns(pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("n")).group_by("src").agg(pl.col("n").min(), pl.col("n").max(), pl.len()))
