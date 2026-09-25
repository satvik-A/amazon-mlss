import polars as pl, sys; sys.path.insert(0, "research/eda"); from load import *
t1, t2, t3 = read("test",1), read("test",2), read("test",3)
for name, df in [("S1",t1),("S2",t2),("S3",t3)]:
    f = df.filter(pl.col("country")=="France")
    print(f"\n==== test {name} France sample")
    for r in f.sample(10, seed=4).iter_rows(named=True): print(f"  {r['entity_id']:14s} | {r['business_name']} | {r['business_address']}")
R = pl.concat([t2,t3])
for c in ("US","India","France"):
    s = t1.filter(pl.col("country")==c).height; r = R.filter(pl.col("country")==c).height
    print(f"{c:7s} test S1 {s:8d}  R {r:8d}  R/S1 {r/s:.2f}")
tr1 = read("train",1); trR = pl.read_parquet(f"{C}/train_R.parquet")
for c in ("US","India"):
    print(f"{c:7s} train R/S1 {trR.filter(pl.col('country')==c).height/tr1.filter(pl.col('country')==c).height:.2f}")
kw = r"(?i)\b(holdings|group|ventures|enterprises|lakeside|eastgate|harbor|downtown)\b"
for c in ("US","India","France"):
    print(c, "decoy-word rate in R names:", R.filter(pl.col("country")==c)["business_name"].str.contains(kw).mean(),
          " in S1:", t1.filter(pl.col("country")==c)["business_name"].str.contains(kw).mean())
fr = R.filter(pl.col("country")=="France")
print("France R: accent", fr["business_name"].str.contains(r"[À-ÿ]").mean(), "addr None", (fr["business_address"]=="None").mean(),
      "5-digit", fr["business_address"].str.contains(r"\b\d{5}\b").mean(), "web", fr["business_name"].str.contains(r"(?i)\.(com|fr)\b|^@").mean(),
      "fka", fr["business_name"].str.contains(r"(?i)f/k/a|formerly|dba").mean())
print("France S1 legal tokens:", t1.filter(pl.col("country")=="France")["business_name"].str.extract(r"(\S+)$").value_counts().sort("count", descending=True).head(15).rows())
