import polars as pl, sys; sys.path.insert(0, "research/eda"); from load import *
rows, pairs = gt()
s1 = read("train",1); R = pl.concat([read("train",2), read("train",3)])
R.write_parquet(f"{C}/train_R.parquet")
j = pairs.join(s1, left_on="s1", right_on="entity_id").join(R, left_on="r", right_on="entity_id", suffix="_r")
j.write_parquet(f"{C}/train_pairs_joined.parquet")
print("country agree:", (j["country"]==j["country_r"]).mean())
print(j.filter(pl.col("country")!=pl.col("country_r")).select("country","country_r").group_by(pl.all()).len())
pl.Config.set_tbl_rows(100); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
for c in ("US","India"):
    ids = j.filter(pl.col("country")==c)["s1"].unique().sample(6, seed=3)
    for i in ids:
        g = j.filter(pl.col("s1")==i)
        print(f"\n### {i} | {g['business_name'][0]} | {g['business_address'][0]}")
        for r in g.iter_rows(named=True): print(f"   {r['r']:14s} | {r['business_name_r']} | {r['business_address_r']}")
