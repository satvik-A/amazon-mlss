import polars as pl, sys, time; sys.path.insert(0, "research/eda"); from load import *
t0=time.time()
rows, pairs = gt(); s1 = read("train",1); R = pl.read_parquet(f"{C}/train_R.parquet")
def toks(df):
    clean = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
    return df.select("entity_id","country",
        pl.concat_list(
            clean("business_name").str.extract_all(r"[a-z]{2,}").list.eval(pl.lit("n:")+pl.element()),
            clean("business_address").str.extract_all(r"[a-z]{3,}").list.eval(pl.lit("a:")+pl.element()),
            clean("business_address").str.extract_all(r"\d+").list.eval(pl.lit("#:")+pl.element().str.strip_chars_start("0")),
        ).list.unique().alias("t")).explode("t").drop_nulls("t").with_columns(pl.col("t").hash().alias("h")).drop("t")
Rt = toks(R); print("R tokens", Rt.height, f"{time.time()-t0:.0f}s")
df = Rt.group_by(["country","h"]).len().rename({"len":"df"})
Nc = R.group_by("country").len().rename({"len":"N"})
df = df.join(Nc, on="country").with_columns((pl.col("N")/pl.col("df")).log().alias("idf")).filter(pl.col("df")<=3000).select("country","h","idf","df")
samp = s1.sample(20000, seed=11); St = toks(samp).join(df, on=["country","h"])
Rt = Rt.join(St.select("country","h").unique(), on=["country","h"])
cand = St.join(Rt, on=["country","h"], suffix="_r").group_by(["entity_id","entity_id_r"]).agg(pl.col("idf").sum().alias("sc"))
cand = cand.with_columns(pl.col("sc").rank("ordinal", descending=True).over("entity_id").alias("rk"))
print("cands", cand.height, f"{time.time()-t0:.0f}s")
P = pairs.filter(pl.col("s1").is_in(samp["entity_id"].implode()))
hit = P.join(cand, left_on=["s1","r"], right_on=["entity_id","entity_id_r"], how="left")
for K in (5,10,15,20,30,50,100):
    print(f"recall@{K:3d}: {(hit['rk'].fill_null(10**9)<=K).mean():.4f}")
miss = hit.filter(pl.col("rk").is_null() | (pl.col("rk")>30)).join(R, left_on="r", right_on="entity_id").join(s1, left_on="s1", right_on="entity_id", suffix="_s1")
print("missed@30 examples:")
pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
for r in miss.sample(min(15, miss.height), seed=0).iter_rows(named=True):
    print(f"  rk={r['rk']} | {r['business_name_s1']} | {r['business_address_s1'][:50]}  ==>  {r['business_name']} | {(r['business_address'] or '')[:50]}")
