import polars as pl, sys, time; sys.path.insert(0, "research/eda"); from load import *
t0=time.time()
rows, pairs = gt(); s1 = read("train",1); R = pl.read_parquet(f"{C}/train_R.parquet").with_row_index("rid")
LEG = r"\b(inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|lp|llp|pllc|pc|the|and|of|l|c|p|a|dba|fka|formerly)\b"
def toks(df, idcol):
    clean = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
    d = df.select(pl.col(idcol).alias("id"), pl.col("country").cast(pl.Categorical),
        clean("business_name").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEG, " ").str.extract_all(r"[a-z0-9]{2,}").list.unique(maintain_order=True).alias("nw"),
        clean("business_address").str.extract_all(r"[a-z]{3,}").alias("aw"),
        clean("business_address").str.extract_all(r"\d+").alias("ad"))
    w = d.with_columns(pl.col("nw").list.head(4).alias("h4"))
    # up to 6 unordered pairs among the first 4 name words
    prs = [pl.when(pl.col("h4").list.len() > j).then(pl.lit("P:") + pl.min_horizontal(pl.col("h4").list.get(i, null_on_oob=True), pl.col("h4").list.get(j, null_on_oob=True)) + "|" + pl.max_horizontal(pl.col("h4").list.get(i, null_on_oob=True), pl.col("h4").list.get(j, null_on_oob=True))) for i in range(4) for j in range(i+1, 4)]
    t = w.select("id","country", pl.concat_list(
            pl.col("nw").list.eval(pl.lit("n:")+pl.element()),
            pl.col("aw").list.eval(pl.lit("a:")+pl.element()),
            pl.col("ad").list.eval(pl.lit("#:")+pl.element().str.strip_chars_start("0")),
            pl.concat_list(pl.lit("N:")+pl.col("nw").list.sort().list.join(" ")),
            pl.concat_list(*prs)).list.drop_nulls().list.unique().alias("t"))
    return t.explode("t").drop_nulls("t").select("id","country", pl.col("t").hash().alias("h"))
Rt = toks(R, "rid"); print("R tokens", Rt.height, f"{time.time()-t0:.0f}s", flush=True)
df = Rt.group_by(["country","h"]).len().rename({"len":"df"})
Nc = R.group_by(pl.col("country").cast(pl.Categorical)).len().rename({"len":"N"})
df = df.join(Nc, on="country").with_columns((pl.col("N")/pl.col("df")).log().alias("idf")).filter(pl.col("df")<=20000).select("country","h","idf")
samp = s1.sample(20000, seed=11).with_row_index("sid"); St = toks(samp, "sid").join(df, on=["country","h"])
Rt = Rt.join(St.select("country","h").unique(), on=["country","h"]); print("R tokens relevant", Rt.height, flush=True)
cand = St.join(Rt, on=["country","h"], suffix="_r").group_by(["id","id_r"]).agg(pl.col("idf").sum().alias("sc"))
cand = cand.with_columns(pl.col("sc").rank("ordinal", descending=True).over("id").alias("rk")).filter(pl.col("rk")<=100)
cand = cand.join(samp.select(pl.col("sid").alias("id"), pl.col("entity_id").alias("s1")), on="id").join(R.select(pl.col("rid").alias("id_r"), pl.col("entity_id").alias("r"), pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic")), on="id_r")
print("cands", cand.height, f"{time.time()-t0:.0f}s")
P = pairs.filter(pl.col("s1").is_in(samp["entity_id"].implode())).join(R.select(pl.col("entity_id").alias("r"), pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic")), on="r")
hit = P.join(cand.select("s1","r","rk"), on=["s1","r"], how="left")
for K in (5,10,15,20,30,50,100):
    print(f"recall@{K:3d}: {(hit['rk'].fill_null(10**9)<=K).mean():.4f}   non-Indic-name: {(hit.filter(~pl.col('indic'))['rk'].fill_null(10**9)<=K).mean():.4f}")
print("share of positives with Indic-script name:", hit["indic"].mean())
