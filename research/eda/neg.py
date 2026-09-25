import polars as pl, sys, re; sys.path.insert(0, "research/eda"); from load import *
pl.Config.set_tbl_rows(100); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250)
rows, pairs = gt(); s1 = read("train",1); R = pl.read_parquet(f"{C}/train_R.parquet")
matched = pairs["r"].implode()
orph = R.filter(~pl.col("entity_id").is_in(matched))
sing = rows.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids")==""))["source1_entity_id"]
S = s1.filter(pl.col("entity_id").is_in(sing.implode()))
def show(title, df, n, seed):
    print(f"\n==== {title}"); 
    for r in df.sample(n, seed=seed).iter_rows(named=True): print(f"  {r['entity_id']:14s} | {r['country']:5s} | {r['business_name']} | {r['business_address']}")
show("orphans", orph, 12, 1); show("singleton S1", S, 8, 2)
# for singletons: look for R with the same (lowercased) first distinctive name token + same house number
def tok(s): 
    t=[w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w)>3]; return t[0] if t else None
Rl = R.with_columns(pl.col("business_name").str.to_lowercase().alias("nl"), pl.col("business_address").str.to_lowercase().alias("al"))
for r in S.sample(6, seed=5).iter_rows(named=True):
    t = tok(r["business_name"]); print(f"\n## SINGLETON {r['entity_id']} | {r['business_name']} | {r['business_address']}   [search '{t}']")
    hits = Rl.filter(pl.col("nl").str.contains(t, literal=True) & (pl.col("country")==r["country"])).head(8)
    for h in hits.iter_rows(named=True): print(f"     {h['entity_id']:14s} {'MATCHED' if h['entity_id'] in set() else ''} | {h['business_name']} | {h['business_address']}")
# field stats per source
for name, df in [("S1", s1), ("S2", read("train",2)), ("S3", read("train",3))]:
    n = df.height
    st = df.select(
        (pl.col("business_address").is_null() | (pl.col("business_address")=="None")).mean().alias("addr_missing"),
        pl.col("business_name").is_null().mean().alias("name_null"),
        pl.col("business_name").str.contains(r"[ऀ-෿]").mean().alias("indic_name"),
        pl.col("business_address").str.contains(r"[ऀ-෿]").mean().alias("indic_addr"),
        pl.col("business_name").str.contains(r"(?i)\b(f/k/a|d/b/a|dba|aka|t/a|formerly)\b").mean().alias("alias"),
        pl.col("business_name").str.contains(r"(?i)\.(com|in|net|org|co)\b|^@|www\.").mean().alias("web_handle"),
        pl.col("business_name").str.contains(r"[À-ÿ]").mean().alias("accent"),
        pl.col("business_address").str.contains(r"\b\d{6}\b").mean().alias("pin6"),
        pl.col("business_address").str.contains(r"\b\d{5}\b").mean().alias("zip5"),
        pl.col("business_address").str.contains(r"(?i)\b(near|opp|opposite|behind|beside)\b").mean().alias("landmark"),
    )
    print(name, n, st.rows(named=True)[0])
