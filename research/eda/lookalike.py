import polars as pl, sys, re, unicodedata, numpy as np; sys.path.insert(0, "research/eda"); from load import *
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250)
rows, pairs = gt(); s1 = read("train",1); R = pl.read_parquet(f"{C}/train_R.parquet")
lab = dict(zip(pairs["r"].to_list(), pairs["s1"].to_list()))
def norm(s):
    s = unicodedata.normalize("NFKD", s or ""); s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))
nm = rows.with_columns(pl.col("matched_entity_ids").fill_null("").str.len_chars().alias("L"))
sing = set(nm.filter(pl.col("L")==0)["source1_entity_id"].to_list())
rng = np.random.default_rng(0)
samp = s1.filter(pl.col("entity_id").is_in(list(sing))).sample(1500, seed=1).vstack(s1.filter(~pl.col("entity_id").is_in(list(sing))).sample(1500, seed=1))
tokexpr = lambda c: pl.col(c).str.to_lowercase().str.extract_all(r"[a-z]{4,}").list.unique()
st = samp.with_columns(tokexpr("business_address").alias("t")).explode("t").drop_nulls("t")
Rt = R.with_columns(tokexpr("business_address").alias("t")).select("entity_id","country","t").explode("t").drop_nulls("t")
df = Rt.group_by(["country","t"]).len()
common = df.filter(pl.col("len")>30000)            # drop state/city-level tokens
st = st.join(common, on=["country","t"], how="anti")
cand = st.select(pl.col("entity_id").alias("s1"),"country","t").join(Rt, on=["country","t"]).group_by(["s1","entity_id"]).len()
ntok = st.group_by(pl.col("entity_id").alias("s1")).len().rename({"len":"nt"})
cand = cand.join(ntok, on="s1").filter(pl.col("len") >= pl.min_horizontal(pl.col("nt"), 2))
print("sample S1:", samp.height, "cand pairs:", cand.height)
cand = cand.join(samp.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1") \
           .join(R.select("entity_id", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="entity_id")
out = []
for r in cand.iter_rows(named=True):
    n1, n2, a1, a2 = norm(r["n1"]), norm(r["n2"]), norm(r["a1"]), norm(r["a2"])
    num1 = set(re.findall(r"\d+", a1)); num2 = set(re.findall(r"\d+", a2))
    out.append(dict(s1=r["s1"], r=r["entity_id"], y=int(lab.get(r["entity_id"])==r["s1"]), owner=("orphan" if r["entity_id"] not in lab else ("self" if lab[r["entity_id"]]==r["s1"] else "otherS1")),
        sing=r["s1"] in sing, ns=fuzz.token_set_ratio(n1,n2), as_=fuzz.token_set_ratio(a1,a2),
        num_eq=int(num1==num2), num_j=len(num1&num2)/max(1,len(num1|num2)), n1=r["n1"], a1=r["a1"], n2=r["n2"], a2=r["a2"]))
o = pl.DataFrame(out); o.write_parquet(f"{C}/lookalike_sample.parquet")
pos_found = o.filter(pl.col("y")==1).height; pos_total = pairs.filter(pl.col("s1").is_in(samp["entity_id"].implode())).height
print(f"address-token blocking recall on sample: {pos_found}/{pos_total} = {pos_found/pos_total:.3f}")
o = o.with_columns(pl.when(pl.col("ns")>=80).then(pl.lit("name>=80")).otherwise(pl.lit("name<80")).alias("nb"),
                   pl.when(pl.col("as_")>=80).then(pl.lit("addr>=80")).otherwise(pl.lit("addr<80")).alias("ab"))
print(o.group_by(["nb","ab","owner"]).len().sort(["nb","ab","owner"]))
print("\nhigh-sim negatives (name>=80 & addr>=80): number-set equal rate by label")
hs = o.filter((pl.col("ns")>=80)&(pl.col("as_")>=80))
print(hs.group_by(["y","owner"]).agg(pl.len(), pl.col("num_eq").mean(), pl.col("num_j").mean()))
print("\nexamples of high-sim negatives:")
for r in hs.filter(pl.col("y")==0).sample(min(25, hs.filter(pl.col("y")==0).height), seed=0).iter_rows(named=True):
    print(f"  [{r['owner']:7s} sing={r['sing']}] {r['n1']} | {r['a1']}\n        vs {r['n2']} | {r['a2']}")
# singletons: fraction having a high-sim lookalike
g = o.filter(pl.col("sing")).group_by("s1").agg(((pl.col("ns")>=80)&(pl.col("as_")>=70)).any().alias("has_look"))
print("\nsingletons with a high-sim lookalike:", g["has_look"].mean(), "of", g.height, "(that were retrieved)")
