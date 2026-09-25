# Kaggle EDA: (1) candidate-search recall at K on a 30k-S1 sample against the FULL S2/S3 pool,
# (2) per-arm unique recall, (3) twin prevalence for matched S1s, (4) noise stats per source.
# Memory-bounded: S1 processed in chunks; posting lists capped by document frequency.
import subprocess, sys, glob, os, time, gc
LOCAL = not os.path.exists("/kaggle")
if not LOCAL: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "polars>=1.30"], check=False)
import polars as pl, numpy as np
T0 = time.time(); WD = "/kaggle/working" if not LOCAL else "."; OUT = open(f"{WD}/results.txt", "w")
def rc(h, K):
    return (h["rk"].fill_null(10**9) <= K).mean() if h.height else float("nan")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
IN = os.path.dirname(glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)[0]) if not LOCAL else "../../research/eda/cache"
rd = lambda n: pl.read_parquet(f"{IN}/{n}.parquet")
s1 = rd("train_s1"); R = pl.concat([rd("train_s2"), rd("train_s3")])
if LOCAL: s1 = s1.head(100000); R = R.head(400000)
R = R.with_row_index("rid")
g = rd("gt_rows").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
    .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
log(f"loaded S1={s1.height} R={R.height} pairs={g.height}  {time.time()-T0:.0f}s")

LEG = r"\b(inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|lp|llp|pllc|pc|the|and|of|l|c|p|a|dba|fka|formerly|sarl|sas|sa|eurl|sasu|sci|ei)\b"
def feats(df, idcol):
    clean = lambda c: pl.col(c).fill_null("").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
    nm = clean("business_name").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEG, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    ad = clean("business_address").str.replace(r"^none$", "")
    d = df.select(pl.col(idcol).alias("id"), pl.col("country").cast(pl.Categorical),
                  nm.str.extract_all(r"[a-z0-9]{2,}").list.unique(maintain_order=True).alias("nw"),
                  nm.str.replace_all(" ", "").alias("ncat"),
                  ad.str.extract_all(r"[a-z]{3,}").alias("aw"),
                  ad.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != "")).alias("ad"))
    h4 = pl.col("nw").list.head(4)
    prs = [pl.when(h4.list.len() > j).then(pl.lit("P:") + pl.min_horizontal(h4.list.get(i, null_on_oob=True), h4.list.get(j, null_on_oob=True)) + "|" +
                                          pl.max_horizontal(h4.list.get(i, null_on_oob=True), h4.list.get(j, null_on_oob=True))) for i in range(4) for j in range(i+1, 4)]
    arms = {
      "tok":  pl.concat_list(pl.col("nw").list.eval(pl.lit("n:")+pl.element()), pl.col("aw").list.eval(pl.lit("a:")+pl.element()), pl.col("ad").list.eval(pl.lit("#:")+pl.element())),
      "npair": pl.concat_list(*prs),
      "nkey": pl.concat_list(pl.lit("N:")+pl.col("nw").list.sort().list.join(" "), pl.lit("C:")+pl.col("ncat")),
      "num_street": pl.concat_list(pl.col("ad").list.head(2).list.eval(pl.element()).list.eval(pl.lit("S:")+pl.element()).list.first() + "|" + pl.col("aw").list.first()),
    }
    out = []
    for arm, e in arms.items():
        t = d.select("id", "country", e.list.drop_nulls().list.unique().alias("t")).explode("t").drop_nulls("t") \
             .filter(pl.col("t").str.len_chars() > 3).select("id", "country", pl.col("t").hash().alias("h"), pl.lit(arm).cast(pl.Categorical).alias("arm"))
        out.append(t)
    return pl.concat(out)

Rt = feats(R, "rid"); log(f"R tokens {Rt.height}  {time.time()-T0:.0f}s")
dfc = Rt.group_by(["country", "h"]).len().rename({"len": "df"})
N = R.group_by(pl.col("country").cast(pl.Categorical)).len().rename({"len": "N"})
dfc = dfc.join(N, on="country").with_columns((pl.col("N")/pl.col("df")).log().alias("idf")).filter(pl.col("df") <= 5000).select("country", "h", "idf")
Rt = Rt.join(dfc.select("country", "h"), on=["country", "h"])          # prune common tokens from postings
log(f"R postings after df<=5000: {Rt.height}")
gc.collect()

samp = s1.sample(2000 if LOCAL else 30000, seed=11).with_row_index("sid")
St = feats(samp, "sid").join(dfc, on=["country", "h"])
res = []
for c0 in range(0, samp.height, 1500):
    ch = St.filter((pl.col("id") >= c0) & (pl.col("id") < c0 + 1500))
    j = ch.join(Rt.select("id", "country", "h"), on=["country", "h"], suffix="_r")
    agg = j.group_by(["id", "id_r"]).agg(pl.col("idf").sum().alias("sc"),
            *[(pl.col("arm") == a).any().alias(f"hit_{a}") for a in ["tok", "npair", "nkey", "num_street"]])
    agg = agg.with_columns(pl.col("sc").rank("ordinal", descending=True).over("id").alias("rk")).filter(pl.col("rk") <= 200)
    res.append(agg)
cand = pl.concat(res); del res, St; gc.collect()
log(f"candidates kept (<=200/S1): {cand.height}  {time.time()-T0:.0f}s")
cand = cand.join(samp.select(pl.col("sid").alias("id"), pl.col("entity_id").alias("s1")), on="id") \
           .join(R.select(pl.col("rid").alias("id_r"), pl.col("entity_id").alias("r")), on="id_r")
Rinfo = R.select(pl.col("entity_id").alias("r"),
                 pl.col("business_name").str.contains(r"[ऀ-෿]").alias("indic"),
                 (pl.col("business_address").is_null() | (pl.col("business_address") == "None")).alias("noaddr"),
                 pl.col("business_name").str.contains(r"(?i)\.(com|in|net|org)\b|^@|www\.").alias("web"),
                 pl.col("business_name").str.contains(r"(?i)f/k/a|d/b/a|t/a|formerly").alias("alias"))
P = g.filter(pl.col("s1").is_in(samp["entity_id"].implode())).join(Rinfo, on="r")
hit = P.join(cand.select("s1", "r", "rk", "hit_tok", "hit_npair", "hit_nkey", "hit_num_street"), on=["s1", "r"], how="left") \
       .join(samp.select(pl.col("entity_id").alias("s1"), "country"), on="s1")
log("\n=== recall@K (pairs) overall and by slice ===")
for K in (5, 10, 20, 30, 50, 100, 200):
    row = [f"K={K:3d} all={rc(hit, K):.4f}"]
    for sl in ["indic", "noaddr", "web", "alias"]:
        h = hit.filter(pl.col(sl)); row.append(f"{sl}={rc(h, K):.3f}(n={h.height})")
    for c in ["US", "India"]:
        h = hit.filter(pl.col("country") == c); row.append(f"{c}={rc(h, K):.4f}")
    log("  ".join(row))
log("\n=== per-arm hit rate among positives found, and unique contribution ===")
f = hit.filter(pl.col("rk").is_not_null())
for a in ["tok", "npair", "nkey", "num_street"]:
    others = [x for x in ["hit_tok", "hit_npair", "hit_nkey", "hit_num_street"] if x != f"hit_{a}"]
    uniq = f.filter(pl.col(f"hit_{a}") & ~pl.any_horizontal(*others)).height
    log(f"  {a:10s} hits {f[f'hit_{a}'].mean():.3f}   unique {uniq}")
log("\n=== entity-level: share of S1 whose ALL matches are in top-K ===")
ent = hit.group_by("s1").agg((pl.col("rk").fill_null(10**9) <= 30).all().alias("all30"), (pl.col("rk").fill_null(10**9) <= 30).mean().alias("frac30"))
log(f"  all matches in top30: {ent['all30'].mean():.4f}   mean frac: {ent['frac30'].mean():.4f}")

# twin prevalence among MATCHED S1: a non-matching candidate in top 30 with a high name+address score
from rapidfuzz import fuzz
import unicodedata, re
def nz(s):
    s = unicodedata.normalize("NFKD", s or ""); s = "".join(ch for ch in s if not unicodedata.combining(ch)); return " ".join(re.findall(r"[a-z0-9]+", s.lower()))
top = cand.filter(pl.col("rk") <= 30).join(samp.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1") \
          .join(R.select(pl.col("entity_id").alias("r"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")), on="r") \
          .join(g.with_columns(pl.lit(1).alias("y")), on=["s1", "r"], how="left").with_columns(pl.col("y").fill_null(0))
top = top.with_columns(pl.struct("n1", "n2").map_elements(lambda x: fuzz.token_set_ratio(nz(x["n1"]), nz(x["n2"])), return_dtype=pl.Float64).alias("ns"),
                       pl.struct("a1", "a2").map_elements(lambda x: fuzz.token_set_ratio(nz(x["a1"]), nz(x["a2"])), return_dtype=pl.Float64).alias("as_"))
tw = top.group_by("s1").agg(((pl.col("y") == 0) & (pl.col("ns") >= 80) & (pl.col("as_") >= 70)).any().alias("has_twin"))
matched = set(g["s1"].unique().to_list())
tw = tw.with_columns(pl.col("s1").is_in(list(matched)).alias("matched"))
log("\n=== twin prevalence (non-matching lookalike in top30) ===")
log(tw.group_by("matched").agg(pl.col("has_twin").mean(), pl.len()))
log("\n=== high-sim precision in top30: P(y=1 | ns>=80 & as>=70) ===", top.filter((pl.col("ns") >= 80) & (pl.col("as_") >= 70))["y"].mean())
log(f"\nDONE {time.time()-T0:.0f}s")
