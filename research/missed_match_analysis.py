"""Missed-match analysis, India train holdout C, pass-2 level-1 scores (er2-global-india scored_India.parquet)."""
import sys; sys.path.insert(0, "code/business_entity_resolution/src")
import polars as pl
from ber.metrics import macro_f05
from ber.decide import rank_threshold, source_caps
from ber.translit import INDIC_RE
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
C = "research/eda/cache"
D = pl.read_parquet("kaggle2/global/kout/india/scored_India.parquet")
S = pl.read_parquet(f"{C}/train_s1.parquet").filter(pl.col("country") == "India")
sC = S.filter(pl.col("entity_id").hash(11) % 100 >= 90)["entity_id"]
gt = pl.read_parquet(f"{C}/gt_rows.parquet").filter(pl.col("source1_entity_id").is_in(sC.implode())) \
       .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
       .filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
rs = D.filter(pl.col("s1").is_in(sC.implode()))["r"].unique()
V = D.filter(pl.col("r").is_in(rs.implode()))                      # closure: C S1s + competitors
sel = source_caps(rank_threshold(V, 0.75, 0.65), V).join(V.select("s1", "r", "p"), on=["s1", "r"])
sel = sel.sort(["r", "p", "s1"], descending=[False, True, False]).unique(subset=["r"], keep="first")
sel = sel.filter(pl.col("s1").is_in(sC.implode()))
base = macro_f05(sel.select("s1", "r"), gt, sC)
print(f"C S1 {len(sC)}  true pairs {gt.height}  BASE F {base['f05']:.5f}  P {base['pair_precision']:.4f} R {base['pair_recall']:.4f} singleton_f {base['singleton_f']:.4f} nonsingleton_f {base['nonsingleton_f']:.4f}")

# record text + source
R = pl.concat([pl.read_parquet(f"{C}/train_s{k}.parquet").filter(pl.col("country") == "India").with_columns(pl.lit(f"s{k}").alias("src")) for k in (2, 3)])
allr = pl.concat([gt["r"], sel["r"]]).unique()
R = R.filter(pl.col("entity_id").is_in(allr.implode())).select(pl.col("entity_id").alias("r"), pl.col("business_name").alias("rn"), pl.col("business_address").alias("ra"), "src")
S1t = S.filter(pl.col("entity_id").is_in(sC.implode())).select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa"))
Dc = D.filter(pl.col("s1").is_in(sC.implode()))
rank = Dc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("rk"))
comp = V.group_by("r").agg(pl.col("p").max().alias("pmax_any"), pl.len().alias("n_s1_claim"))
T = gt.join(rank.select("s1", "r", "p", "rk", "noaddr_r"), on=["s1", "r"], how="left") \
      .join(sel.select("s1", "r").with_columns(pl.lit(True).alias("chosen")), on=["s1", "r"], how="left") \
      .join(R, on="r", how="left").join(comp, on="r", how="left") \
      .with_columns(pl.col("chosen").fill_null(False), pl.col("p").is_not_null().alias("in_pool"),
                    pl.col("rn").fill_null("").str.contains(INDIC_RE).alias("indic"),
                    pl.len().over("s1").alias("nt"))
T = T.with_columns(pl.when(pl.col("chosen")).then(pl.lit("0 found"))
                   .when(~pl.col("in_pool")).then(pl.lit("1 not in candidates"))
                   .when(pl.col("pmax_any") > pl.col("p") + 1e-9).then(pl.lit("2 in pool, another S1 scored higher"))
                   .when(pl.col("p") >= 0.5).then(pl.lit("3 in pool, p>=0.5 but rank/caps cut"))
                   .when(pl.col("p") >= 0.1).then(pl.lit("4 in pool, p 0.1-0.5"))
                   .otherwise(pl.lit("5 in pool, p<0.1")).alias("why"))
print("\n== where true pairs go (C) =="); print(T.group_by("why").agg(pl.len().alias("n"), (pl.len() / T.height).alias("share")).sort("why"))
print("\n== by script of the record =="); print(T.group_by("indic", "why").len().pivot(on="why", index="indic", values="len").sort("indic"))
print(T.group_by("indic").agg(pl.len(), pl.col("chosen").mean().alias("recall"), pl.col("in_pool").mean().alias("pool_recall")))
print("\n== by source / no-address =="); print(T.group_by("src", "noaddr_r").agg(pl.len(), pl.col("chosen").mean().alias("recall"), pl.col("in_pool").mean().alias("pool")).sort("src", "noaddr_r"))
print("\n== by S1 cluster size =="); print(T.with_columns(pl.col("nt").clip(1, 8)).group_by("nt").agg(pl.len(), pl.col("chosen").mean().alias("recall")).sort("nt"))

T.write_parquet("/private/tmp/claude-501/-Users-aderlasatvik-Downloads-amazon-ml/a1672bb9-d17b-468e-bbe6-559ef7719ba5/scratchpad/miss_T.parquet")
# oracle gains: add missed true pairs of one bucket, recompute F
def gain(mask, name):
    add = T.filter(mask & ~pl.col("chosen")).select("s1", "r")
    f = macro_f05(pl.concat([sel.select("s1", "r"), add]), gt, sC)["f05"]
    print(f"  oracle add {name:45s} {add.height:7d} pairs  F {f:.5f}  ({f-base['f05']:+.5f})")
print("\n== oracle: recover missed pairs of a bucket ==")
for w in T["why"].unique().sort().to_list()[1:]: gain(pl.col("why") == w, w)
gain(pl.col("in_pool"), "ALL in-pool misses"); gain(pl.lit(True), "ALL misses")
gain(pl.col("indic"), "Indic-script records"); gain(pl.col("noaddr_r").fill_null(0) == 1, "no-address records")
fp = sel.join(gt, on=["s1", "r"], how="anti")
f = macro_f05(sel.join(fp, on=["s1", "r"], how="anti").select("s1", "r"), gt, sC)["f05"]
print(f"  oracle drop ALL false positives {fp.height:7d} pairs  F {f:.5f}  ({f-base['f05']:+.5f})")
fps = fp.join(gt.select("s1").unique(), on="s1", how="anti")
f = macro_f05(sel.join(fps, on=["s1", "r"], how="anti").select("s1", "r"), gt, sC)["f05"]
print(f"  oracle drop FPs on singleton S1s {fps.height:7d} pairs  F {f:.5f}  ({f-base['f05']:+.5f})")

fp.join(R, on="r", how="left").join(S1t, on="s1").write_parquet("/private/tmp/claude-501/-Users-aderlasatvik-Downloads-amazon-ml/a1672bb9-d17b-468e-bbe6-559ef7719ba5/scratchpad/miss_FP.parquet")
