import polars as pl, os
D = "student_resource/dataset"
C = "research/eda/cache"; os.makedirs(C, exist_ok=True)
def read(split, s):
    p = f"{C}/{split}_s{s}.parquet"
    if not os.path.exists(p):
        pl.read_csv(f"{D}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None,
                    infer_schema=False, missing_utf8_is_empty_string=False).write_parquet(p)
    return pl.read_parquet(p)
def gt():
    p = f"{C}/gt_pairs.parquet"
    if not os.path.exists(p):
        g = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
        g.write_parquet(f"{C}/gt_rows.parquet")
        g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
         .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")) \
         .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"}).write_parquet(p)
    return pl.read_parquet(f"{C}/gt_rows.parquet"), pl.read_parquet(p)
