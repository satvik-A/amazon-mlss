"""End-to-end candidate generation per country (same logic as the v5 sweep job, packaged for full train/test runs).

candidates(S, R, S_all, nz)  ->  annotated pool [s1, r, sc, prk, xrk, a0.., exp, gid, rel, rsc, rbest, rev_margin]
  1. normalise the S2/S3 pool, build the inverted index, query the S1s (primary + auxiliary searches)
  2. sibling expansion (records sharing a signature with a retrieved record), group ids
  3. number relation S1 vs record
  4. reverse preference: which S1 of the whole country each record prefers
     (FULL mode: from the forward candidates; SAMPLE mode: index over ALL S1s. The definitions differ, so train the
      matcher on FULL-mode pools when the test pool is built in FULL mode.)
Pruning to the final candidate set is blocking.prune(...) with the policy chosen on the train frontier.
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl

from . import blocking as B


def norm_chunks(nz, df: pl.DataFrame, chunk: int = 1_000_000) -> pl.DataFrame:
    return pl.concat([nz.transform(df.slice(i, chunk)) for i in range(0, df.height, chunk)])


def candidates(S_raw: pl.DataFrame, R_raw: pl.DataFrame, S_all_raw: pl.DataFrame, nz, log=print, keep_prk: int = 60,
               rev_cap: int = 3) -> pl.DataFrame:
    """S_raw: S1s to query; R_raw: the country's S2/S3 records; S_all_raw: ALL S1s of the country (reverse lookups).
    Deterministic: inputs are put in entity_id order, so row ids (rank tie-breaks) do not depend on input order."""
    S_raw, R_raw, S_all_raw = (x.sort("entity_id") for x in (S_raw, R_raw, S_all_raw))
    R = norm_chunks(nz, R_raw).with_columns(pl.Series("id", np.arange(R_raw.height, dtype=np.uint32)))
    idx = B.Index(R)
    Q = norm_chunks(nz, S_raw).with_columns(pl.Series("id", np.arange(S_raw.height, dtype=np.uint32)))
    cand = idx.query(Q)
    sig = B.signatures(R)
    ex = B.expand(cand, sig)
    del idx; gc.collect()
    cand = pl.concat([cand, ex.with_columns(pl.lit(0.0, dtype=pl.Float32).alias("sc"), pl.lit(None, dtype=pl.UInt32).alias("prk"),
                                            pl.lit(None, dtype=pl.UInt32).alias("xrk"), *[pl.lit(False).alias(f"a{k}") for k in B.ARMS])
                      .select(cand.columns)], how="vertical_relaxed")
    cand = cand.with_columns((pl.col("prk").is_null() & pl.col("xrk").is_null()).alias("exp"))
    cand = cand.filter((pl.col("prk") <= keep_prk) | pl.col("xrk").is_not_null() | pl.col("exp"))
    cand = cand.join(sig.rename({"id": "id_r"}), on="id_r", how="left").with_columns(
        pl.when(pl.col("kind") == "none").then(pl.col("id_r").cast(pl.UInt64) + (1 << 62)).otherwise(pl.col("sig")).alias("gid")).drop("sig", "kind")
    cand = cand.with_columns(pl.col("sc").max().over(["id", "gid"]).alias("_g")).with_columns(
        pl.when(pl.col("exp")).then(pl.col("_g") * 0.999).otherwise(pl.col("sc")).alias("sc")).drop("_g")
    cand = cand.join(B.number_relation(cand.select("id", "id_r"), Q, R), on=["id", "id_r"], how="left")
    same = S_all_raw.height == S_raw.height and S_all_raw["entity_id"].equals(S_raw["entity_id"])
    if same:
        # FULL mode (every S1 of the country queried): reverse preference from the forward candidates themselves.
        # Only S1s that list the record as a candidate can compete for it, so no S1 index is needed.
        cand = cand.with_columns(pl.col("sc").alias("rsc"), pl.col("sc").max().over("id_r").alias("rbest"))
    else:
        # SAMPLE mode: other S1s are not queried -> reverse lookups in an index over ALL S1s of the country
        S1n = norm_chunks(nz, S_all_raw).with_columns(pl.Series("id", np.arange(S_all_raw.height, dtype=np.uint32)))
        sidx = B.Index(S1n, arms=(0,))
        RQ = R.join(cand.select(pl.col("id_r").unique().alias("id")), on="id")
        rev = sidx.query(RQ, caps={0: rev_cap})
        del sidx, S1n, RQ; gc.collect()
        samp = S_all_raw.with_row_index("sall").join(S_raw.with_row_index("sq").select("entity_id", "sq"), on="entity_id").select("sall", "sq")
        rev = rev.select(pl.col("id").alias("id_r"), pl.col("id_r").alias("sall"), pl.col("sc").alias("rsc")).join(samp, on="sall", how="left")
        best = rev.group_by("id_r").agg(pl.col("rsc").max().alias("rbest"))
        mine = rev.filter(pl.col("sq").is_not_null()).select("id_r", pl.col("sq").alias("id"), "rsc")
        cand = cand.join(mine, on=["id", "id_r"], how="left").join(best, on="id_r", how="left")
    cand = cand.with_columns(((pl.col("rsc").fill_null(0.0) - pl.col("rbest")) / pl.col("rbest")).fill_null(0.0).alias("rev_margin"))
    ids = Q.select("id", pl.col("entity_id").alias("s1"))
    rids = R.select(pl.col("id").alias("id_r"), pl.col("entity_id").alias("r"))
    out = cand.join(ids, on="id").join(rids, on="id_r")
    log(f"  cands {out.height} ({out.height / max(Q.height, 1):.1f}/S1)")
    return out
