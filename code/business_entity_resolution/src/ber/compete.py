"""Competition + sibling features (pass 4) from level-1 scores of EVERY S1 of a country.

Competition: a record belongs to at most one S1, so how strongly OTHER S1s claim it matters (no-address copies with a
generic name are claimed by dozens of same-name S1s). Sibling: copies of one S1 in the same source share the generator's
noise (S2-S2 address similarity 76 vs 40 to the S1, 22% identical noisy addresses), so a candidate that looks like the
S1's confident same-source candidates is likely a copy too.
India train C, level-1 base: one owner 0.9682 -> competition 0.9699 -> + siblings 0.9703."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz

COMP = ["rk_r", "n_r", "so_other_r", "q05", "share_r", "rk_s", "n_s", "pmax_s", "psum_s"]
SIB = ["sib_a", "sib_n", "sib_exa", "n_sib"]
E = 1e-6


def comp_features(D: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    """D: [s1, r, <p>] for ALL S1s of a country (competitors included). Adds COMP columns."""
    o = pl.col(p).clip(0, 1 - E) / (1 - pl.col(p)).clip(E, 1)
    D = D.with_columns(o.alias("_o")).with_columns(
        pl.len().over("r").alias("n_r"), pl.col("_o").sum().over("r").alias("_so"),
        pl.col(p).rank("ordinal", descending=True).over("r").alias("rk_r"),
        pl.col(p).rank("ordinal", descending=True).over("s1").alias("rk_s"), pl.len().over("s1").alias("n_s"),
        pl.col(p).max().over("s1").alias("pmax_s"), pl.col(p).sum().over("s1").alias("psum_s"))
    return D.with_columns((pl.col("_o") / pl.col("_so").clip(E)).alias("share_r"), (pl.col("_so") - pl.col("_o")).alias("so_other_r"),
                          (pl.col("_o") / (0.5 + pl.col("_so"))).alias("q05")).drop("_o", "_so")


def _sib(rows) -> np.ndarray:
    out = np.full((len(rows), 4), -1, dtype=np.float32)
    for i, (r, rn, ra, cr, crn, cra) in enumerate(rows):
        if not cr:
            continue
        best_a = best_n = -1.0; ex_a = 0; k = 0
        for r2, n2, a2 in zip(cr, crn, cra):
            if r2 == r:
                continue
            k += 1
            if ra and a2:
                best_a = max(best_a, fuzz.ratio(ra, a2)); ex_a |= (ra == a2)
            best_n = max(best_n, fuzz.ratio(rn, n2))
        out[i] = (best_a, best_n, ex_a if k else -1, k)
    return out


def sib_features(D: pl.DataFrame, R: pl.DataFrame, p: str = "p1", thr: float = 0.5, chunk: int = 1_000_000) -> pl.DataFrame:
    """D: [s1, r, <p>]; R: raw records [entity_id, business_name, business_address]. Adds SIB columns: best address / name
    ratio and exact-address flag against the S1's confident (p > thr) candidates of the same source, and their count."""
    T = R.select(pl.col("entity_id").alias("r"), pl.col("business_name").fill_null("").str.to_lowercase().alias("_rn"),
                 pl.col("business_address").fill_null("").str.to_lowercase().alias("_ra"))
    D = D.join(T, on="r", how="left").with_columns(pl.col("r").str.slice(0, 2).alias("_src"), pl.col("_rn").fill_null(""), pl.col("_ra").fill_null(""))
    conf = D.filter(pl.col(p) > thr).group_by("s1", "_src").agg(pl.col("r").alias("_cr"), pl.col("_rn").alias("_crn"), pl.col("_ra").alias("_cra"))
    D = D.join(conf, on=["s1", "_src"], how="left")
    cols = ["r", "_rn", "_ra", "_cr", "_crn", "_cra"]
    F = np.vstack([_sib(D.slice(i, chunk).select(cols).rows()) for i in range(0, D.height, chunk)]) if D.height else np.zeros((0, 4), np.float32)
    return D.drop(cols[1:] + ["_src"]).with_columns(*[pl.Series(n, F[:, j]) for j, n in enumerate(SIB)])
