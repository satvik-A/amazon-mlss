"""Matcher glue shared by the training and submission jobs: features for a pool, the S1-level has-match head,
and applying a saved decision config (blocking policy + decision rule) to scored pairs.

decision.json keys: policy {alpha, G, gate, beta}, rule (rank-threshold | expected-F | expected-F + exclusivity |
expected-F + has-head), t1, t2, gamma, features [..], head_features [..].
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import blocking as B
from .decide import exclusive_posterior, expected_f_decode, rank_threshold, source_caps
from .features import feature_columns, pair_features

RULES = ("rank-threshold", "expected-F", "expected-F + exclusivity", "expected-F + has-head")


def prune_pool(pool: pl.DataFrame, policy: dict) -> pl.DataFrame:
    """pool: annotated candidates [s1, r, sc, gid, rel, a4, rev_margin, ...] -> rows kept by the blocking policy."""
    beta = policy.get("beta")
    beta = None if beta is None or beta < 0 else beta
    A = pool.select(pl.col("s1").alias("id"), pl.col("r").alias("id_r"), "sc", "gid", "rel", "a4", "rev_margin")
    kept = B.prune(A, alpha=policy.get("alpha", 0.0), G=policy.get("G", 99), gate=policy.get("gate", False), beta=beta)
    return pool.join(kept.rename({"id": "s1", "id_r": "r"}), on=["s1", "r"], how="semi")


def pool_features(pool: pl.DataFrame, S1: pl.DataFrame, R: pl.DataFrame, nz, chunk: int = 5_000_000) -> pl.DataFrame:
    """Features for every pool row. S1/R: raw records (entity_id, business_name, business_address, country).
    Rows are processed in S1-complete slices so the per-S1 context features see the whole candidate list."""
    s1s = pool["s1"].unique().sort()
    per = max(1, int(len(s1s) * chunk / max(pool.height, 1)))
    out = []
    for i in range(0, len(s1s), per):
        ids = s1s.slice(i, per)
        P = pool.filter(pl.col("s1").is_in(ids.implode()))
        QN = nz.transform(S1.filter(pl.col("entity_id").is_in(ids.implode())))
        RN = nz.transform(R.filter(pl.col("entity_id").is_in(P["r"].unique().implode())))
        out.append(pair_features(P, QN, RN))
    return pl.concat(out, how="vertical_relaxed")


def s1_agg(d: pl.DataFrame, p: str = "p") -> pl.DataFrame:
    """S1-level inputs for the has-match head (from scored pairs + features)."""
    d = d.sort("r").with_columns(pl.col(p).rank("ordinal", descending=True).over("s1").alias("_prk"))
    return d.group_by("s1").agg(
        pl.col(p).max().alias("p1"), pl.col(p).filter(pl.col("_prk") == 2).first().fill_null(0).alias("p2"),
        pl.col(p).sum().alias("psum"), (pl.col(p) > 0.5).sum().alias("n05"), pl.len().alias("nc"),
        pl.col("nm_tset").max().alias("nm_max"), pl.col("num_jacc").max().alias("num_max"),
        (pl.col("house_rel") == 0).sum().alias("n_house_eq"), pl.col("noaddr_s").first().alias("noaddr"),
        pl.col("n_core_s").first().alias("ncore"), pl.col("rev_margin").max().alias("rev_max"))


HEAD_FEATURES = ["p1", "p2", "psum", "n05", "nc", "nm_max", "num_max", "n_house_eq", "noaddr", "ncore", "rev_max"]


def efd(d: pl.DataFrame, gamma: float = 1.0, col: str = "p", has: pl.DataFrame | None = None) -> pl.DataFrame:
    x = d.with_columns((pl.col(col) ** gamma).alias("pp"))
    return source_caps(expected_f_decode(x, p="pp", has=has), x, p="pp")


def decide(d: pl.DataFrame, cfg: dict, head=None) -> pl.DataFrame:
    """d: scored pairs [s1, r, p, + features]; cfg: decision.json; head: LightGBM Booster for the has-head rule."""
    rule, g = cfg["rule"], cfg.get("gamma", 1.0)
    if rule == "rank-threshold":
        return source_caps(rank_threshold(d, cfg["t1"], cfg["t2"]), d)
    if rule == "expected-F":
        return efd(d, g)
    if rule == "expected-F + exclusivity":
        return efd(exclusive_posterior(d).with_columns(pl.col("q").alias("p")), g)
    if rule == "expected-F + has-head":
        H = s1_agg(d)
        H = H.with_columns(pl.Series("h", head.predict(H.select(HEAD_FEATURES).to_numpy().astype(np.float32))))
        return efd(d, g, has=H.select("s1", "h"))
    raise ValueError(rule)


def X(d: pl.DataFrame, fc: list[str]) -> np.ndarray:
    return d.select([pl.col(c).cast(pl.Float32) for c in fc]).to_numpy()


__all__ = ["RULES", "prune_pool", "pool_features", "s1_agg", "HEAD_FEATURES", "efd", "decide", "X", "feature_columns"]


# ---- context features computed on the WHOLE country pool / S1 table (identically in training and at submission) ------
def claim_stats(path: str) -> pl.DataFrame:
    """Per record, from a whole country pool FILE with streaming (never loads the pool): how many S1s retrieved it
    (n_s1_for_r) and how many score it within 5% of its best S1 (n_claim). A no-address record with a common name ties
    across every same-name S1; the cross-encoder cannot see that."""
    lf = pl.scan_parquet(path).select("r", "sc")
    best = lf.group_by("r").agg(pl.col("sc").max().alias("_b"), pl.len().cast(pl.Int32).alias("n_s1_for_r"))
    cl = lf.join(best.select("r", "_b"), on="r").filter(pl.col("sc") >= 0.95 * pl.col("_b")).group_by("r").agg(pl.len().cast(pl.Int32).alias("n_claim"))
    return best.join(cl, on="r", how="left").select("r", "n_s1_for_r", pl.col("n_claim").fill_null(0)).collect(engine="streaming")


def claim_features(pool: pl.DataFrame) -> pl.DataFrame:
    """In-memory version of claim_stats (pool = the whole country pool)."""
    if "n_claim" in pool.columns:
        return pool
    best = pl.col("sc").max().over("r")
    return pool.with_columns(pl.len().over("r").cast(pl.Int32).alias("n_s1_for_r"),
                             (pl.col("sc") >= 0.95 * best).cast(pl.Int32).sum().over("r").alias("n_claim"))


def s1_name_freq(S_all: pl.DataFrame, nz, chunk: int = 1_000_000) -> pl.DataFrame:
    """[s1, s1_name_freq]: number of S1s of the same country with exactly the same core-name set."""
    N = pl.concat([nz.transform(S_all.slice(i, chunk)).select("entity_id", "country", pl.col("core").list.sort().list.join(" ").alias("k"))
                   for i in range(0, S_all.height, chunk)])
    return N.with_columns(pl.len().over(["country", "k"]).cast(pl.Int32).alias("s1_name_freq")).select(pl.col("entity_id").alias("s1"), "s1_name_freq")


# ---- deterministic candidate cut-offs (blocking rules on similarity features; thresholds tuned on train) ------------
# Ordered OR-rules from a greedy search on the v5 train pool (6k S1, tuned on one half, checked on the other):
# first n rules -> held-out share of in-pool positives kept: n=8: 8.4 cands/S1, 0.984 ; n=10: 11.9, 0.993 ;
# n=11: 13.7, 0.9955 ; n=14: 26.4, 0.9975 (no cut-off: 91.5 cands/S1).
def _cut_rules():
    c = pl.col
    return [
        (c("house_rel") == 0) & (c("sk_jacc") >= 0.5),
        c("sc_rank") <= 1,
        (c("nm_tset") >= 90) & (c("ad_tset") >= 60),
        (c("nm_tset") >= 90) & (c("noaddr_r") == 1),
        c("sc_rank") <= 3,
        (c("nm_tset") >= 80) & (c("noaddr_r") == 1),
        (c("nm_tset") >= 80) & (c("ad_tset") >= 60),
        c("sc_rank") <= 5,
        (c("nm_tset") >= 70) & (c("noaddr_r") == 1),
        c("num_rel") == 0,
        (c("nm_tset") >= 70) & (c("ad_tset") >= 60),
        c("num_rel").is_in([0, 1, 2]) & (c("sk_jacc") >= 0.5),
        c("house_rel") == 0,
        c("num_rel").is_in([0, 1, 2]),
    ]


def cutoff_mask(n: int) -> pl.Expr:
    """True for candidates kept by the first n cut-off rules (feature frame from pool_features)."""
    rules = _cut_rules()[:n]
    e = rules[0]
    for r in rules[1:]:
        e = e | r
    return e.fill_null(False)


def apply_policy(F: pl.DataFrame, policy: dict) -> pl.DataFrame:
    """Blocking policy on a FEATURE frame: {'cutoff': n} (feature rules) or a group policy (prune_pool)."""
    if policy.get("cutoff"):
        return F.filter(cutoff_mask(int(policy["cutoff"])))
    return prune_pool(F, policy)
