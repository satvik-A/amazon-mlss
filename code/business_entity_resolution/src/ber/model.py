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
