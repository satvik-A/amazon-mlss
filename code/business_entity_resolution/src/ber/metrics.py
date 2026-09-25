"""Exact competition metric: macro F0.5 over ALL S1 entities (singletons included:
empty prediction on an empty truth = 1.0; any prediction on an empty truth = 0.0)."""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> dict:
    """pred / truth: long pairs [s1, r]; s1_ids: every S1 in the evaluation set. Returns overall + parts."""
    b2 = beta * beta
    P = pred.select("s1", "r").unique().with_columns(pl.lit(True).alias("p"))
    T = truth.select("s1", "r").unique().with_columns(pl.lit(True).alias("t"))
    j = P.join(T, on=["s1", "r"], how="full", coalesce=True).with_columns(pl.col("p").fill_null(False), pl.col("t").fill_null(False))
    per = j.group_by("s1").agg((pl.col("p") & pl.col("t")).sum().alias("tp"), pl.col("p").sum().alias("np"), pl.col("t").sum().alias("nt"))
    per = pl.DataFrame({"s1": s1_ids}).join(per, on="s1", how="left").fill_null(0)
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0) \
          .when((pl.col("tp") == 0)).then(0.0) \
          .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("nt") + pl.col("np")))
    per = per.with_columns(f.alias("f"))
    return {"f05": per["f"].mean(), "n": per.height,
            "singleton_f": per.filter(pl.col("nt") == 0)["f"].mean(),
            "nonsingleton_f": per.filter(pl.col("nt") > 0)["f"].mean(),
            "pair_precision": (per["tp"].sum() / max(per["np"].sum(), 1)),
            "pair_recall": (per["tp"].sum() / max(per["nt"].sum(), 1)), "per": per}


def _selftest():
    # brief example: predict [a,b,c], truth [a,c] -> P=2/3, R=1 -> 0.714
    pred = pl.DataFrame({"s1": ["x", "x", "x"], "r": ["a", "b", "c"]})
    truth = pl.DataFrame({"s1": ["x", "x"], "r": ["a", "c"]})
    m = macro_f05(pred, truth, pl.Series(["x", "y", "z"]))
    # x: 0.714 ; y: singleton, empty pred -> 1.0 ; z: singleton -> 1.0
    assert abs(m["per"].sort("s1")["f"][0] - 0.7142857) < 1e-6, m
    assert abs(m["f05"] - (0.7142857 + 2) / 3) < 1e-6, m
    m2 = macro_f05(pl.DataFrame({"s1": ["y"], "r": ["q"]}), truth, pl.Series(["x", "y"]))
    assert m2["per"].sort("s1")["f"].to_list() == [0.0, 0.0], m2  # x: nothing predicted -> 0 ; y: FP on singleton -> 0
    return True


if __name__ == "__main__":
    print("selftest", _selftest())
