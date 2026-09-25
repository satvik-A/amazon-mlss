"""Submission writers. Enforce every rule of utils/validate_submission.py before writing:
one row per test S1 (incl. empty), S2-/S3- ids only, ids exist in test, no duplicates, matches ⊆ candidates."""
from __future__ import annotations

import polars as pl


def _lists(pairs: pl.DataFrame, s1_all: pl.Series, col: str) -> pl.DataFrame:
    agg = pairs.select("s1", "r").unique().sort(["s1", "r"]).group_by("s1", maintain_order=True).agg(pl.col("r").str.join(",").alias(col))
    return pl.DataFrame({"source1_entity_id": s1_all}).join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left") \
             .with_columns(pl.col(col).fill_null(""))


def write_submission(matches: pl.DataFrame, candidates: pl.DataFrame, s1_all: pl.Series, valid_r: pl.Series,
                     match_path: str, cand_path: str) -> None:
    """matches/candidates: long [s1, r]. s1_all: all test S1 ids. valid_r: all test S2/S3 ids."""
    assert s1_all.n_unique() == len(s1_all), "duplicate S1 ids"
    for name, df in (("matches", matches), ("candidates", candidates)):
        assert df["r"].str.contains(r"^S[23]-").all(), f"{name}: non S2/S3 id"
        assert df["r"].is_in(valid_r.implode()).all(), f"{name}: id not in test"
        assert df["s1"].is_in(s1_all.implode()).all(), f"{name}: unknown S1"
    missing = matches.join(candidates.select("s1", "r"), on=["s1", "r"], how="anti")
    assert missing.height == 0, f"{missing.height} matched ids are not candidates (matches must be ⊆ candidates)"
    _lists(matches, s1_all, "matched_entity_ids").write_csv(match_path, separator="\t", quote_style="never")
    _lists(candidates, s1_all, "candidate_entity_ids").write_csv(cand_path, separator="\t", quote_style="never")
