# CONTEXT: current state (single source of truth; read this first)
Updated 2026-09-25 (evening). Older docs are in `research/archive/` (history only). Detailed ledger: `research/CHECKLIST.md` (older entries reference archived paths). Rules: `research/organiser_answers.md`.

## 1. Task and scoring
- Match S2/S3 records to each S1 business (S1 is deduplicated). Output `matching_results.tsv` (LB-scored) + `candidate_pairs.tsv` (final zip).
- Metric: **macro F0.5 per S1**; a singleton scores 1 if predicted empty, 0 otherwise.
- **Candidate-set size per S1 counts toward final ranking** (smaller is better). `candidate_pairs` = input to the **first scoring model**, so all shrinking must be deterministic blocking (similarity, keys, rules, cut-offs tuned on train); no learned pair-scorer before it.
- Models: each one separately MIT/Apache-2.0 (base model's licence too), ≤ 8B total params, offline, fine-tuned only on provided data. Multiple models OK. **Qwen3-8B / Qwen3-Reranker-8B = 8.19B → excluded.** Verified shortlist: `research/models.md`.
- Allowed: RapidFuzz/LightGBM/pandas/polars, small hand-written dictionaries, unsupervised statistics / clustering / self-training / synthetic pairs on test records. Prohibited: geo/postal packages (libpostal), large static geo tables, external lookups, hosted LLM APIs.
- Country is an open set; France appears only in test (public + private splits).

## 2. Data facts (verified)
- Train: 2.21M S1, 10.32M S2+S3. Test: 1.73M S1 (US 0.66M, India 0.81M, **France 0.26M**), 9.97M S2+S3; test has 22% more S2/S3 per S1.
- Singletons 5.6% (train). Matches per S1: mean 3.46, max 11; per source S2 ≤ 5, S3 ≤ 6.
- Each matched record belongs to exactly one S1 (0 exceptions of 7.64M); not confirmed for test → enforce in the decision layer only.
- Matches always share the country. 26% of S2/S3 are orphans.
- **Twin decoys:** same name (± a qualifier word), nearby house number (shift ≤ 25 / 1-digit edit); 84% of singletons have one. Number sets: identical in 62–71% of true pairs vs ~1–2% of look-alikes; disjoint 2–5% vs ~70%.
- **Test decoys use NEW qualifier words** (Medicals, Steel, Sweets, Bakery…; France too), and some share the exact address, differing only by a name word → generic "unexplained extra word" features, never a fixed train list.
- Exact-equal names country-wide are the same business only 8.6% of the time → name-only statistics are useless; learn word stats from the address-conditioned candidate pool.
- Noise seen: scrambled letters, accents, look-alike digits (5mart), legal suffix swaps/moves, "Fake F/K/A Real" (real name on the right), web handles, Indian-script names (9% of S2), literal `None` address (3.3%), component reordering, number noise (#, leading zeros, dropped digit, 7nd).
- No leakage in IDs or row order (checked).

## 3. Pipeline (package `code/business_entity_resolution/src/ber/`, pip-installed from GitHub in Kaggle jobs)
| Module | Role |
|---|---|
| `translit.py` | Rule-based Indic → Latin (fallback only) |
| `tables.py` | Small hand dictionaries: street types (US/IN/FR), number words, legal forms, alias markers, honorific noise words, leet map |
| `text.py` | fold (NFKD), dot collapse, tokenise, skeleton, leet fix |
| `artifacts.py` | Learned from provided data: Indic lexicon, sibling-based lexicon extension (test-legal), OOV fallback, address/name synonym pairs, vocab; label-free `pseudo_pairs` + test-time per-country address synonyms |
| `normalize.py` | `Normalizer(art_dir).transform(df)` → nw, core, alias, legal, decoy, sk, aw (street types mapped before the length filter; global + per-country synonyms expanded), ad, au, noaddr. Same for all sources/splits |
| `blocking.py` | `Index` (per-country inverted index, IDF, freq cap 5000 / keys 200), arms: primary combined (name+skeleton+address+numbers+units+number×word+word pairs+name×word), namepair, keys, nameonly, trigram; `expand` (sibling signatures); `number_relation`; `prune` (alpha, G, gate, beta) |
| `features.py` | `pair_features(P, QN, RN)`: 49 features (rapidfuzz batch ratios, token/skeleton sets, numbers/house/unit relations, address, blocking, context), ~60k pairs/s |
| `decide.py` | Decision layer: exact expected-F0.5 top-k decoding (Poisson-binomial DP), at-most-one-owner posterior per record, per-source caps, rank-threshold baseline |
| `metrics.py` | Exact macro F0.5 (self-tested on the brief's example) |
| `io.py` | Validator-safe writer (asserts every rule; matches ⊆ candidates) |

Operations: the laptop runs everything via `research/guard.sh 6 <cmd>` (6 GB kill switch), small subsets only. Full runs on Kaggle (user `satvikaderla`, internet on, private dataset `amazon-ml-er-2026-data` = parquet of all files). Repo: github.com/satvik-A/amazon-mlss (public).

## 4. Experiment results
**Blocking (30k train S1 vs full pool)**
| Run | Pairs found | Cands/S1 | S1 fully found | Notes |
|---|---|---|---|---|
| v1 IDF token overlap | 87.3% @30 | 30 | 72.2% | Indian-script 39% |
| v2 per-arm RRF fusion | 84.5% @30 | 30 | 67.7% | **rejected**: loses joint evidence |
| v3 combined primary + aux + siblings | 94.1% | 51 | 84.0% | Indian-script 82% |
| **v4** + word pairs, name×word, digit-drop, name-only | **95.8%** | 61.5 | **88.3%** | US 97.9 / India 92.8; no-address 74% (weakest) |
| v5 (normalised + trigram + groups + pruning sweep) | running | — | — | `kaggle/blocking_v5` |

**Artefacts (full data, `kaggle/artifacts`)**
- Indic lexicon 1,318 words (consistency 0.998). Sibling lexicon on hidden train words: coverage 100%, accuracy 100% (fallback 52%). Test unseen Indic words: 200 (5.3% of tokens): 25 via siblings, 175 via fallback (mostly decoy qualifiers).
- Address synonyms learned (st/street, state codes ↔ names, transliterated states); union-find merging rejected (ct, tn, "new" ambiguities).

**Normaliser fix (A/B, 52k true train pairs):** street abbreviations and 2-letter state codes were dropped before mapping. US address-word jaccard 0.617 → **0.863** (negatives 0.027 → 0.050); India 0.701 → 0.758; false unit mismatch (India) 4.9% → 2.75%. Commit a4c1490.

**Label-free pseudo-pairs** (same house number + same core name + ≥ 3 shared address words): train precision US **0.9975**, India 0.914, recall ~0.5; cover 90% of France test S1. Test-time per-country synonyms from them: France region↔department + street abbreviations; France address jaccard 0.648 → 0.705 (commit 46e2965).

**Determinism + scale (commits 2a1c90e, 1c53379):** rank ties were broken arbitrarily (~24% of an S1's candidates changed with batch composition). Now: ties by record id, integer-quantised IDF (exact sums), inputs in entity_id order → identical candidates for subset / full / shuffled inputs. Query: sorted-array postings gathered per chunk + chunks sized by expected join rows (identical output; no more full-table scans per chunk). ~37k joined posting rows per query; 69% from tokens with df 2000–5000; trigram arm = 37% of rows.

**Decision-rule simulation:** never use a 0.5 cut-off; rank-aware cut-offs or expected-F on context-aware probabilities + a has-match head.

**Validation:** exact scorer OK. **LB probe (all-empty) = 0.05642 → test singleton rate 5.64%** (train 5.6%: consistent). A wrong non-empty prediction on a singleton costs its full 1/N; every non-singleton S1 needs ≥1 correct match to score anything.

## 5. Running / next
1. `er-blocking-v5` (running; pre-fix normaliser, sample mode): frontier table (recall / completeness vs cands/S1) + labelled candidate pool `cand_train_sample.parquet`.
2. **Full candidate generation, FULL mode** (every S1 queried; code 988ce1d = all fixes + test-time synonyms): `er-cands-{train-us, train-india, test-us, test-india}` running, `er-cands-test-france` queued (5 concurrent CPU sessions max). Output per job: `pool_<split>_<country>.parquet` (internal pool, annotated; train with `y`) + for train `frontier_train_<country>.csv` (full-mode pruning frontier on ALL S1).
3. `er-matcher-v1` (ready; launch after v5; pinned 8764e93 = baseline normaliser): LightGBM on the v5 sample pool, 4 decision rules on holdout C, end-to-end F0.5 per blocking policy.
4. `er-matcher-v2` = same on the fixed normaliser → acceptance gate vs v1.
5. GPU (quota 30 h/week, resets Fri 00:00 UTC): `er-xenc-v1` (bge-reranker-v2-m3; zero-shot AUC 0.992) and `er-xenc-v1b` (Qwen3-Reranker 0.6B + 4B, LoRA) running; `er-xenc-v2` (ByT5-base, gte-multilingual-reranker) queued; `er-xenc-score` (scores level-1 uncertain band) and `er-stack` (level-2, acceptance gate) ready.
6. Then: matcher trained on the FULL-mode train pools (rev_margin definition must match test) → first real LB submission; cluster/context stage-2 + has-match head + global assignment; decoy-word stats from the pool; France self-training (pseudo-pairs); heavy models (cross-encoder, Qwen3-8B judge) as features to a GBDT combiner; bi-encoder distilled into blocking; second-pass re-retrieval; per-arm df-cap tuning (speed); packaging.

## 5b. Overnight automation (2026-09-25/26)
- `kaggle/orchestrate.py` (local background process, log `kaggle/orchestrator.log`, state `kaggle/.orchestrator_state.json`): launches matcher-full → submit (+ fetch + official validator), xenc-score → stack, and the GPU bake-offs v2/v3 as upstream jobs finish and slots free (5 CPU / 2 GPU).
- Account 2 (`satvik006`, token in `~/.kaggle2/access_token`, **not phone-verified → no internet, CPU only**): private dataset `satvik006/er-bundle` (data parquet, artefacts, `ber` source, Linux wheels for polars 1.44.2 / rapidfuzz 3.14.6 / lightgbm 4.6.0). First workload: `kaggle2/recall_lab` (5 blocking variants). Use `KAGGLE_API_TOKEN=$(cat ~/.kaggle2/access_token) kaggle ...`.
- Never use account 2 for leaderboard submissions (fair play).

## 6. Kaggle jobs (folder → kernel)
`kaggle/artifacts` → er-artifacts · `kaggle/blocking_v5` → er-blocking-v5 · `kaggle/matcher_v1` → er-matcher-v1 · `kaggle/matcher_v2` → er-matcher-v2 · `kaggle/cands_full/jobs/<split>_<country>` → er-cands-<split>-<country> (generated by `make_jobs.py <git ref>`; config baked into the script because Kaggle uploads only the code file). Push: `.venv/bin/kaggle kernels push -p <folder>`; results: `kaggle kernels output satvikaderla/<kernel> -p <folder>/kout`.
