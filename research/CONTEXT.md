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
| v5 (normalised + trigram + groups, sample mode) | 96.2% (internal pool) | 91.6 | 89.6% | pruning sweep: reverse-preference filters (beta) are a cliff (→ ≤ 77% complete at ≤ 6 cands); no-beta + gate: 88.2% at 43 cands. Sample-mode rev lookups (top-3) miss many true S1s → rely on FULL-mode frontiers; shrink the set with deterministic feature cut-offs instead |

**Recall lab (account 2; full S2/S3 pool, ~29k train S1 per country; internal pool = primary top-60 + aux + siblings)**
| Variant | US recall / complete | India recall / complete | No-address copy recall US / IN | Indic copy recall | cands/S1 | ms/query |
|---|---|---|---|---|---|---|
| r0 base | 98.16 / 94.0 | 92.92 / 82.7 | 74.2 / 70.5 | 85.8 | ~89 | ~10 |
| r1 aux caps ×2 | 98.43 / 94.9 | 93.56 / 84.1 | 78.7 / 74.2 | 86.8 | ~117 | ~10 |
| r3 df cap 2000 | 98.05 / 93.7 | 92.75 / 82.3 | 73.8 / 70.0 | 85.4 | ~86 | **4.4** |
| r4 top-100 + trigram 25 | 98.30 / 94.4 | 93.55 / 84.1 | 74.4 / 71.2 | 87.0 | ~136 | 7.4 |
| r5 + arm 7 (no-address names) | 98.80 / 96.1 | 93.72 / 84.8 | 87.6 / 90.7 | 85.9 | ~98 | 7.9 |
| r6 arm 7 cap 20 | 98.92 / 96.6 | 93.78 / 85.1 | 90.2 / 92.4 | 85.9 | ~108 | 13 |
| r7 arm 7 cap 20 + aux ×2 + df 2k | 98.87 / 96.4 | 94.07 / 85.6 | 89.6 / 91.4 | 86.6 | ~128 | 5.1 |
| r8 arms 7 + 8 (name key × address) | 98.98 / 96.5 | 96.37 / 89.9 | 87.6 / 90.7 | 94.9 | ~99 | 11.7 |
| **r9 arms 7 + 8 + aux ×2 + df 2k = new default (d83f580)** | **99.05 / 96.8** | **96.58 / 90.4** | 89.6 / 91.4 | **95.1** | ~129 | **5.2** |
Findings: caps move recall < 1 pt; the hole is **copies with no address (~72%)** → new arm 7 (name tokens of no-address records indexed on their own; local sample 0.905 → 0.967), **confirmed at full scale (r5): completeness +2.1 (US) / +2.2 (India), no-address copy recall +13 / +20 pts, +9 cands/S1** → adopted for the second candidate pass (current Kaggle pools = 988ce1d, without arm 7). df cap 2000 is 2.3× faster for −0.1/−0.2 pt.
Full-mode TRAIN India (first pass, all 883k S1): internal pool 89.2/S1, pair recall 0.9286, S1 complete 0.8253; group/beta pruning by blocking score is very lossy (top-5 groups: 13.8 cands, recall 0.846) — many true copies come from auxiliary arms with a low primary score → prune with similarity-feature cut-offs, not sc. Full-mode test pools (first pass, 988ce1d): France 259k S1 → 24.6M rows (95.0/S1) in 43 min; US 663k S1 → 58.8M internal rows (88.6/S1) in 1h54m; India 810k → 71.7M (88.5/S1) in 2h18m.

**Cross-encoder (bge-reranker-v2-m3, fine-tuned on A-split pairs, eval on C-split top-15 + aux):** AUC 0.9997 (zero-shot 0.992; Qwen3-Reranker-0.6B zero-shot 0.980; blocking score 0.941); top-1 0.998; singletons: best candidate p ≥ 0.5 for 22% → needs number features / has-match head (stacking). Qwen3-Reranker-4B: 0.9907 zero-shot → 0.99954 after only 18.6k pairs, but 11 pairs/s on a T4.

**Second pass, train India (all 883k S1, r9 blocking):** internal pool 130.6/S1, pair recall **0.9662** (first pass 0.9286), S1 complete **0.9043** (0.8253) — matches the recall lab. 97 min on 4 CPUs (sharded). **Train US** (1.32M S1): 169.9M rows (128.4/S1), pair recall **0.9911**, S1 complete **0.9697** (49k-S1 sample), 2h20m.

**Candidate cut-off rules** (`ber.model._cut_rules`, greedy OR-rules on similarity features, v5 pool, held-out half): first 8 rules → 8.4 cands/S1 keep 98.4% of in-pool positives; 10 → 11.9, 99.3%; 11 → 13.7, 99.55%; 14 → 26.4, 99.75% (vs 91.5 cands/S1 without). The matcher chooses the depth end to end.

**First pass died:** train US (988ce1d, unsharded) OOM at 165.5M raw candidates. **Second pass (r9 blocking, sharded, gather ids)** now runs on BOTH accounts (account 1 `er-cands2-*`, account 2 `er2-cands2-*`); whichever finishes first feeds matcher-full2 → submit2; stacking (xenc-score → stack) follows matcher-full2 on account 1.

**Artefacts (full data, `kaggle/artifacts`)**
- Indic lexicon 1,318 words (consistency 0.998). Sibling lexicon on hidden train words: coverage 100%, accuracy 100% (fallback 52%). Test unseen Indic words: 200 (5.3% of tokens): 25 via siblings, 175 via fallback (mostly decoy qualifiers).
- Address synonyms learned (st/street, state codes ↔ names, transliterated states); union-find merging rejected (ct, tn, "new" ambiguities).

**Normaliser fix (A/B, 52k true train pairs):** street abbreviations and 2-letter state codes were dropped before mapping. US address-word jaccard 0.617 → **0.863** (negatives 0.027 → 0.050); India 0.701 → 0.758; false unit mismatch (India) 4.9% → 2.75%. Commit a4c1490.

**Label-free pseudo-pairs** (same house number + same core name + ≥ 3 shared address words): train precision US **0.9975**, India 0.914, recall ~0.5; cover 90% of France test S1. Test-time per-country synonyms from them: France region↔department + street abbreviations; France address jaccard 0.648 → 0.705 (commit 46e2965).

**Determinism + scale (commits 2a1c90e, 1c53379):** rank ties were broken arbitrarily (~24% of an S1's candidates changed with batch composition). Now: ties by record id, integer-quantised IDF (exact sums), inputs in entity_id order → identical candidates for subset / full / shuffled inputs. Query: sorted-array postings gathered per chunk + chunks sized by expected join rows (identical output; no more full-table scans per chunk). ~37k joined posting rows per query; 69% from tokens with df 2000–5000; trigram arm = 37% of rows.

**Matcher v1 vs v2 (normaliser A/B; v5 sample pool, 30k S1, LightGBM on A, tuned on B, holdout C = 3,091 S1):**
| | C macro F0.5 | singleton | non-singleton | US | India |
|---|---|---|---|---|---|
| v1 old normaliser (8764e93) | 0.9590 | 0.945 | 0.960 | 0.972 | 0.941 |
| **v2 fixed normaliser (a4c1490)** | **0.9615 (+0.0025, gate passed)** | 0.969 | 0.961 | 0.973 | 0.944 |
Ceiling (perfect decisions on this pool) 0.9837. Decision rules within ±0.0005 of each other (rank-threshold best on B for v2). End-to-end vs candidate-set size (v2): 5.9 cands → 0.9444, 43 → 0.9582, 91.5 (no pruning) → 0.9615 → group/beta pruning is too lossy; next: deterministic feature cut-offs.

**Second-pass matcher (`er-matcher-full2`; r9 blocking, full-mode pools, 80k S1/country, context features; holdout C = 16,038 S1):**
- C macro F0.5 **0.9725** (rank-threshold t1 0.80 / t2 0.75; no candidate cut-off) — US 0.9785, India 0.9666; singleton 0.967, non-singleton 0.973; pair P 0.9933 R 0.9401. Ceiling (perfect decisions on the pool) 0.9928. Expected-F rules within 0.0002; has-head no gain.
- Pool: 97.9% of true pairs in the sampled pool. LightGBM 642 rounds; top features rev_margin (full-mode), prk, num_rel, bag_ratio, ad_tset, **a8**, n_extra_sk.
- Cut-off policies (end to end on C): 5 rules → 8.2 cands/S1, 0.9645; 8 → 13.8, 0.9703; 10 → 21.8, 0.9715; **11 → 23.9, 0.9718 (chosen: smallest within 0.0005 of best on B)**; 14 → 39.7, 0.9723; none → 124. Group-G pruning is much worse (G=10: 25.7 cands, 0.946).
- Label-free reference: C pseudo-pairs predicted as match US 0.9995, India 0.9951 (France number comes from er-submit2).
- Known bug: prune(G=99) drops ~0.1% of positives (null-gid join) — irrelevant now (cut-off policies).

**Level-2 stacking (`er-stack`, accepted):** GBDT on [level-1 features + level-1 p + bge / CANINE / MuRIL logits], cross-encoders scored only on the uncertain band 0.005 < p < 0.995 (7.6% of rows). Trained on B1, tuned on B2, holdout C (16,038 S1, cut-off 11 rules):
| | C macro F0.5 | singleton | non-singleton | P | R |
|---|---|---|---|---|---|
| level 1 | 0.9719 | 0.9626 | 0.9724 | 0.9933 | 0.9374 |
| **level 2** | **0.9811 (+0.0093)** | **0.9904** | 0.9805 | 0.9977 | 0.9486 |
US +0.0072, India +0.0113. On the band alone: level-1 p AUC 0.965, MuRIL 0.948, CANINE 0.925, bge 0.890 — the gain comes from combining.

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
- Account 2 (`satvik006`, token in `~/.kaggle2/access_token`, **phone-verified on 2026-09-26 → internet + GPU (T4 × 2, own 30 h/week quota) now allowed**): private dataset `satvik006/er-bundle` (data parquet, artefacts, `ber` source, Linux wheels for polars 1.44.2 / rapidfuzz 3.14.6 / lightgbm 4.6.0). First workload: `kaggle2/recall_lab` (5 blocking variants). Use `KAGGLE_API_TOKEN=$(cat ~/.kaggle2/access_token) kaggle ...`.
- Never use account 2 for leaderboard submissions (fair play).
- Account 3 (`satvik0006`, token in `~/.kaggle3/access_token`, from `KAGGLE_API_TOKEN3` in kaggle.json; phone-verified 2026-09-26): GPU + internet; bundle `satvik0006/er-bundle`. Same rule: never submits to the leaderboard.
- Capacity now: 3 accounts × (5 CPU + 2 GPU sessions, 30 GPU h/week each).

**Error analysis on held-out C (level 2 = 0.9811; loss 0.0189), 2026-09-26:**
| Cause | F0.5 lost |
|---|---|
| true copy never in the candidate list (India 0.0051, US 0.0015) | 0.0065 |
| true copy in the list but rejected (77% of these copies have NO address; only 4.4% of all copies do) | 0.0059 |
| S1 with copies, none found (71 S1; 37 with every copy outside the list) | 0.0044 |
| false matches (122 pairs; 61 belong to another S1) | 0.0020 |
- No-address copies are genuinely ambiguous: an exact-name no-address record is a true copy only 8% of the time (50% even when the S1 name is unique and it is the only such record) — other same-name businesses outside S1. Model is calibrated there; little to gain.
- **Blocking hole found:** Indian copies keep "first number + city + state"; the number x word tokens used only the first 6 address words, and in the long S1 address the city is at the end. Missed copies share a number x word key with their S1 31.5% (first 6 words) -> 64% (+ last 3 words). `blocking.ADDR_TAIL` (recall lab r10).
- **Region-specific places:** India city renames link many true pairs only through the alias (Telangana/Andhra Pradesh 36.9k, Kolkata/Calcutta 17.8k, Noida/Gautam Buddha Nagar 19.0k, Odisha/Orissa 13.5k, Bengaluru/Bangalore 10.1k, Trivandrum 6.7k, Gurugram 6.4k, Secunderabad 6.1k ...); learned synonyms covered only Orissa/Keralam/Delhi. Learned synonyms chained states through "pradesh" (MP records got ap/up/telangana) and "west" (West Bengal got wv). Copies add filler "door", literal "null". -> `BER_PLACES=1` (tables.PLACE_*), recall lab r11.
- Level-2 unsure band on C: 0.05<p<0.95 = 1.3% of rows, perfect-judge ceiling +0.0076; 0.01..0.99 = 3.8%, +0.0094 -> level-3 judge `er2-judge-v1` (account 2).
- LB submission 1 = `er-submit2` (level 1, validator PASS): 25.7 cands/S1, 3.34 matches/S1, 5.81% S1 empty.
- **LB result: 0.958** (holdout estimate 0.972; rank ~800+). Label-free diagnosis: test France assigns 3.62% of its predicted records to 2+ S1s (45,136 guaranteed-wrong pairs; US 0.08%, India 0.21%; train never). France has twin S1s: same name + house number + city, different STREET (e.g. "Nantes Foyer, 13 Av. des Préludes" vs "Nantes Foyer SARL, 13 Av. de l'Eperonnière"); US/India twins differ in the number, so the matcher never learned it. If US/India transferred, France would be ~0.88.
- Fixes (c98e425): `model.one_owner` (one S1 per record, across shards) + `model.street_filter` (France only; drops pairs whose street lines disagree, rapidfuzz < 50; hurts US/India on C -0.0009/-0.0028 so not applied there). France: 931k -> 876k pairs, matches/S1 3.59 -> 3.38, empty 4.8% -> 5.4%. In er-submit3, er-final3, er-l1test3. Fallback file `kaggle/submit2f/` (submit2 + France fixes, validator PASS).
- France training fix (5bbf7a8, in the pass-3 matcher): normaliser `stw` (street-line words), feature `st_sim`, 120k synthetic street twins (A-split copies, street line swapped, house number kept) as negatives.

**Plan to 0.988+ (top 50; leader 0.9956), 2026-09-26.** Theory: the leader loses only 0.0044, so the no-address loss (0.0059 on C), which looked unrecoverable per S1, must be recoverable: the records almost always belong to SOME same-name S1 (C: 20,183 exact-name no-address candidate rows hold 1,614 positives, ~1 per record). Per-S1 decisions cannot see that; our holdout samples S1s, so it cannot either.
| # | Lever | Targets (holdout loss) | Status |
|---|---|---|---|
| 1 | Global assignment: records compete across ALL S1s (exclusive posterior with a no-owner prior k; one owner per record) | no-address 0.0059, zero-TP part of 0.0044, FPs 0.0020 | measuring on full train countries (`er2-global-india/us`, account 2) |
| 2 | France: one owner + street filter (tonight) then the street feature + synthetic twins (pass 3) | most of the 0.014 LB gap | submit3 tonight; pass 3 |
| 3 | Blocking ADDR_TAIL=3 | never-shortlisted 0.0065 | pass 3 running |
| 4 | Level-3 judges (Qwen3.5-2B/4B, Qwen3-Reranker-4B, mxbai) | close calls | accounts 2 + 3 |
| 5 | keep_prk 60 -> 100, a 12th cut-off rule | 98 + 255 of 1,429 missed copies | next pass |

**Results 2026-09-26 evening:**
- Pass 3 (ADDR_TAIL=3, street feature `st_sim` #6 by gain, 120k synthetic street twins): level 1 C 0.9743 (pass 2 0.9718; India 0.9666 -> 0.9701), pool positives 0.9867, ceiling 0.9958, 26.9 cands/S1; **level 2 C 0.9842** (+0.0104 over level 1; pass 2 0.9811). France test: street filter now removes only 2,385 pairs (37,440 with the pass-2 model), duplicate claims 13,399.
- `er-submit3` (pass-2 stack + France fixes): validator PASS; France 3.30 matches/S1, empty 5.6%.
- **Level-3 judges rejected** (`er2-judge-v1`): Qwen3.5-2B AUC 0.898 and Qwen3-Reranker-4B 0.909 on the C band vs level-2 p 0.9375; blend +0.0007 on C (gate +0.002). Pair-wise text judges cannot resolve these pairs: what decides them is competition (who else claims the record), not the text.

## 6. Kaggle jobs (folder → kernel)
`kaggle/artifacts` → er-artifacts · `kaggle/blocking_v5` → er-blocking-v5 · `kaggle/matcher_v1` → er-matcher-v1 · `kaggle/matcher_v2` → er-matcher-v2 · `kaggle/cands_full/jobs/<split>_<country>` → er-cands-<split>-<country> (generated by `make_jobs.py <git ref>`; config baked into the script because Kaggle uploads only the code file). Push: `.venv/bin/kaggle kernels push -p <folder>`; results: `kaggle kernels output satvikaderla/<kernel> -p <folder>/kout`.

## Global assignment — India result (er2-global-india, pass-2 level-1 matcher, full country, C holdout)
- Shipped per-S1 rule 0.96632 -> + one_owner (retuned) 0.96823 (+0.0019) -> exclusive posterior q_k0.5 (t1 0.8, t2 0.7) 0.96901 (+0.0027 total, +0.0008 over one_owner).
- P_C ~0.992, R_C ~0.93: the remaining gap is RECALL of pairs that are in the pool but not selected, not precision.
- Verdict: exclusive posterior is a small real gain; add to final decision after US confirms. Not the 0.988 lever by itself.

## LB: submit3 = 0.976 (2026-09-26)
- submit2 0.958 -> submit3 0.976 (+0.018) from France fixes (one_owner + street_filter) + pass-2 stack. Held-out C for that model 0.981 -> LB gap now ~0.005 (was ~0.023).
- Pass-3 (C 0.9842) expected ~0.979 on LB if gap holds. 0.988 needs ~+0.009 more: recall of in-pool pairs (R ~0.93, P ~0.992) is the main lever.

## Organiser answer (2026-09-26): private LB = same test files (US/India/France), no new countries/languages; names may be Indic in one source and Latin in another.
- Checked test scripts: only Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam), India S2/S3 only, same mix as train. US/France: only accented Latin. Test S1 is all Latin.
- Already covered: indic_lexicon (learned from train pairs), oov_map, unsupervised test sibling map (organiser answer 11), translit fallback, MuRIL/CANINE cross-encoders. Pool recall for Indian-script copies 98.0%.
- To do: measure selected recall on Indic-script pairs vs Latin pairs in the missed-match analysis.

## Rules (organiser answers, 2026-09-26)
- Final ranking = PRIVATE score of our BEST PUBLIC submission -> the best public file must also be our real best; never keep a lucky public outlier as best.
- Prohibited: external data, geo/postal/gazetteer tables, city-level tables, hosted LLM APIs. Allowed: MIT/Apache <=8B offline models, pure-algorithm libs, small hand dictionaries (state abbreviations ok), translit libs (indic-transliteration, anyascii, uroman), IndicXlit, self-training and synthetic pairs from provided records, unsupervised stats on test.
- Compliance: shipped models MuRIL, CANINE (Apache-2.0), bge-reranker-v2-m3 (Apache-2.0), Qwen3-Reranker (Apache-2.0), all <=4B. tables.PLACE_ALIAS (Indian city renames) is a city-level table -> must stay OFF (BER_PLACES=0, never shipped; r11 gave no gain anyway). PLACE_PHRASE state tokens are OK, but ride on the same flag -> keep off.

## Missed-match analysis (research/missed_match_analysis.py; India train C, pass-2 level-1, full-country scoring, shipped rule + one_owner)
- Base F 0.9682, P 0.9929, R 0.9286. Where true pairs go: found 92.9%; not in candidates 3.8%; in pool but another S1 scored higher 1.1%; p>=0.5 but rank/caps cut 0.6%; p 0.1-0.5 1.1%; p<0.1 0.4%.
- Oracle gains: all blocking misses +0.0144 (pass 3 ADDR_TAIL already lifts India pool 0.966 -> 0.980, ~40% of it); all in-pool misses +0.0125; drop all FPs +0.0062; no-address records +0.0056; Indic-script records +0.0056.
- No-address records: 3.6% of true pairs, recall only 52-57% (vs 98% for records with an address); 3178 of their 5029 misses are "another S1 scored higher" (same name claimed by up to ~1900 S1s). Indic-script records: recall 0.940, slightly BETTER than Latin (0.926) -> cross-script matching is not the weak spot.
- No leakage: file row order and ID numbers are uncorrelated between S1 and its matches (corr 0.004 / 0.0005).
- Pass-3 level 2 on C (sampled): P 0.9968, R 0.9582 -> remaining loss is recall: ~1.3 pts blocking, ~2.9 pts selection.

## 2026-09-26 night: user directive = reach LB 0.99; candidate-set size no longer a constraint.
- France check: the 91.8% pseudo-pair hit rate was a false alarm. France names are generic (city + Club/Comite + legal form), so pseudo-pairs include same-number different-street look-alikes. With street similarity >= 80, France predicted 99.64% (India 99.6%). Real France misses: near-identical records not in candidates (generic names hit per-arm caps) -> loosen caps.
- Competition stack (India C, level-1 base): one_owner 0.96823, q_k0.5 0.96892, learned competition stack 0.96953 (+0.0013 over one_owner).
- **Sibling signal:** copies of the same S1 in the SAME source share noise: S2-S2 address similarity 76 vs 40 to the S1, 22% identical noisy addresses (same dropped digit / typo); S3-S3 12-19% identical. S2-S3 not (different formats). -> sibling features (similarity of r to the confident same-source candidates of the S1) in the level-2 stack. Prototype running (scratchpad sibstack.py).

## Pass 4 (started 2026-09-26 22:05, all 3 accounts)
- Blocking: arm caps x2 {0:200,3:60,4:60,5:80,6:50,7:40,8:20}, keep_prk 200, sibling expansion top-40 / groups <= 40 (df caps unchanged 2000/200). Jobs `er-cands4-*` (account 1, train + test) and `er3-cands4-train-*` (account 3, same config).
- Matcher 4 picks the cut-off with the best F on B (size no longer a constraint). Level-1 test in 12 shards.
- Chain (account 1, orchestrator): matcher4 -> l1test4 x3 + xenc-score4 -> stack4 (also saves stack_pred_B2/C) -> xenc-score-test4 x3 -> final4.
- Competition + sibling layer (`ber.compete`): account 3 `er3-global4-*` scores every train S1 with matcher4 (uploaded as dataset `satvik0006/er3-matcher4`) and writes feats_<country>; trained locally on stack_pred_B2 (+ feats), reported on C; uploaded as dataset `satvikaderla/er-comp4` (comp.txt + comp.json); final4 uses it when present.
- Account 2: recall lab round 5 (r12 caps x2, r13 caps x4 + df caps, r14 wide sibling expansion) + er2-global-us.
- Global assignment US (er2-global-us, pass-2 level-1, C): per-S1 rule 0.97819 -> one_owner 0.97886 (+0.0007) -> exclusive posterior best 0.9793 (q_k1.0; +0.0011 total, +0.0004 over one_owner). Consistent with India (+0.0008 over one_owner); the learned competition layer (pass 4) supersedes the fixed formula.
- Pass 4 blocking OOM: doubled limits with 250k-S1 shards were killed (India test, US train on acc3). Relaunched (India, US; train + test) with primary cap 120, keep_prk 120, other arms x2, sibling expansion 40/40, 100k-S1 shards. France test pool finished with the first (x2, keep_prk 200) config.
- Recall lab round 5 (30k S1/country): r12 (all caps x2, prk 200, df 5000) India recall 0.9796 -> 0.9887, complete 93.8 -> 96.4, no-address 91.4 -> 94.0; US 0.9932 -> 0.9953; 345 cands/S1. r13 (x4) India 0.9931 at 733/S1. r14 (sibling expansion 40/40) +70 cands/S1 for +0.0005 -> useless. Earlier labs: primary cap / keep_prk / df cap alone move little; arms 7 (no-address) + 8 (name x address) caps carry the gain.
- **Final pass-4 blocking (jobs4c, 23:xx):** arms 3-8 caps x2 {3:60,4:60,5:80,6:50,7:40,8:20}, df cap 5000 / key cap 400, primary 100 / keep_prk 60 unchanged, expansion unchanged, 100k-S1 shards. Superseded runs were stopped by deleting the kernels (the CLI cannot cancel a session or delete one version).

## LB: final3 = 0.978 (2026-09-26 night) — best so far (submit3 0.976). Held-out C 0.9842 -> LB gap ~0.006.
## Embedding search arm (er2-embed-lab, multilingual-e5-small, 30k train S1/country)
- India: word 0.9796 -> +top-10 0.9880 (+4.5 cands/S1), top-20 0.9894 (+13), top-50 0.9907 (+41); rescues 41-48% of word misses; Indic copies 0.980 -> 0.988; no-address copies ~unchanged (0.914 -> 0.917).
- US: 0.9932 -> top-10 0.9945 (+3.3), top-20 0.9949 (+10.7).
- Far cheaper than looser word caps (r12: +215 cands for 0.9887). -> pass 4 includes the embedding arm (top-20, always kept by the cut-off; erk / esim become matcher features). Jobs er-emb4-* (GPU) feed er-cands4-*.

## 2026-09-27 00:40 — why LB < C: test has ~2x orphan records per S1 (label-free)
- Records/S1: train 4.67 (US) / 4.68 (India), 75% linked (orphans 1.2/S1). Test 5.76 / 5.82 / 5.53 (France); final3 links only 58-60%.
  final3 predicts 3.3-3.4 matches/S1 = what train C gives (3.51 true x R 0.958 / P 0.997), so test true matches/S1 ~ train and the
  extra ~1.1 records/S1 are ORPHANS: test ~2.4-2.5 orphans/S1 vs train 1.2 (India 2.04M unlinked of 4.72M).
- Unlinked test records look exactly like train orphans (name equals some S1: India 0.319 vs 0.315, US 0.253 vs 0.269; name equals 2+ S1:
  0.27 vs 0.27 / 0.12 vs 0.14) -> same generator, twice the density -> twice the false-positive chances -> precision lower on test.
- Extra-name-word rate of predicted pairs is the same as train true pairs (US 0.405 vs 0.411, India 0.445 vs 0.456): no decoy flood.
- Action: thresholds tuned with orphan false positives weighted x2 (kaggle/comp4/train_comp.py W_ORPH; out/stack_thr_w.json; final4 reads it).
- France self-training is being tested by simulation first (kaggle2/selftrain, er2-selftrain): India-only teacher, pseudo-labelled US,
  scored on US C. Earlier evidence (CHECKLIST 123): France transfer looked fine, and pair-wise LLM judges lost to the stack (AUC 0.90 vs 0.94).
- Kaggle T4 queue: accounts 1 and 3 waited 1h+; account 2 started at once. Embedding outputs re-routed as datasets
  (satvik0006/er3-emb4-india, er-emb4-us / er3-emb4-us via scratchpad us_reroute.sh).
