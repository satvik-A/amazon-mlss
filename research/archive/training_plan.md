# End-to-end build and training plan

Read this with `plan.md` (the data findings). This file covers **how** we build, train, validate, tune, test and iterate.
Any choice that depends on an organiser answer that hasn't arrived yet is marked **[Qn]**, and the fallback is given.

---

## 0. Guiding principles
1. **Every decision is measured with the exact metric** (macro F0.5 per S1) on held-out data, sliced by country, entity size and twin presence.
2. **Cheap and deterministic first.** Normalisation plus LightGBM gets most of the score. Neural models are added only when they beat a measured bar.
3. **Everything generalises to an unseen country.** No word lists that only work for US/India; the cross-country check guards this.
4. **Laptop safety.** Integer ids, chunked processing, `guard.sh 6`. Heavy full-data runs go to Kaggle (about 30 GB RAM, free T4/P100 GPU).
5. **Log every experiment** in `experiments.csv`: git hash, config, CV overall and per slice, cross-country score, LB score, notes.

---

## 1. Models: what, why, and the licence

| Role | Choice | Licence | Size | Why | Fallback |
|---|---|---|---|---|---|
| Core matcher (S1 ↔ record, S1 ↔ cluster) | **LightGBM** | MIT | — | Best accuracy per hour on tabular similarity features; fast on CPU; handles missing values and categories | CatBoost (Apache) as a second model for the ensemble |
| Record ↔ record clustering model | **LightGBM** (same feature code) | MIT | — | Labels are free from the ground truth | — |
| Has-match head | **LightGBM** on per-S1 summary features | MIT | — | Separates "no match" from "the twin is here but it isn't me" | Logistic regression |
| String similarity | **rapidfuzz** | MIT | — | Fast C++ Jaro-Winkler, Levenshtein and token ratios, pair by pair | — |
| Transliteration (Indian scripts → Latin) | **Our own ~60-line mapper.** The Unicode Indic blocks are *parallel* (same offset = same sound across Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam), so one table covers all nine | ours | — | No external dependency; deterministic | PyICU "Any-Latin" **[Q5]** |
| Dense retrieval + cosine feature | **intfloat/multilingual-e5-small** | MIT | 118M | Recovers transliteration, acronym and French cases the token search misses; cheap enough for 10M records | bge-m3 (MIT, 568M) on Kaggle GPU if e5-small adds real recall |
| Pair re-scorer (optional stage) | **BAAI/bge-reranker-v2-m3**, fine-tuned | Apache 2.0 | 568M | Multilingual (French, Hindi); handles scrambled letters and odd forms better than hand-built features | microsoft/mdeberta-v3-base (MIT, 280M) |
| LLM (conditional) | **Qwen3-4B / Qwen3-8B** | Apache 2.0 | 4–8B | French synthetic data and abbreviation tables **[Q3]**; judge on uncertain pairs **[Q2]** | skip |

Parameter budget: LightGBM + e5-small (0.12B) + reranker (0.57B) = about 0.7B. That fits even if the 8B limit applies to the whole pipeline **[Q2]**. An 8B LLM fits only if the limit is per model.

---

## 2. Data preparation (deterministic code, written once)

### 2.1 Normaliser (a pure function; unit tests with real examples from EDA)
**Name**, producing several views: `raw`, `clean`, `core`, `sorted_core`, `aliases[]`, `acronym`, `phon`, `anagram_key`.
- NFKD accent fold; lowercase; map digits that stand in for letters inside words (5→s, 1→l/i, 0→o, 6→g, 3→e, 4→a) only where the token mixes letters and digits.
- Strip wrappers and junk: `[..]`, `(..)`, `--`, `<<`, `(ID: n)`, `#n`, a leading "the".
- Split aliases on f/k/a, d/b/a, t/a, aka, formerly. Keep all parts, and mark which part is the "old name" (the real one in the F/K/A pattern).
- Website/handle: strip `@`, `www.`, TLDs; keep the concatenated form for matching against the S1 name with spaces removed.
- Indian script → Latin (the parallel-block mapper), then the phonetic key.
- Legal forms: **found automatically** (tokens that appear often at name edges, across all countries including test France). Removed from `core`, but kept as a separate feature.
- `anagram_key` = each token's letters sorted. It catches letter scrambles when the letters are only permuted; a letter-bag similarity catches scrambles with extra letters inserted.

**Address**:
- `None` → missing.
- **Numbers:** strip `#` and leading zeros; split compound numbers (`8-2-248/1/7` → parts plus the joined form); turn ordinal words into digits (Seventh → 7); fix wrong ordinal suffixes (7nd → 7th).
- House number = the number next to a street word, or after No/H.No/D.No/Plot/Door. Unit = after Unit/Suite/Apt/Flat/Floor.
- **Street:** street words with abbreviations normalised (the list is mined from training true pairs; France uses the generic abbreviation match).
- **City / state:** a state alias table **mined from training true pairs** (S1 state ↔ S2/S3 state co-occurrence, which also covers native-script states). For France: region/department tokens by frequency.
- Order-free token sets for everything.

**Output:** parquet shards per `(split, country)` with integer ids. The normaliser runs streaming, about 500k rows per chunk.

### 2.2 Dev universe (keeps everything laptop-sized)
We use **region shards** rather than a random S1 sample, so the density of decoys and competition stays realistic.
- Pick about 6 US states and 4 Indian states, roughly 10% of training. Include **every** S1 and S2/S3 record in them (assigned by the parsed state; records with no state go to the region their city maps to).
- All development, feature work and tuning happens on this universe.
- Full training data is used only for the final fits (on Kaggle).

---

## 3. Candidate search (step 1 of the pipeline)

**Aim:** recall ≥ 97% (of true pairs) at ≤ 30 candidates per S1, or at the cluster level.

**Search arms** (all chunked by country and a batch of S1s; posting lists capped by frequency):
1. IDF-weighted token overlap on name words, name word pairs (first 4 words only), address words and numbers. This is the arm the earlier probe measured.
2. Exact keys: normalised `core` name; `sorted_core`; acronym; phonetic key; `(street word, house number)`; `(house number, city)`; website-concatenated name.
3. Digit-drop tolerant house-number key: `(street word, house number with its first or last digit dropped)`.
4. Dense: e5-small embeddings of `core name | street | city` with FAISS (IVF-flat, per country), top 20. **Kept only if its unique recall is ≥ 1 point.**
5. Reverse direction: for each S2/S3 record (or cluster), its top 5 S1s, so the competition features have the rival present.

**Fusion:** union → a cheap LightGBM ranker (about 20 features: arm hit flags, IDF score, quick rapidfuzz ratios) → top K per S1.
**`candidate_pairs.tsv` = the output of this ranker** (the exact set the final model scores).

**Measurements:** recall at 10/20/30/50 overall and per slice (Indian script, `None` address, website names, aliases, France-like cases); unique recall per arm; candidates per S1.

---

## 4. Record clustering (if [Q1] allows grouping test S2/S3 records; otherwise cluster statistics are computed per S1 on the fly)

### 4.1 Training data
- **Pairs:** S2/S3 ↔ S2/S3 pairs from an internal candidate search on the dev universe (same keys as §3, run S2/S3 against S2/S3).
- **Labels:** records under the same S1 = 1. Records under different S1s = 0. An orphan paired with a matched record = 0 (an orphan is by definition a different entity). **Orphan–orphan pairs are unlabelled** and dropped from training.
- **Features:** the same pair features as §5 (symmetric versions), plus the source pair (S2–S2, S2–S3, S3–S3).

### 4.2 Model
LightGBM binary, starting parameters: `num_leaves=127, lr=0.05, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1`, early stopping on a validation fold.

### 4.3 Clustering algorithm (compare on validation)
- **A (baseline):** links with p > τ_high → connected components → split any component larger than 12 at its weakest link.
- **B:** average-linkage agglomerative clustering within each connected block, stopping at p < τ.
- **C:** greedy correlation clustering (pivot) on p − 0.5.

Tune τ by B-cubed precision and recall **and** by the downstream F0.5. The downstream score decides.

**Cluster profile:** every name view, the majority house/unit number (count-weighted), how many copies agree with it, the union of address tokens, which sources are covered, size, and cohesion (mean and min internal p).

---

## 5. The main matcher: S1 ↔ record, with cluster context

### 5.1 Features (about 150, in groups; each group can be switched off for ablation)
| Group | Features |
|---|---|
| G1 Name strings | JW / Levenshtein / token-sort / token-set / partial ratios on `clean`, `core`, `sorted_core`, the best alias, the transliterated name; letter-trigram TF-IDF cosine; letter-bag similarity (scrambles); acronym match; website-concat match |
| G2 Name tokens | IDF of shared / S1-only / record-only tokens (sum and max); count of unexplained extra words; **qualifier score** of the extra word (how often that word appears as an extra word in S1-vs-S1 look-alikes of the *same country*, so it adapts to France without labels); legal form same / different / missing |
| G3 Numbers | set relation of address numbers (identical / subset / overlap / disjoint / missing); house number relation (equal / shift ≤ 25 with the shift size / 1-edit / digit-drop / swap / missing); unit same / different / missing; Jaccard of the number sets |
| G4 Address text | street word similarity; city match via the alias table; state match via the alias table; landmark overlap; token-set ratio of the whole address; address missing |
| G5 Cluster | whether the record's cluster majority number agrees with S1; share of cluster copies agreeing with S1's house number; cluster size; cohesion; best name similarity across the cluster's aliases; sources covered |
| G6 Competition (from out-of-fold stage-1 scores) | the record's rank in S1's list; S1's rank in the record's list; margin to the best other S1 for this record; margin to S1's next-best cluster; "each is the other's top pick"; number of S1s scoring above 0.5 |
| G7 Neural (optional) | e5 cosine; reranker logit (only for S1's top 5 candidates) |
| G8 Meta | source (S2/S3); name and address lengths; how complete each field is |

**Deliberately absent:** a country one-hot, raw token identities, and entity-id numbers.

### 5.2 Training setup
- **Rows:** all candidate pairs (from §3) for training-fold S1s. Hard negatives come naturally (twins, rivals).
- **Weighting:** try (a) no weights and (b) weight 1/(candidates of this S1), which mirrors the macro-per-entity metric. CV picks.
- **Objective:** binary log-loss (main). Try `lambdarank` grouped by S1 as a second model and stack the two.
- **Two stages:**
  - **Stage 1:** G1–G5, G7, G8. Produce K-fold out-of-fold predictions for *all* dev S1s.
  - **Stage 2:** Stage 1 features + G6 built from the out-of-fold predictions.

  This follows the standard stacking discipline: the scores behind a row's competition features must come from a model that never saw that row's labels.
- **Cluster-level variant:** rows = (S1, cluster) with aggregated features (mean/max over members + the cluster profile). The final decision uses whichever of record-level or cluster-level wins, or both, as features for a small decision model.

### 5.3 Has-match head
One row per S1. Features: top-1/top-2 cluster probability and the gap; the top cluster's number relation; the top cluster's size and support; whether the top cluster prefers another S1; name rarity; S1 completeness. Label: S1 has any match. Train on out-of-fold stage-2 outputs.

### 5.4 Decision layer (tuned on out-of-fold predictions against the exact metric)
1. Greedy global assignment: sort all (S1, record) by probability; a record goes to at most one S1.
2. Per S1: if P(has match) < θ₀ → predict nothing.
3. Otherwise take the best cluster's members with p > θ₁. Add records from other clusters only if p > θ₂ (expected θ₂ > θ₁).
4. **Alternative:** expected-F0.5 decoding on calibrated probabilities (`research/sims/decode_sim2.py`).
5. **Tuning:** coordinate search over θ₀, θ₁, θ₂ on out-of-fold data. Report a 95% bootstrap confidence interval. Pick the most **robust** point (a flat region), not the sharpest peak.

---

## 6. Optional neural stage (only if it earns its place)
- **When:** after step 5 exists. Look at the remaining errors; if more than about 30% are name-noise cases (scrambles, transliteration, website names) that the features miss, train the reranker.
- **Data:** about 1M (S1, record) pairs from the candidate search on the training dev universe, balanced 1:3, including twins. Input: `name [SEP] address` for each side, max 128 tokens.
- **Training (Kaggle T4):** bge-reranker-v2-m3; lr 2e-5; batch 32 (gradient accumulation to 64); 1–2 epochs; warmup 5%; fp16; binary cross-entropy. Validate on a held-out S1 group. Score each pair in both orders and average.
- **Use:** the logit is a G7 feature, computed only for each S1's top 5 candidates (about 9M test pairs, about 2–3 GPU-hours).
- **Keep it if** CV improves by at least 0.5 pt and the cross-country check doesn't get worse.
- **[Q3] favourable:** Qwen3-8B generates French noisy-copy pairs following the noise catalogue, and those are added to reranker training. Validate on the France sanity checks (§8.4).

---

## 7. Validation design

| Level | What | Used for |
|---|---|---|
| V1 | 5-fold **GroupKFold by S1** within the dev universe; candidate search and clustering run once over the whole universe (they use no labels); anything learned from labels (qualifier words, state aliases, models) is fit **inside the folds** | Every experiment |
| V2 | **Cross-country:** train on US, score India, and the reverse | The France stand-in; required for any feature or normalisation change |
| V3 | **Second region universe** (different states), never tuned on | Final check against overfitting the dev universe |
| V4 | Public leaderboard | A sanity check only; at most 1–2 submissions per real improvement **[Q10]** |

**Slices reported for every run:** country; singletons; S1s with a twin in their candidates; entity size 1 / 2–3 / 4–6 / 7+; records with Indian-script names; `None` addresses; website/alias names.
**Pipeline health metrics:** candidate recall at K; clustering B-cubed P/R; stage-1 AUC; calibration (reliability curve, ECE); decision F0.5.
**Rule for keeping a change:** V1 improves by ≥ 0.2 pt (paired bootstrap p < 0.1), V2 not worse by > 0.2 pt, and no slice drops by > 1 pt.

---

## 8. Tuning and iterating

### 8.1 Hyperparameters
- **Optuna, 50–100 trials,** on V1 fold 1 with early stopping. Search space: num_leaves 31–255, lr 0.02–0.1, min_data_in_leaf 50–1000, feature_fraction 0.5–1, bagging 0.6–1, lambda_l1/l2 0–10, max_bin 63–255.
- Then refit on all folds with the best 3 settings and average them (a cheap ensemble).
- Only tune after the features stabilise. Features are worth far more than hyperparameters here.

### 8.2 Feature iteration
- **Group ablation** (drop G1…G8 one at a time) → keep the groups that pay.
- **Permutation importance** on out-of-fold data → prune features that are noisy or drift between countries.
- **Adversarial check:** a classifier to tell train dev-universe features from test **France** features. Any feature that makes this easy is a France risk: bin it more coarsely or drop it.

### 8.3 The error-analysis loop (after every major step)
1. Sample 50 false positives and 50 false negatives from out-of-fold data, stratified by slice.
2. Tag each with a cause from the noise catalogue (scramble, twin with the same number, digit-drop, alias, transliteration, missing address, candidate-search miss, clustering merge or split, decision cut-off).
3. The biggest cause becomes the next normaliser rule or feature. Fix it and re-measure. Record it in `experiments.csv`.

### 8.4 France (no labels): how we adjust using the results
- Run the full pipeline on test France and check that it *looks like* US/India. Compare:
  - predicted matches per S1 (train mean about 3.5);
  - the share predicted empty (about 5.6%, or the value from the leaderboard probe);
  - the cluster size distribution;
  - the share of top clusters with identical number sets;
  - the score histograms.
- If France predicts far fewer matches, the candidate search or normalisation is failing. Inspect 50 French S1s by hand and fix what you see (not a word list: generic rules).
- **If [Q9] says the public leaderboard includes France:** submit "best model everywhere" vs "best model, but France left empty" to measure France's contribution.
- **Prior correction:** EM re-estimation of the positive rate on French candidates; apply it only if the cross-country check (V2) shows it helps the held-out country.

### 8.5 Leaderboard probes (budget [Q10])
1. **All-empty** → the test singleton rate → set the prior for the has-match head.
2. **Baseline** (exact address + name) → confirms the scorer and file format end to end.
3. From then on, submit only when V1 and V2 both improve.

---

## 9. Test inference (the full run)
1. Normalise every test file (chunked). Cluster test S2/S3 per country **[Q1]**.
2. Candidate search per country in S1 chunks of 50k → `candidate_pairs.tsv`.
3. Features → stage 1 → competition features → stage 2 → has-match head → decisions (global assignment, then per-S1 rules).
4. Write `matching_results.tsv`. Assert: every test S1 appears exactly once, no duplicates, only S2/S3 ids, matches ⊆ candidates. Then run `utils/validate_submission.py`.
5. Runtime targets: normalise < 30 min; candidate search < 2 h; features + models < 2 h on Kaggle CPU (+2–3 h GPU if the reranker is used).

---

## 10. Final models and ensembling
- Final fits on the **full training data** (Kaggle): LightGBM stage 1 and stage 2 (3 seeds × best 3 settings), plus CatBoost stage 2. Average the probabilities, then recalibrate (isotonic) on out-of-fold data.
- Re-tune the decision cut-offs on full out-of-fold data. Adjust θ₀ with the probed test singleton rate.
- **Two final candidates:** the best V1+V2 score, and the most robust one (flattest cut-offs, fewer parameters). If final selection is allowed, pick the robust one **[Q10]**.

---

## 11. Packaging (from day 1, not at the end)
```
code/business_entity_resolution/
  src/ normalize.py  blocking.py  cluster.py  features.py  train.py  decide.py  infer.py  score.py  config.yaml
  README.md (one command: data → candidates → matches), requirements.txt (pinned), seeds fixed
output/ matching_results.tsv  candidate_pairs.tsv
Documentation_template.md  (filled in from plan.md + experiments.csv)
```

---

## 12. Build order and milestones
| Milestone | Contents | Exit criterion |
|---|---|---|
| M1 Foundations | Normaliser + unit tests; scorer; dev universe; validator wrapper; guard | Scorer matches the brief's example; universe built < 6 GB |
| M2 Baseline submission | Candidate search v1 + rule baseline; all-empty probe | 2 valid LB submissions |
| M3 Candidate search v2 | All search arms, fusion ranker, recall report | ≥ 97% recall at 30 on the dev universe |
| M4 Matcher v1 | G1–G4 + G8 LightGBM stage 1 + tuned cut-offs | CV and LB improve substantially |
| M5 Clustering | Record-vs-record model + clustering + G5 + cluster decision | V1 twin-slice improves; B-cubed ≥ 0.95 |
| M6 Context | Stage 2 with G6 + has-match head + global assignment | V1 +; singleton slice improves |
| M7 France hardening | Generic qualifier feature, French normalisation, adversarial check, sanity checks, V2 | V2 gap narrows; France sanity checks look like US/India |
| M8 Neural (optional) | e5 search arm / reranker feature | Passes the §6 bar |
| M9 Final | Full-data fits, ensemble, cut-offs, package + documentation | Validator PASS; reproducible from a clean checkout |

## 13. Organiser answers that change this plan
| Q | If favourable | If not |
|---|---|---|
| Q1 test-data use | Record clustering on test; self-training on confident test France predictions | Cluster statistics computed per S1 from its candidates only |
| Q2 8B scope | LLM judge on uncertain pairs as an extra feature | Reranker only |
| Q3 LLM synthetic data | French synthetic training pairs → reranker/LightGBM | Generic features only |
| Q4 domain rules | Short hand-written French/Indian address tables | Everything mined or generic |
| Q5 libraries | PyICU transliteration; possibly libpostal | Our own parallel-block transliterator; regex parser |
| Q6/Q7/Q8 test facts | Hard exclusivity, country partitioning, confident priors | Keep them as soft features; recalibrate using probes |
| Q9 LB split | France probe | Rely on sanity checks |
| Q10–Q12 rules | Probe budget; choice of final submission; larger candidate sets | Conservative |
