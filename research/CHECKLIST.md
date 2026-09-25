# Master checklist: nothing gets lost

Legend: `[x]` done or verified (with evidence) · `[ ]` to do · `[~]` in progress / partial · `[!]` risk or needs a decision · `[-]` rejected (with reason).
Update this file whenever anything is found, decided or dropped. It is the single index; details live in the linked docs.

---

## A. Verified facts (evidence in `research/eda/`, `kaggle/*/results_*.txt`)
- [x] Scale: train 2.21M S1 / 10.32M S2+S3; test 1.73M S1 (France 15%) / 9.97M. (`h1.py`)
- [x] Singletons 5.6% train; matches per S1: mean 3.46, max 11; per source S2 ≤ 5, S3 ≤ 6. (`h1.py`)
- [x] Exclusivity in train: 0 of 7,638,365 matched records under > 1 S1. (`h1.py`) **[!] not confirmed for test**
- [x] True pairs share the country 100%. (`pairs.py`)
- [x] Orphans 26% of S2/S3; test has 22% more S2/S3 per S1 (5.76 vs 4.67). (`h1.py`, `france.py`)
- [x] Twin decoys: same name (+ a qualifier word), nearby house number; 84% of singletons have one. (`lookalike.py`)
- [x] Number sets: identical in 71% of true pairs vs 1.3% of look-alike false pairs; disjoint 4.7% vs 72%. (`num_noise.py`)
- [x] Decoy extra words (Holdings, Group, Enterprises, Infratech, Lakeside…) vs noise extra words (Center, Services, dba, Sri, Mr, Dr). (`num_noise.py`)
- [x] Decoy words are ~absent in France (0.1% vs ~10%), so France needs generic features. (`france.py`)
- [x] Indian-script names: closed vocabulary, 1,347 types, 99.99% align word for word, 98.8% consistent; test 171 unseen types = 3.6% of tokens; addresses contain only 16 Indian-script phrases (states). (`indic_lexicon_probe.py`)
- [x] **No leakage:** row order and ID numbers are random (sibling row gap ~1.1M, corr 0.0001). (`leak_check.py`)
- [x] Candidate-search v4: pair recall 95.8% at ~62 candidates/S1, 88.3% of S1s complete; weak spots: no-address 74%, Indian script 82%. (`kaggle/blocking_v4/results_v4.txt`)
- [x] v2 lesson: ranking each search method separately loses joint evidence (recall dropped); keep one combined score.
- [x] Memory: index per country, chunked, hash-only frequency counts (the 436M-token OOM is fixed).
- [x] Kaggle phone-verified (2026-09-25): kernels can use the internet → pip install and Hugging Face model downloads work in notebooks (set `enable_internet: true`). Final package must still run **offline**: vendor the weights and pin the versions.

## B. Rules (from `organiser_answers.md`)
- [x] **Candidate-set size per S1 counts toward final ranking.**
- [x] candidate_pairs = input to the FIRST scoring model; matches ⊆ candidates.
- [x] Model limits are **per model**: MIT/Apache (check the base model's licence too), ≤ 8B **total** params, offline, fine-tuned by us only on provided data; preprocessing models count.
- [x] Unsupervised test statistics, self-training and synthetic pairs from provided records are allowed.
- [x] Small hand-written dictionaries allowed; geo/postal packages and large static geo tables prohibited; RapidFuzz, jellyfish, fastText, wordfreq allowed.
- [x] Include every duplicate; row order doesn't matter; country may be a feature (open set); France is in both LB splits.
- [!] Exclusivity in test not confirmed → enforce in the decision layer; only a soft prune in blocking.
- [!] Interpretation: a **bi-encoder / embedding retriever inside blocking** counts as candidate generation, not a "scoring model" (the organisers list "embedder" as a normal model). A **cross-encoder or GBDT that prunes** would make its input the candidate set. Document this clearly.
- [!] Pool compute across **each teammate's own** accounts: fine. **Don't** use extra accounts to multiply leaderboard submissions or probes; that's a fair-play violation and risks disqualification.

## C. Pipeline status (master plan: `strategy_v3.md`)
| # | Step | Status |
|---|---|---|
| 0 | EDA, hypotheses, leakage check | [x] |
| 1 | Learned artefacts (Indic dictionary + fallback, state aliases, noise/decoy words, pseudo-word alias detector) + small hand tables (FR/US/IN) | [ ] |
| 2 | Blocking v5 "tight" (cluster closure + deterministic pruning) + frontier sweep | [ ] |
| 2b | Fine-tuned bi-encoder in blocking (distilled from the cross-encoder) | [ ] |
| 3 | Full train-shard + full-test candidate generation, writer, validator | [ ] |
| 4 | Level-1 models: hand-feature GBDT · cross-encoder (0.6B → 4B/8B) · listwise LLM judge · bi-encoder cosine | [ ] |
| 5 | Level-2 stacker (GBDT on out-of-fold level-1 outputs + features) + calibration | [ ] |
| 6 | Decision: has-match head, cluster choice, global assignment, expected F0.5, per-source caps | [ ] |
| 7 | Second pass ("re-crossing"): profile-enriched re-retrieval for missed copies | [ ] |
| 8 | France: hand table, self-training, synthetic pairs, LB probe | [ ] |
| 9 | Final fits, package, documentation (every model + licence + params) | [ ] |

## C2. Verified plan decisions (2026-09-25)
- [x] Plan phases 1–8 checked against rules B (see chat/strategy_v3).
- [x] **Stacking via split blending, not K-fold:** train S1s split by group into A 60% (level-1 models) / B 30% (level-2 combiner) / C 10% (final holdout for cut-offs and calibration). Saves 5× GPU training on the heavy models.
- [x] Code lives in the real package `code/business_entity_resolution/src/ber/`; Kaggle notebooks `pip install` it from GitHub.

## D. Idea backlog (each gets tested; nothing dropped silently)
**Cross-stage reuse ("give earlier steps what later steps know")**
- [ ] Distil the cross-encoder into the blocking bi-encoder → better ranking → smaller candidate set at equal recall.
- [ ] Noise/decoy word lists learned for the matcher are reused in blocking normalisation (noise removed; decoy words kept as pruning evidence).
- [ ] Number-relation logic from the matcher, used as a deterministic blocking gate (reject twins early; this shrinks the candidate set **and** helps precision).
- [ ] Test-wide sibling clustering (unsupervised, allowed) feeds blocking: retrieve clusters, not records.
- [ ] Second pass: confident matches → merged S1 profile (all name variants + missing address parts) → re-retrieve → score only the new candidates (they're added to candidate_pairs, since the model runs on them).
- [ ] Out-of-fold matcher errors → new normalisation rules (error-analysis loop).
- [ ] Record-vs-record model reused for clustering **and** as sibling-support features.

**Exploiting the generator's structure (legit, uses only provided data)**
- [ ] Closed vocabularies: Indian-script words, noise words, decoy words, legal forms → exact tables learned from train.
- [ ] Pseudo-word detector for "Xyzqwe F/K/A Real Name" aliases (made-up names like Nexarcriza, Tavobrix): strip the fake part.
- [ ] Number-perturbation signatures: true copies = dropped digit / leading zero / bad ordinal suffix / missing; twins = shift ≤ 25 / 1-digit edit. Feature + gate.
- [ ] Per-source caps (S2 ≤ 5, S3 ≤ 6) and total ≤ 11 in decoding.
- [ ] Twin structure: measure at full scale whether each orphan cluster is the twin of exactly one S1; if so, use "exactly two nearby clusters" reasoning.
- [ ] S1×S1 look-alikes on test (S1 is deduplicated) = guaranteed negatives for self-training.
- [ ] Global exclusivity across all test S1s: each record assigned once (greedy / min-cost flow).

**Leaderboard probes (within our own submission budget)**
- [ ] All-empty → exact test singleton rate.
- [ ] France on/off → France contribution.
- [!] Don't over-tune to the public LB (it's a subset); prefer CV plus the country-holdout check.

**Heavy models (compute is not a constraint now)**
- [ ] Cross-encoder bake-off: Qwen3-Reranker 0.6B / 4B / 8B vs bge-reranker-v2-m3; with and without digits in the input.
- [ ] Listwise judge: Qwen3-8B, LoRA; input = S1 + its candidate clusters; output = which cluster(s), or none. First on the uncertain band, then on everything if it helps.
- [ ] Bi-encoder: Qwen3-Embedding-4B/8B vs bge-m3 (dense + sparse + multi-vector).
- [ ] Ensembling across seeds, folds and model families; order-swap test-time augmentation for the cross-encoders.

## E. Rejected (with reasons)
- [-] Ranking each search method separately (v2): recall dropped 87.3% → 84.5%.
- [-] Qwen transliteration of every record: the closed vocabulary makes a learned dictionary exact and 1,000× cheaper.
- [-] libpostal, gazetteers, NLLB, Jina, Gemma/Llama-based models, Qwen3.5-9B, MoE > 8B total: rules or licences.
- [-] Transitive-closure merging: chains of twins collapse precision.
- [-] ID or row-order tricks: verified there is no signal.
- [-] Extra accounts for leaderboard probing: fair-play violation.
