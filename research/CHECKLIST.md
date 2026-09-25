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

## C. Pipeline status (current state: `research/CONTEXT.md`; old master plan: `archive/strategy_v3.md`)
| # | Step | Status |
|---|---|---|
| 0 | EDA, hypotheses, leakage check | [x] |
| 1 | Learned artefacts (Indic dictionary + fallback, state aliases, noise/decoy words, pseudo-word alias detector) + small hand tables (FR/US/IN) | [x] v1 done; [~] v2 adds test-time per-country address synonyms (Kaggle `er-artifacts` v2) |
| 2 | Blocking v5 "tight" (cluster closure + deterministic pruning) + frontier sweep | [~] running (`er-blocking-v5`) |
| 2b | Fine-tuned bi-encoder in blocking (distilled from the cross-encoder) | [ ] |
| 3 | Full train-shard + full-test candidate generation, writer, validator | [ ] |
| 4 | Level-1 models: hand-feature GBDT · cross-encoder (0.6B → 4B/8B) · listwise LLM judge · bi-encoder cosine | [ ] |
| 5 | Level-2 stacker (GBDT on out-of-fold level-1 outputs + features) + calibration | [ ] |
| 6 | Decision: has-match head, cluster choice, global assignment, expected F0.5, per-source caps | [~] `ber/decide.py` (exact expected-F0.5 top-k, at-most-one-owner posterior, caps); compared in matcher v1 |
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
- [x] Per-source caps (S2 ≤ 5, S3 ≤ 6) in decoding (`decide.source_caps`); total ≤ 11 still to add.
- [ ] Twin structure: measure at full scale whether each orphan cluster is the twin of exactly one S1; if so, use "exactly two nearby clusters" reasoning.
- [ ] S1×S1 look-alikes on test (S1 is deduplicated) = guaranteed negatives for self-training.
- [ ] Global exclusivity across all test S1s: each record assigned once (greedy / min-cost flow).

**Leaderboard probes (within our own submission budget)**
- [x] All-empty → **LB 0.05642 = test singleton rate 5.64%** (train 5.6%). (2026-09-25)
- [ ] France on/off → France contribution.
- [!] Don't over-tune to the public LB (it's a subset); prefer CV plus the country-holdout check.

**Heavy models (compute is not a constraint now)**
- [~] Cross-encoder bake-off: `er-xenc-v1` (bge-reranker-v2-m3 full FT; zero-shot AUC 0.992, top-1 0.998), `er-xenc-v1b` (Qwen3-Reranker 0.6B LoRA + 4B LoRA), `er-xenc-v2` queued (ByT5-base bytes, gte-multilingual-reranker). 8B excluded (8.19B).
- [ ] Recursive / iterative ideas (no time limit): iterative collective classification (neighbour predictions as features, 2–3 rounds); recurrent-depth LM (Huginn-3.9B, Apache) as judge; RWKV-7 / Mamba-2 hybrids for long listwise inputs; TRM-style tiny recursive set decoder trained from scratch.
- [!] **Qwen3-8B is 8.19B total params → over the 8B limit (excluded)**; so are Qwen3-Reranker-8B and granite-3.3-8b. Listwise judge candidates: Qwen3-4B (thinking), DeepSeek-R1-Distill-Qwen-7B (MIT), Qwen2.5-7B, Mistral-7B-v0.3, OLMo-2-7B, Phi-4-mini. Full shortlist with verified licences: `research/models.md`.
- [ ] Bi-encoder: Qwen3-Embedding-4B/8B vs bge-m3 (dense + sparse + multi-vector).
- [ ] Ensembling across seeds, folds and model families; order-swap test-time augmentation for the cross-encoders.

**Phase-1 findings (local sample; full numbers pending)**
- [x] Sibling-copy lexicon extension (unsupervised, test-legal): recovers hidden dictionary words with 81% coverage at 97.9% accuracy, vs 55% for the skeleton fallback.
- [x] Exact same name ≠ same business: only 73.7% of exact-equal-name pairs are true (twins/decoys), so blocking must never auto-accept on name alone.
- [x] Noise words (extra, still a true copy): company, smt, mr, dr, llp, sri/shri, the, limited, corporation (p_true 0.6–0.9). Decoy words (p_true ≈ 0): group, holdings, associates, care, private, dental, clinic, coastal, highland, metro, harbor…
- [x] Address synonyms learned: st/street, rd/road, state codes ↔ names (US + India), transliterated Indian states ↔ Latin. **Union-find merging rejected** (ct = court/Connecticut, tn = Tennessee/Tamil Nadu, "new" = New York/New Delhi) → synonym pairs used as extra tokens.

**Phase-1 full-data results (Kaggle `er-artifacts`)**
- [x] Indic lexicon 1,318 words (mean consistency 0.998). Sibling method on hidden train words: **coverage 100%, accuracy 100%** (fallback 52%).
- [x] Test: 200 unseen Indic words (5.3% of tokens): 25 resolved via siblings, 175 via fallback. **The unseen words are mostly test DECOY qualifiers** (Medicals, Steel, Sweets, Bakery, Jewellers, Motors…): decoy entities have no Latin sibling.
- [!] **Test decoys use NEW qualifier words** (like France): decoy features must be generic ("unexplained extra word"), never a fixed train list.
- [!] Some test decoys share the **exact same address** as the real entity and differ only by a name word ("Lakshmi Baba Consultancy Steel / Producer / Power" at one address), so name-difference features are as important as numbers.

**From the friend's plan (reviewed 2026-09-25)**
- [ ] Character-trigram / 4-char-prefix name search, for scrambled names with no address (our remaining misses).
- [x] Adopt a stricter acceptance gate: V1 ≥ +0.002, V2 not worse by > 0.002, no slice worse by > 0.010.
- [-] Its threshold table (θ = 0.84 → F0.5 0.99323), "98.5% recall at ≤ 25", "< 1.5 GB RAM", "< 4 min inference": unmeasured and internally inconsistent; not used.
- [-] "False positive costs 4× a false negative": actually ~2.7× with 4 true matches, and a total loss on singletons; we use the rank-aware decision layer instead of one global θ.

**Phase-2 findings (2026-09-25 evening)**
- [x] `ber/decide.py`: exact expected-F0.5 top-k decoding (Poisson-binomial DP, equal to brute-force enumeration; 0.16 s per 50k S1), at-most-one-owner posterior, per-source caps. Matcher v1 compares rank-threshold / expected-F / + exclusivity / + has-match head on holdout C.
- [x] **Normaliser bug fixed:** address words were length-filtered (≥ 3) BEFORE street-abbreviation mapping, so `St`, `Rd`, `R`, `Bd`, `Av` and **all 2-letter state codes** were dropped, and the learned state code↔name synonyms never fired on the code side. A/B on 52k true train pairs: US address-word jaccard **0.617 → 0.863** (identical 21% → 54%), random negatives 0.027 → 0.050; India 0.701 → 0.758. `No 15` no longer counted as a unit (India false unit mismatch on true pairs 4.9% → 2.75%). French `St` = Saint (per-country override). Name function words (de/du/la/and/of…) join the noise set; `compagnie` = legal form.
- [x] France test patterns: `R`/`R.` for rue (25% of S2/S3 addresses), `Rte`, `Bd`, `Av`, `N°`/`No`, region on one side vs department on the other (Nouvelle-Aquitaine ↔ Gironde…), random accents added (Mâison, SÊRVICE), `Cie` ↔ `Compagnie`, dotted legal forms, no postcodes (only house numbers).
- [x] **Label-free pseudo-pairs** (same first address number + same core-name set + ≥ 3 shared address words): precision on train **US 0.9975, India 0.914**, recall ~0.5; cover 90% of France test S1. Use: test-time synonyms (done), France self-training / calibration (to do).
- [x] **Test-time per-country address synonyms** mined from pseudo-pairs (`artifacts.fit_country_addr_synonyms`): France recovers region↔department (aquitaine/gironde, hauts/nord, pays/atlantique, pas) + street abbreviations; France name+house pairs address jaccard 0.648 → 0.705.
- [x] India pseudo-pairs 91% precise → `pseudo_pairs(num_equal=True)` requires equal number sets: India precision 0.992 (keeps 88%). India twins differ in a later number (6-3-891 vs 6-3-896); 70% of wrong pairs belong to another S1 (exclusivity layer targets exactly this).
- [x] **Determinism:** ties broken by record id, integer-quantised IDF, canonical input order → identical candidates regardless of batch composition / input order (was ~24% different). Required for reproducible candidate_pairs.
- [x] **Scalable query:** sorted-array postings + adaptive chunks (identical output). FULL mode computes reverse preference from forward candidates (no S1 index); SAMPLE mode keeps the S1 index → train the final matcher on FULL-mode pools.
- [~] Full-mode candidate pools for all train/test countries (Kaggle `er-cands-*`).
- [ ] Per-arm df caps (trigram arm = 37% of join rows) — speed vs recall experiment.
- [ ] France self-training: pseudo-pairs as positives + S1×S1 look-alikes as negatives → France-specific calibration of the matcher.

## E. Rejected (with reasons)
- [-] Ranking each search method separately (v2): recall dropped 87.3% → 84.5%.
- [-] Qwen transliteration of every record: the closed vocabulary makes a learned dictionary exact and 1,000× cheaper.
- [-] libpostal, gazetteers, NLLB, Jina, Gemma/Llama-based models, Qwen3.5-9B, MoE > 8B total: rules or licences.
- [-] Transitive-closure merging: chains of twins collapse precision.
- [-] ID or row-order tricks: verified there is no signal.
- [-] Extra accounts for leaderboard probing: fair-play violation.
