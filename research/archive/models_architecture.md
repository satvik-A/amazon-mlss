# Model research and reusable architecture (2026-09-25)

What this covers: which models to use (translation/transliteration, embedding, pair classification, LLM judge), how to **reuse one backbone for many jobs**, what runs in parallel, and **every constraint**, each with its source and where it is enforced.
Companion documents: `plan.md` (data findings), `training_plan.md` (training/validation process).

---

## 0. Summary
1. **Translation is mostly solved by data, not models.** The generator converts names to Indian scripts **word by word from a closed vocabulary**. In training: 1,347 Indian-script word types; 551,230 of 551,240 true pairs line up word for word; the most common mapping is right 98.8% of the time. Test: 1,518 types, **171 unseen** (3.6% of tokens). → A **dictionary learned from training** handles 96.4% of tokens. A tiny transliteration model (IndicXlit, 11M params, MIT) or Qwen handles only the **171 unseen word types**: translate the vocabulary once, not 1.4M records.
2. **Fine-tuned small models beat large zero-shot ones for entity matching.** AnyMatch at 1.3B roughly equals GPT-4, and fine-tuned small LLMs beat zero-shot GPT-4 on most datasets (EDBT 2025). → Fine-tune one small multilingual backbone. Don't prompt a big model.
3. **Split the work by what each model is good at.** Twins differ in **digits** (house numbers), which transformers compare poorly. The **GBDT** handles numbers, structure and competition. The **transformer** handles name noise (scrambled letters, transliteration, aliases, website names).
4. **One backbone, several LoRA heads:** a Qwen3-0.6B-family model serves (a) the retrieval embedding, (b) the pair classifier for S1↔R **and** R↔R, and (c) name normalisation checks. An optional Qwen3-4B "select" judge covers only the uncertain band. The limit is **per model** (confirmed), so each slot can use up to an 8B model; §3.

---

## 1. Constraints (source → how we enforce it)
| # | Constraint | Source | Enforced where |
|---|---|---|---|
| C1 | S1 is deduplicated: one record per real business | **Statement** (README l.7, l.50) | Basis of C2; S1×S1 pairs = guaranteed negatives for training |
| C2 | **An S2/S3 record matches at most one S1** | **Not stated explicitly**; it follows from C1 (each business appears in S1 once). **Verified in training: 0 of 7,638,365 matched records are under more than one S1.** Pending organiser Q7 for test | `decide/assign.py`: greedy global assignment, each record to ≤ 1 S1. Also rival-score features |
| C3 | An S1 may match 0, 1 or many records | Statement (l.7) | Decision layer allows empty and multi-record outputs |
| C4 | Exactly one row per test S1; no duplicate ids; S2/S3 ids only; ids must exist in test | Statement (l.93–95, 183–185) | `io/submission.py` asserts + `validate_submission.py` |
| C5 | Matches ⊆ candidates | Statement (l.128; validator warns) | Assert in writer |
| C6 | `candidate_pairs.tsv` = exactly the set the final model scores | Statement | The writer takes the scorer's input frame, not a separate list |
| C7 | Final model MIT/Apache 2.0, ≤ 8B parameters | Statement (l.186); per-model vs total is **ambiguous** (Q2) | §3 budget: design ≤ 8B **total** |
| C8 | No external data lookup (APIs, registries, geocoding, internet augmentation) | Statement (l.240–252) | Dictionaries learned only from provided data; no libpostal/gazetteers; model weights only |
| C9 | Country is an open set; France appears only in test | Statement | Features computed per country string; no one-hot; unit test with a made-up country |
| C10 | True pairs always share the country | **Empirical** (100%) | Build the index per country |
| C11 | Per S1: 1–6 copies per source, ≤ 11 in total | Empirical | Cluster size cap 12; prior for how many to predict |
| C12 | Closed Indian-script vocabulary (train 1,347 types; test 171 unseen) | Empirical | Dictionary + fallback for unseen words (§2.A) |
| C13 | 26% of S2/S3 records are orphans (mostly twin decoys); test has 22% more S2/S3 per S1 | Empirical | Decoy features; has-match head; prior from the LB probe |
| C14 | Train singleton rate 5.6%; test unknown | Empirical | All-empty LB probe calibrates it |
| C15 | Compute: laptop 16 GB (guard at 6 GB); Kaggle CPU 30 GB / 4 cores; 2×T4 GPU about 30 h/week; **no internet in kernels** | Environment | Weights and wheels uploaded as private Kaggle datasets; chunked, per-country jobs |

---

## 2. Model landscape (licences checked on the model cards; re-verify before final use)

### A. Transliteration and translation
| Model | Licence | Size | Verdict |
|---|---|---|---|
| **Learned dictionary** (ours, fitted on training pairs) | ours | — | **Primary.** Exact generator vocabulary; covers 96.4% of test tokens |
| **IndicXlit** (AI4Bharat) | MIT | ~11M | **Fallback for unseen words.** Converts Indian scripts to Latin with top-k candidates; snap to the S1 vocabulary by skeleton or edit distance |
| IndicTrans2 (AI4Bharat) | MIT | ~200M distilled / 1B | Not needed: these names are *transliterated*, not translated |
| Qwen3-4B / Qwen3.5-4B | Apache | 4B | Alternative fallback on the 171 unseen words (seconds). Pending Q3 |
| NLLB-200, Jina, EmbeddingGemma, Llama/Gemma family | CC-BY-NC / custom | — | **Excluded** (licence) |
| Aksharamukha | AGPL | lib | Avoid (copyleft); our parallel-block transliterator covers this |

### B. Embeddings (retrieval method + cosine feature)
| Model | Licence | Size | Notes |
|---|---|---|---|
| **Qwen3-Embedding-0.6B** | Apache | 0.6B | 100+ languages; instruction-aware; **shares the Qwen3 base with Qwen3-Reranker, so it can be reused** |
| **BAAI/bge-m3** | MIT | 568M | One pass gives **dense + sparse + multi-vector (ColBERT)**: three retrieval signals from one model; strong cross-lingual |
| multilingual-e5-small/base | MIT | 118M/278M | Cheapest; fallback if GPU is tight |
| mmBERT / Ettin encoders | MIT (check mmBERT card) | 140M–1B | Modern encoders; good fine-tuning bases |

### C. Pair classifier / cross-encoder (the reusable workhorse)
| Model | Licence | Size | Notes |
|---|---|---|---|
| **Qwen3-Reranker-0.6B** | Apache | 0.6B | Same base as B; fine-tune → S1↔R and R↔R classifier |
| **bge-reranker-v2-m3** | Apache | 568M | XLM-R base, strong multilingual; same family as bge-m3 |
| mxbai-rerank-base-v2 | Apache | 0.5B | Qwen2.5-based |
| Ettin reranker | MIT | 32M–1B | Very fast small options for distillation |

### D. Generative judge (optional, uncertain band only)
| Model | Licence | Size | Notes |
|---|---|---|---|
| **Qwen3-4B / Qwen3.5-4B** | Apache | 4B | LoRA; listwise "select" prompt (S1 + ≤ 5 candidate clusters → which one, or none) |
| Qwen3-8B | Apache | 8B | Allowed (limit is per model): best judge choice |
| Qwen3.5-9B | Apache | **9B** | **Excluded (> 8B)** |
| Phi-4-mini | MIT | 3.8B | Alternative |

### E. Structure and tabular
| Tool | Licence | Use |
|---|---|---|
| **LightGBM** | MIT | Main matcher, has-match head, candidate ranker |
| CatBoost | Apache | Second GBDT for the ensemble |
| GLiNER multi v2.1 | Apache (v2.1 only; v1 is CC-BY-NC) | Optional: zero-shot address-part parsing for France if regex fails |

---

## 3. Parameter budget: PER MODEL (confirmed by the organisers, answers 7/10)
"The limit is per model, so every model you use (embedder, reranker, matcher, or any preprocessing model) must **independently** meet it."
- There is **no combined budget**. Several models, ensembles and cascades are allowed.
- Each model must independently: be MIT/Apache-2.0 (including its **base model's** licence), have ≤ 8B **total** parameters (MoE total counts, so 35B-A3B is out), run offline, and be fine-tuned by us only on provided data.

What this unlocks (each model ≤ 8B, all can coexist):
| Slot | Best allowed choice | Alternative |
|---|---|---|
| Transliteration fallback (preprocessing) | IndicXlit 11M (MIT) | Qwen3-4B |
| Embedding / retrieval | Qwen3-Embedding-**4B/8B** or bge-m3 | Qwen3-Embedding-0.6B (fast) |
| Pair classifier (S1–R and R–R) | Qwen3-Reranker-**4B** (LoRA) | bge-reranker-v2-m3, Qwen3-Reranker-0.6B |
| Listwise judge | Qwen3-**8B** (LoRA) | Qwen3-4B / Qwen3.5-4B |
| Tabular | LightGBM + CatBoost ensemble | — |

Limits in practice are now **compute** (Kaggle T4 about 30 GPU-h/week; test has 1.73M S1) and **candidate-set size** (strategy v3), not licence budget. Choose model size per slot by measured gain per GPU-hour.

## 4. Architecture: one backbone, many heads

```
                         ┌──────────── learned artefacts (fitted on train only) ────────────┐
                         │ Indic dictionary · state table · noise words · decoy words ·      │
                         │ abbreviation map · legal forms · skeleton rules                    │
                         └───────────────────────────────┬──────────────────────────────────┘
raw records ──► normalise (text/) ── same function for S1, S2, S3, train and test ─┐
                                                                                   │
        ┌────────────────────────── block/ (per-country index) ───────────────────┤
        │  token methods (v4) ─┐                                                   │
        │  Qwen3-Emb-0.6B      ├─► candidates S1→R  (+ R→R for clustering,         │
        │  (LoRA-E) FAISS ─────┘     + R→S1 reverse for rival features)            │
        └──────────────────────────────┬────────────────────────────────────────────┘
                                       ▼
        features/pair.py  (ONE symmetric feature function: S1–R and R–R)
          numbers relation · name aliases · skeleton · noise/decoy words · addresses
          + cross-encoder logit  ◄── Qwen3-Reranker-0.6B (LoRA-C), fine-tuned
                                     on S1–R + R–R + S1×S1 negatives
                                       ▼
        cluster/  R–R GBDT (+ LoRA-C logit) → clusters + profiles (majority house number)
                                       ▼
        match/    GBDT stage 1 → rival features (from out-of-fold scores) → GBDT stage 2
                  (+ optional Qwen3-4B "select" judge on the uncertain band → feature)
                                       ▼
        decide/   has-match head · global assignment (C2) · choose cluster · expected-F0.5
                                       ▼
        io/       candidate_pairs.tsv (= scorer input, C6) · matching_results.tsv (C4, C5)
```

**What gets reused:**
- **One normalise function** for every source and split. Its artefacts are fitted once, versioned and saved as JSON.
- **One index builder** for three jobs: S1→R candidates, R→R clustering candidates, and R→S1 rival lookup.
- **One symmetric pair-feature function** for S1–R and R–R, so the clustering model and the matcher share code and features.
- **One cross-encoder (LoRA-C)** trained jointly on S1–R and R–R pairs, with a source-pair tag in the input. It is used in clustering, in matching, and as the distillation target for the judge.
- **One Qwen3-0.6B base.** The embedding adapter (LoRA-E) and classifier adapter (LoRA-C) are swapped on the same weights, so there is one download and one upload to Kaggle.
- **One exact scorer and one slice report** used by every experiment.

---

## 5. How the neural parts are trained
**Cross-encoder (LoRA-C on Qwen3-Reranker-0.6B; bge-reranker-v2-m3 as the other contender):**
- **Input:** `[SRC=S1|S2|S3] name ⟂ address` for each side, max 96 tokens, after normalisation (dictionary applied).
- **Data:** about 2M pairs from v4 candidates on training dev shards:
  - all positives;
  - hard negatives: twins, rival S1s' records, high-scoring candidates;
  - S1×S1 look-alike negatives;
  - R–R pairs labelled from the ground truth.

  Ratio about 1:3.
- **Augmentation:** swap the pair order (the model must be symmetric); re-apply noise operators sampled from the learned noise statistics.
- **Variant to test:** mask the digits in the transformer input, so the transformer focuses on names and the GBDT owns numbers.
- **Settings:** lr 1e-4 (LoRA r=16) or 2e-5 (full fine-tune), 1 epoch, bf16/fp16, group-aware validation split. Kaggle T4 ×2 with data parallelism.
- **Use:** its logit is a GBDT feature. Inference only on top-N candidates (about 15 per S1 → about 26M test pairs; cost to be measured).

**Embedding (LoRA-E):** contrastive (InfoNCE) with in-batch negatives plus mined twins; S1 as the query, R as the document. Keep it only if it adds unique recall at N ≥ 1 pt over v4.

**Judge (optional):** Qwen3-4B + LoRA, listwise select: "S1 + candidate clusters A–E → the matching letter, or none". Trained on about 50k S1s. Run on about 10% of S1s (the uncertain band). Its output becomes a feature, or its knowledge is distilled into LoRA-C.

---

## 6. Parallel work plan
**Inside one Kaggle job** (4 CPU + 2×T4): split by country × S1 chunk.
- CPU: normalise, index, token retrieval, GBDT features (polars is multi-threaded).
- GPU0: embeddings.
- GPU1: cross-encoder scoring.

Stages communicate through parquet files, each keyed by a hash of its config, so every stage is cached and can be rerun on its own.

**Independent work tracks** (separate notebooks; dependencies shown):
| Track | Work | Depends on |
|---|---|---|
| T1 | Learned artefacts: Indic dictionary, state table, noise/decoy words, IndicXlit fallback | — |
| T2 | Blocking v5 (v4 + T1 artefacts); then write candidate files for train shards + full test | T1 |
| T3 | Pair features + GBDT stage 1 + scorer + first LB submission | T2 |
| T4 | Cross-encoder bake-off (Qwen3-Reranker-0.6B vs bge-reranker-v2-m3) on the name-noise slice | T2 (pairs) |
| T5 | R–R model + clustering + cluster features | T3 code |
| T6 | Rival features, stage 2, has-match head, assignment, expected-F decoding | T3, T5 |
| T7 | Optional: embedding method / judge | T4 |

T1 and T4 can start now. T3 starts as soon as T2 writes train candidates.

---

## 7. Decisions still to measure
| Question | How we decide |
|---|---|
| Qwen3-Reranker-0.6B vs bge-reranker-v2-m3 | F0.5 improvement as a GBDT feature on the name-noise slice + inference cost |
| Mask digits in the transformer input? | Ablation on the twin slice |
| Is the embedding method worth it? | Unique recall at N over v4 |
| Is the judge worth it? | CV improvement on the uncertain band vs GPU hours |
| IndicXlit vs Qwen for unseen words | Accuracy of snapping to S1 vocabulary on train words held out from the dictionary |

---

## 8. Sources
- Qwen3 Embedding / Reranker (Apache 2.0, 0.6B/4B/8B, 100+ languages): [blog](https://qwenlm.github.io/blog/qwen3-embedding/), [paper](https://arxiv.org/abs/2506.05176), [HF card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- Qwen3.5 small series (Apache 2.0; 0.8B–9B): [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- IndicXlit (MIT, ~11M, 21 languages): [GitHub](https://github.com/AI4Bharat/IndicXlit); IndicTrans2 (MIT): [licence](https://github.com/ai4bharat/IndicTrans2/blob/main/LICENSE)
- Rerankers (bge-reranker-v2-m3, mxbai-rerank-v2 Apache): [Mixpeek 2026](https://mixpeek.com/curated-lists/best-rerankers), [Ettin rerankers](https://huggingface.co/blog/ettin-reranker)
- Ettin (MIT): [GitHub](https://github.com/JHU-CLSP/ettin-encoder-vs-decoder); mmBERT: [paper](https://arxiv.org/html/2509.06888v1)
- GLiNER v2.1 Apache (v1 CC-BY-NC): [overview](https://superlinked.com/glossary/what-is-gliner)
- Small fine-tuned models for entity matching: [AnyMatch](https://arxiv.org/html/2409.04073v1), [Fine-tuning LLMs for EM](https://arxiv.org/pdf/2409.08185), [Cross-dataset EM, EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-224.pdf), [OpenSanctions Pairs 2026](https://arxiv.org/html/2603.11051v1)
