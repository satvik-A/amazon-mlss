# Strategy v3: rethought after the organiser answers (2026-09-25)

**This is now the master strategy.** It supersedes the priorities in `plan.md`, `training_plan.md` and `models_architecture.md`; their details still apply where they don't conflict.
Inputs: `organiser_answers.md` (the rules), `plan.md` (data findings), the blocking v1–v4 results (`plan.md` §8–9).

---

## 1. What changed, and what it means

| Change | Implication |
|---|---|
| **Candidate-set size counts toward the final ranking** ("smaller candidate set per S1 ranked higher, beyond the leaderboard"); the blocking code is reviewed | The objective now has two parts: **(a) maximise private F0.5, (b) minimise candidates per S1.** v4's ~62/S1 is far too large. The new target is the **frontier**: as few candidates as possible for a given recall |
| **candidate_pairs = input to the FIRST scoring model** in a cascade | A learned model (e.g. a LightGBM ranker) can't be used to shrink the candidate set behind the scenes. **Everything that shrinks candidates must be blocking**: unsupervised similarity, deterministic rules, and cut-offs (tuned on train) |
| Limits are **per model**; preprocessing models count; fine-tune only on provided data; offline | An 8B judge (Qwen3-8B) is allowed; IndicXlit (MIT, 11M) is allowed as preprocessing. Qwen3.5-9B is out (> 8B). Every model's licence goes in the documentation |
| **Small hand-written dictionaries allowed**; large static geo tables and geo/postal packages prohibited | Hand-write small French/US/India abbreviation and ordinal tables. State aliases **learned from the provided data** are fine (an algorithm on provided records). No libpostal or city lists |
| **Test-time statistics, self-training and synthetic pairs from provided records are allowed** | Clustering test S2/S3, pseudo-labelling France and generating synthetic French pairs are all clean now |
| Include **every** duplicate / content-identical copy | The decision layer takes the **whole matching cluster**, not a top-1. This fits the cluster design |
| France is in both public and private splits | The public LB reflects France, so it can be probed |
| Exclusivity (a record matches ≤ 1 S1) **still not confirmed for test** | Enforce it in the decision layer (training: 0 exceptions in 7.64M). In blocking, only use it as a *soft* prune when a record strongly prefers another S1 |

---

## 2. The two objectives, made measurable
- **Primary:** macro F0.5 on the private LB (proxy: grouped cross-validation + the country-holdout check).
- **Secondary (ranking tie-breaker, weight unknown):** candidates per S1, reported as mean and p95, plus pair recall, the share of S1s with all matches found, and the reduction ratio.
- **The floor:** true matches average **3.46 per S1** (max 11). A perfect blocker would output about 3.5 per S1. Twins add about 3 more if blocking can't reject them.
- **Operating-point rule:** give up candidate recall only where the matcher would have lost the pair anyway. Concretely, choose the cut-off that maximises **end-to-end** F0.5 minus a small size penalty. Present a curve of candidates/S1 against end-to-end F0.5, and pick a point near the knee (target **≤ 10/S1** with ≤ 0.5 pt F0.5 loss vs the rich set; stretch ≤ 7/S1).

Why this is attainable: the true answer comes in **clusters**. All 3–6 copies of one business share their number set, name skeleton and street. Blocking can therefore choose **clusters**, not a fixed top-K of records.

---

## 3. New blocking design: "retrieve wide internally, emit tight"
All steps are unsupervised or deterministic. Cut-offs are tuned on training labels. No learned model runs before `candidate_pairs.tsv`.

```
B1  Normalise (learned Indic dictionary, noise-word removal, look-alike digit fixes, small hand-written abbreviation tables)
B2  Wide retrieval (v4 methods, per-country index): internal pool of about 60 per S1   ← never emitted
B3  Cluster-closure: group the internal pool into sibling groups (exact number set / name skeleton /
    same (house number, street) key, union-find restricted to the pool); a group's score = max member score
B4  Deterministic pruning, in order:
      a. relative score cut-off: keep groups with score ≥ α·best (α tuned)
      b. compatibility gate: drop a group whose number set CONFLICTS with S1
         (disjoint, and not explained by digit-drop / 1-edit / leading zero / missing), unless its name is
         an exact-key match. Tuned for ≤ δ recall loss.
      c. soft exclusivity: drop (s, group) if the group's best S1 in the reverse lookup is s' ≠ s by a margin β
      d. keep at most G groups (G tuned, e.g. 2–3) and at most K records in total
B5  Emit candidate_pairs.tsv = the members of the surviving groups   (= input to the first scoring model)
```

Gate (b) is the sensitive one. Twins mostly differ by a shift of ≤ 25 or a 1-digit edit (56%), while only 2.7% of true pairs show those patterns. It needs precise house-number parsing, and it runs only where S1 has a clear house number.

**Metrics for each rule, on the full-scale Kaggle harness:** Δ recall, Δ entity completeness, Δ candidates/S1, and (once the matcher exists) Δ end-to-end F0.5.

---

## 4. Matching: a heavy stack, with GBDT as the combiner (not the whole model)
Compute is no longer the constraint (the team pools its own accounts), so every slot uses the strongest allowed model. **LightGBM stays, but as the level-2 stacker**, for reasons that have nothing to do with compute:
- The decisive signals are **structured**: digit relations (5619 vs 5623), cluster majorities, rival margins, per-source counts. Hand-built features + GBDT capture these exactly; transformers compare digits unreliably.
- GBDT combines heterogeneous scores (cross-encoder logit, LLM choice, bi-encoder cosine, 100+ hand features), handles missing values, calibrates well, and retrains in minutes, which the decision-layer tuning needs.
- This is how comparable competitions were won (Foursquare 1st place: multi-stage LightGBM with BERT-family scores added in later stages).

**Level 1** (all fine-tuned on provided data only; each ≤ 8B, MIT/Apache; out-of-fold predictions for training rows):
| Model | Role | Starts at | Scales to |
|---|---|---|---|
| M1 hand-feature GBDT | numbers, names, clusters, rivals | LightGBM | + CatBoost |
| M2 cross-encoder (pairwise, S1–R and R–R) | name noise, transliteration, aliases | Qwen3-Reranker-0.6B / bge-reranker-v2-m3 | Qwen3-Reranker-4B/8B |
| M3 listwise LLM judge | sees S1 + **all** its candidate clusters at once, so exclusivity and "twin vs real" are judged in context | Qwen3-4B LoRA on the uncertain band | Qwen3-8B on every S1 |
| M4 bi-encoder | cosine feature (also used in blocking, §3) | bge-m3 / Qwen3-Embedding-0.6B | Qwen3-Embedding-4B/8B |

**Level 2:** LightGBM/CatBoost on [M1 features + M2 logit (and its rank/margin within the S1) + M3 choice probability + M4 cosine] → isotonic calibration → decision layer (has-match head, whole-cluster choice, global assignment, per-source caps S2 ≤ 5 / S3 ≤ 6, expected F0.5).

**Compute notes:**
- T4s have no bf16. 8B models need 2×T4 tensor parallelism or AWQ/4-bit for inference, and QLoRA for training.
- M3 on all 1.73M test S1s ≈ 0.7B prompt tokens: roughly 60+ GPU-hours at 8B, so it's spread across accounts. Start with the uncertain band and widen only if CV shows the gain.
- Model sizes are chosen by bake-off (CV F0.5 gain per GPU-hour), not assumed.

## 4b. Cross-stage reuse ("re-crossing"): later knowledge fed back to earlier stages
| Later asset | Fed back into | How |
|---|---|---|
| M2 cross-encoder | Blocking (bi-encoder) | Distil M2 into the M4 bi-encoder (it learns to rank like the cross-encoder) → smaller candidate set at the same recall. Blocking still uses **no pair-scoring model** |
| Matcher's number-relation logic | Blocking gate B4b | The same parser/relations, as a deterministic rule |
| Learned noise/decoy word lists | Normalisation (B1) and the gate | Noise words removed for retrieval; decoy words as rejection evidence |
| Record clustering (unsupervised on test, allowed) | Blocking B3 | Retrieve **clusters**; siblings with Indian-script names or no address come along with their cluster |
| Confident matches | Second pass | Merged S1 profile (every name variant + filled-in address parts) → re-retrieve → score only the new candidates. The new pairs are added to candidate_pairs because the model runs on them |
| Out-of-fold errors | Normaliser + features | Weekly error-analysis loop (checklist D) |
| Test France pseudo-labels | M1/M2 fine-tuning | Self-training (allowed) |

## 5. France (now with more tools allowed)
- **Hand-written small French table** (allowed): rue/r, avenue/av, boulevard/bd/blvd, allée/all, impasse/imp, place/pl, chemin/ch, route/rte, quai, cours, faubourg/fbg; bis/ter/quater; Nº/N°; saint/st/ste; legal forms SARL/SAS/SASU/EURL/SA/SCI/SNC/EI; "& Cie".
- **Self-training** (allowed): pseudo-label high-confidence French test matches (cluster agrees, numbers identical, name ≥ threshold), fine-tune LightGBM / the cross-encoder on them, then re-score.
- **Synthetic French pairs** (allowed): apply the learned noise operators to French test S1 records. S1×S1 French look-alikes serve as guaranteed negatives.
- **LB probe:** France is in both splits; compare "France predictions on" vs "France empty" to measure the France contribution.

---

## 6. Revised build order (each step ends in a measured artefact)
| # | Step | Output / exit criterion |
|---|---|---|
| S1 | Learned artefacts (Indic dictionary + IndicXlit fallback, state aliases, noise/decoy words) + small hand-written tables | JSON artefacts + unit tests |
| S2 | **Blocking v5 "tight":** B1–B5 with a sweep over α, the gate, β, G, K on the Kaggle harness | **Frontier table: recall / entity-complete vs candidates/S1**; pick 2–3 operating points |
| S3 | Full train-shard + full-test candidate generation at the chosen points; `candidate_pairs.tsv` writer + validator | Validator PASS; timing |
| S4 | Pair features + LightGBM stage 1 + exact scorer → first LB submission (+ all-empty probe) | CV F0.5, LB score |
| S5 | Cluster features, stage 2 context, decision layer, assignment | CV gain on the twin slice |
| S6 | Pick the final blocking point by **end-to-end** F0.5 vs size | Decision recorded |
| S7 | France: hand tables, self-training, synthetic pairs, LB probe | France contribution measured |
| S8 | Cross-encoder feature (then the optional judge) | ≥ 0.5 pt CV gain, or drop it |
| S9 | Final fits, packaging (src/, README, pinned requirements, documentation with every model's licence) | Reproducible zip |

## 7. Compliance checklist (for the documentation and code review)
- Blocking contains no learned model; candidate_pairs = the first model's input (answer 3); matches ⊆ candidates is asserted.
- Every model: name, licence, parameter count, and the fact it's fine-tuned only on provided data (answers 7/10).
- No geo/postal packages; hand-written dictionaries are small and included in `src/`; learned tables are built from provided records by code (answers 2/8).
- Country handled as an open set (answer 15); unit test with a made-up country.
- Test-time statistics, self-training and synthetic pairs documented as permitted (answer 11).
