# Plan v2: based on the real data (2026-09-25)

This supersedes the priorities in `strategy.md`. The ideas there still apply, but the data re-ranked them.
EDA scripts are in `research/eda/`. Run heavy jobs through `research/guard.sh 6 <cmd>`, which kills anything above 6 GB.

---

## 1. What the data says

### Scale
| | S1 | S2 | S3 | S2+S3 per S1 |
|---|---|---|---|---|
| train | 2.21M (US 1.32M, India 0.88M) | 5.03M | 5.29M | 4.67 |
| test | 1.73M (US 0.66M, India 0.81M, **France 0.26M = 15%**) | 4.89M | 5.08M | **5.76** |

The test set has about 22% more records per S1 than training. That suggests more decoys, so precision matters even more on test.

### Structure (verified on training data)
- **Exclusivity holds exactly.** 0 of 7.64M matched records belong to more than one S1.
- **Only 5.6% of S1 entities have no match.** Matches per S1: 1 (5.4%), 2 (17%), 3 (24%), 4 (22%), 5 (15%), 6 (7.5%), 7+ (4%). Each source contributes 1–6 copies.
- **About 26% of S2/S3 records are orphans** (no S1). Most of them are **copies of decoy "twin" entities**.
- **Country always agrees** on true pairs (100%), so partitioning by country is safe.

### The generator (the key insight)
Each S1 entity E has several noisy copies in S2/S3. Separately, the generator creates **twin decoy entities** E′:
- **Name:** the same name, often with **one extra qualifier word**. In training these are Holdings, Group, Enterprises, Infratech, Overseas, Ventures, Exports, Industries, Public, Lakeside, Eastgate, Harbor, Downtown, West, and so on.
- **Address:** the same street, with a **nearby house number** (5619→5623, 702→707, 317→328, 139→160, 1-edit changes).
- **Copies:** E′ has its own set of noisy copies.

Measured on 3,000 sampled S1s:
- **84%** of singleton S1s have a high-similarity twin among the S2/S3 records.
- Among pairs with name and address token-set similarity both at least 80: 3,497 true vs **3,429 false**. The usual fuzzy-matching approach is a coin flip here.
- How the address's number sets compare, within true pairs and within look-alike false pairs:

  | Number sets | True pairs | Look-alike false pairs |
  |---|---|---|
  | identical | 71% | 1.3% |
  | disjoint | 4.7% | **72%** |
  | overlapping | 10% | 26% |
  | S2/S3 record has no numbers | 8.9% | 0.5% |

- First-number relation in look-alike false pairs: shifted by 25 or less (31%), a 1-digit edit (25%), unrelated (28%). In true pairs, the first number is identical 78% of the time.
- Extra name words on **true** pairs are noise words: Center, Services, dba, formerly, Shri/Sri, Mr, Dr, legal suffixes. On **decoys** they are the qualifier list above.
- **France:** the English qualifier words appear in only **0.1%** of French records, vs about 10% in US/India. French decoys use other words, so the feature has to be generic ("an unexplained extra word"), not a word list.

### The noise catalogue (build a normaliser for each)
| Noise | Examples | Rough frequency |
|---|---|---|
| Letters scrambled inside a word | Partnaedsr, Efohdcironlogy, Madnufacnuring | common |
| Injected accents | Cáré, Éndocrinology, Prívate | about 6% of names |
| Look-alike digits in words | 5mart, Si1ver, GR0UP, 6andhidham | common |
| Legal-suffix swaps or moves | LP/LLC/Inc/Co/Ltd; "Pvt. EFS Print Ventures Ltd."; "Private KSE Pharmaceuticals [Limited]" | very common |
| Wrappers and junk | `[..]`, `(..)`, `--`, `<<`, "The", `(ID: 10654)`, `(France)` | common |
| Word reordering | "KSE LIMITED PRIVATE PHARMACEUTICALS" | common |
| Dropped words | "Dermatology Smart", "Dermatology" | common |
| **Alias with a random name** | "Nexarcriza F/K/A Shaw Properties Partners"; "Tavobrix t/a Caucase Club SARL" (France) | **about 2% of S3** |
| **Website or handle as the name** | monamanufacturing.com, @sarangidigital #67560 | **about 4.3% of S2/S3** |
| Acronyms | MNH = Mark Nursing Home | rare |
| **Names in Indian scripts** | సాయి ఇండో హాస్పిటాలిటీ..., राम मार्केटिंग प्राइवेट लिमिटेड | **9.4% of S2, 5.3% of S3** |
| Address missing | literal `None` | 3.3% of S2/S3 (0% in France) |
| Address parts reordered | "OH, MANCHESTER, 601 6TH ST" | very common |
| House-number noise | `#146`, `00904`, `##594`, `08-2-248`, 5658→658 (dropped digit), `7nd`, "Seventh" | common |
| State written differently | IL / Illinois / महाराष्ट्र / MH / TG vs Andhra Pradesh | common |
| City dropped or varied | Winston-Salem vs Lewisville; Calcutta/Kolkata | occasional |
| France | SARL/SAS/EURL/SA/SASU/SCI/EI; R = Rue, AV, Bis/Ter, Nº; **almost no postal codes** | — |

### Candidate-search probe (first attempt, on a 20k-S1 sample)
IDF-weighted token overlap, capped at frequency ≤ 3,000: **recall 72% at 10 candidates, 76% at 30, 80% at 100.** Too low. The misses have clear causes:
- records with no address whose name tokens are too common for the cap;
- Indian-script names;
- acronyms;
- dropped digits (1120→120).

Everything below fixes these. A larger probe used too much memory and was stopped; the next one will be chunked (see §4).

---

## 2. What this changes

1. **Think in clusters, not pairs.** Every S1's matches are copies of *one* underlying entity. Every look-alike is a copy of a *different* entity, and that entity also has several copies. So the real decision per S1 is **"which cluster of S2/S3 records (if any) is this entity?"**, followed by trimming outliers. Getting the cluster wrong costs everything. For example, with 4 true copies:

   | Prediction | F0.5 |
   |---|---|
   | 3 right | 0.94 |
   | 4 right + 1 wrong | 0.83 |
   | the true cluster + a 3-copy decoy cluster | 0.63 |
   | only the decoy cluster | **0** |

2. **Numbers decide precision.** The house number and unit number, compared at the **cluster** level, separate twins from true matches. The majority number across a cluster's copies removes one-off typos (a single copy with 5658→658 is outvoted by the copies that say 5658).
3. **Recall matters more than I assumed.** Singletons are only 5.6%, and entities average about 3.5 matches. Missing copies costs points on almost every entity. Candidate-search recall is the ceiling, and the records hardest to find (Indian-script names, websites, aliases, missing addresses) are exactly the ones a cluster brings along through their siblings.
4. **Singleton S1s are mostly "the twin is here, but I'm not it".** The has-match head reduces to "is the best cluster's number conflicting?"

---

## 3. The pipeline

```
0. Normalise (per record, streaming)
   name:  NFKD accent fold; look-alike digits → letters (5→s, 1→l, 0→o, 6→g) inside words; strip wrappers/junk;
          split aliases on f/k/a | d/b/a | t/a | formerly | aka (keep every alias);
          split websites/handles (monamanufacturing.com → "monamanufacturing", matched later against S1 names with spaces removed);
          Indian scripts → Latin via a deterministic transliterator (ICU "Any-Latin"), then a phonetic key;
          separate legal forms (found by frequency, which also covers SARL/SAS for France); sorted-letter key (catches scrambled words)
   addr:  `None` → missing; parse numbers (strip leading zeros and #, split 8-2-248/1 into parts); street words; unit;
          state/region aliases mined from training true pairs (IL = Illinois = native script); order-free token sets

1. Cluster S2/S3 into entities (within country)
   Candidate search: exact normalised address; (street words + house number); name keys + city
   R–R pair model (LightGBM); labels are free from the ground truth (same S1 = match, different S1 = not);
   strict cut-off, then connected components with a size cap (a true entity has at most about 11 copies)
   Cluster profile: every name variant, the majority house/unit number, the union of address tokens

2. S1 → candidate clusters (top K per S1; both directions: S1→cluster and cluster→S1)
   Keys: IDF token overlap against the cluster profile, name pairs, exact name key, acronym, phonetic key,
         house number ± digit-drop tolerance
   candidate_pairs.tsv = every record in the candidate clusters (+ any unclustered records retrieved directly)

3. Scoring model (LightGBM), on (S1, record) and (S1, cluster):
   - Number relation for the record and for the cluster majority: identical / subset / overlap / disjoint / missing;
     first-number: equal / shift ≤ 25 / 1-edit / truncation / missing; unit number the same
   - Name: similarity over every alias; sorted-letter match; unexplained extra words (count, IDF,
     qualifier-likeness = how often the word is an extra word in S1-vs-S1 look-alikes)
   - Cluster: size, cohesion, how many copies support the S1's number, sources covered
   - Competition: rank and margin of the best cluster vs the runner-up; "each is the other's top pick";
     whether this cluster scores better with another S1 (exclusivity)

4. Decision per S1
   Choose the best cluster or none (has-match head) → keep its records → drop low-scoring outliers;
   optionally add stray unclustered records using expected-F0.5 decoding.
   Every S2/S3 record is assigned to at most one S1 (greedy global pass by probability).
```

### Why this should beat a plain pair classifier
- Twins differ in exactly one number or word. A single noisy copy can hide the difference (a typo), but the majority across 3–6 copies rarely does.
- The hard-to-find copies (Indian script, website names, aliases, `None` addresses) link to their siblings by address or name inside the R–R clustering, so they come along with the cluster.
- It matches the metric: F0.5 punishes choosing the wrong cluster, and scores near 1.0 once the cluster is right.

**Fallback:** if clustering is noisy, the same features run pairwise, with cluster statistics as soft features ("sibling support"). Both use the same code.

---

## 4. Compute plan (16 GB M1 Pro; no more runaway jobs)
- **Integer ids everywhere.** Strings only at the edges.
- **Process by country, and in chunks of about 50–100k S1.** Cap posting lists (skip tokens with frequency above N). No unbounded self-joins.
- **Every script runs under `research/guard.sh 6`.**
- **Develop on a sample:** about 150k training S1 plus the S2/S3 records they retrieve. Scale up only once code is memory-profiled.
- **Full runs:** test has 1.73M S1 × 10M records. Do the heavy steps (clustering, candidate search over all of test, feature building) either chunked locally overnight, or on a **Kaggle notebook (about 30 GB RAM)**.
- **Models:** LightGBM on CPU is enough for most of the gain. A cross-encoder (multilingual, ≤ 600M, Apache/MIT) is optional later on Kaggle GPU, for name-noise cases only.

---

## 5. Validation
- Exact local scorer (macro F0.5, singletons count). Group folds by S1.
- Report results by slice: singletons, entities with twins, entity size 1/2–3/4+, Indian-script copies, missing address.
- **France stand-in:** train US, test India (and the reverse), with the qualifier-word features **turned off**, to check that the generic features carry the load.
- **Leaderboard probe:** an all-empty submission scores exactly the singleton rate of the public test set. That calibrates the has-match head for test, which has more decoys per S1.

---

## 6. Build order (each step produces a scored submission or a go/no-go number)
| # | Step | Pass if |
|---|---|---|
| 1 | Normaliser + scorer + validator wrapper; all-empty and exact-address baselines | Scorer unit tests pass; first valid upload |
| 2 | Chunked candidate search v2 (fixes from §1), recall at K on a sample | ≥ 95% at 30 with records, or with clusters |
| 3 | Pairwise LightGBM with number-relation + alias + extra-word features; tuned decision rule | Big jump over the baseline; LB submission |
| 4 | R–R clustering + cluster features + cluster-level decision | Beats step 3 in CV, especially on the twin slice |
| 5 | Competition / exclusivity + greedy global assignment | Improves CV, especially precision |
| 6 | Hardening for France (generic qualifier features, transliteration, French address parsing), cross-country check | Cross-country gap narrows |
| 7 | Optional cross-encoder on name pairs; ensembling; final package + documentation | — |

## 7. Questions for the organisers (unchanged)
1. Is the 8B limit per model or for the whole ensemble?
2. May unlabelled test records be used (e.g. clustering test S2/S3 is unsupervised on test inputs; it's needed at inference anyway, so it should be fine)?
3. How many submissions per day?
