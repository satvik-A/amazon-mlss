# Business Entity Resolution: Strategy and Research Notes

> **Superseded in priority by [plan.md](plan.md)** (written after exploring the real data). The ideas here still apply.

Status: **written before the data arrived** (2026-09-25). Everything here is a hypothesis to test.
Section 8 is the ordered experiment backlog. Section 1.2 lists the data facts that decide which ideas are worth it.

---

## 0. The 30-second version

The base is **not novel and is not optional**. The teams that won the closest public competition (Kaggle Foursquare Location Matching, 2022) all used the same shape:
- a candidate search ("blocking") that almost never misses a true match,
- a LightGBM model on similarity features,
- a fine-tuned transformer whose score becomes another feature.

Build that first. Rank is won in six places where this problem has unusual structure. Almost all of them are cheap:

| # | Direction | Why it pays here | Effort |
|---|---|---|---|
| D1 | **Metric-aware decision layer**: context-aware probabilities, a "does this entity have any match?" head, and expected-F0.5 decoding | Scoring is per entity, and singletons score 0 or 1. In simulation, the naive 0.5 cutoff loses 4–14 pts against a tuned rule | Low |
| D2 | **Competition / exclusivity stage**: a second-round model with rank and margin features in both directions | S1 is deduplicated, so each S2/S3 record belongs to at most one S1. Chain businesses are where precision dies. D1 also needs this | Medium |
| D3 | **Conflict-aware structured comparison**: every address/name part is marked match, conflict, or missing | "Different PIN code" is near-certain proof of a non-match. Generic similarity blurs it into "0.8 similar" | Medium |
| D4 | **Generalisation-first design for France**: validate by holding out a country, abbreviation logic with no dictionary, multilingual models, S1×S1 negatives | France is only in test. Teams that hard-code US/India rules will collapse on that share | Medium |
| D5 | **Reverse-engineering the noise generator**: mine the edit rules from training pairs and measure the "unexplained leftover" | The brief's noise list reads like a synthetic generator's rule list. That noise is finite and learnable | Medium |
| D6 | **Using all three sources together**: an S2/S3-vs-S2/S3 model, support from sibling records, pruning inconsistent matches, merging profiles | The S3 name may be far from S1's abbreviation but identical to S2's. The labels for this come free from the ground truth | Medium–High |
| D7 | **Neural models**: a multilingual cross-encoder as a feature. An LLM judge only on uncertain pairs (stretch) | Typos, transliteration, and world knowledge (Bombay = Mumbai), within the licence rules | High |

**What we are deliberately not doing:** merging everything that links (transitive closure), graph neural networks, a zero-shot LLM as the main matcher, libpostal or any external dictionary, one-hot encoding the country, exploiting ID or row order. Reasons are in section 5.

---

## 1. Problem anatomy

### 1.1 What the metric actually rewards

The score is F0.5 **per S1 entity**, averaged. That changes what "good" means:

- **Each entity weighs the same.** A singleton counts as much as an entity with 5 matches. Pairwise AUC or F1 is the wrong proxy. Every decision must be checked with an exact copy of the official scorer.
- **Singletons are all-or-nothing.** Predicting nothing scores 1.0. Predicting anything scores 0.
- **Predicting nothing for an entity that has matches scores 0.**

**Cost table** (F0.5 for one entity):

| True matches | Prediction | Score |
|---|---|---|
| 0 | {} | **1.000** |
| 0 | anything | 0.000 |
| 1 | {right} | 1.000 |
| 1 | {right, wrong} | 0.556 |
| 1 | {wrong} or {} | 0.000 |
| 2 | {right, right} | 1.000 |
| 2 | {right} | 0.833 |
| 2 | {right, wrong} | 0.500 |
| 3 | {right, right} | 0.909 |

**Break-even probability for adding one more candidate**, assuming the already-predicted ones are right (computed in `research/sims/decode_sim.py`):

| Already predicted (all right) | Add the next one only if p > | Cost of a wrong add | Cost of skipping a true one |
|---|---|---|---|
| 0 (it's the only candidate) | 0.500 | — | — |
| 1 | **0.727** | 1.00 → 0.556 | 1.00 → 0.833 |
| 2 | 0.759 | 1.00 → 0.714 | 1.00 → 0.909 |
| 3 | 0.771 | 1.00 → 0.789 | 1.00 → 0.938 |

The first pick is cheap to attempt. Every extra pick needs much more confidence. Losing recall on entities with several matches is cheap.

#### Simulation: which decision rule actually works

This is a synthetic world, so the magnitudes are only illustrative. Setup:
- 35% of entities have no match; 40% have 1; 15% have 2; 7% have 3; 3% have 4.
- The candidate search keeps 95% of true matches.
- Each entity has about 10 easy negatives. 30–50% of entities are "chains" with about 3 hard negatives.
- Each entity gets a random difficulty shift that moves all its scores together, so pairs within one entity are *not* independent.
- Probabilities come from isotonic calibration fitted on a separate calibration half.

| World | 0.5 cutoff | Tuned global cutoff | Tuned 1st/2nd+ cutoffs | Expected-F, **pooled** probabilities | Expected-F, **rank-aware** probabilities + has-match head | Oracle ceiling |
|---|---|---|---|---|---|---|
| easy | 0.843 | 0.845 (t=0.40) | 0.852 | 0.843–0.850 | **0.864** | 0.974 |
| medium | 0.648 | 0.684 (t=0.33) | 0.691 | 0.658–0.666 | **0.700** | 0.978 |
| hard | 0.362 | 0.505 (t=0.28) | 0.505 | 0.427–0.433 | **0.506** | 0.978 |

(Columns come from two runs of the same generator, so there is about ±0.005 of noise between them.)

**Takeaways:**
1. **Never use 0.5.** Tuning the decision rule on out-of-fold predictions, against the exact metric, was worth +0.2 to +14 pts. The weaker the matcher, the more the rule matters. Early in the hackathon it is the cheapest lever we have.
2. **Expected-F decoding on plain pairwise probabilities loses to a tuned cutoff, by −2 to −8 pts.** This corrects my first-pass advice.

   Why: a probability calibrated over all pairs is not the probability *for this entity*. At the same score, a top-ranked candidate is true more often than a second-ranked one, because the second-ranked one usually sits behind the real match. Scores within an entity also move together. That is why the tuned first-pick cutoff comes out near 0.3, not the theoretical 0.5.
3. **Fix the probabilities, then decode.** Calibrate separately by rank and add an entity-level "does it have any match?" head. Then expected-F decoding with no tuning at all matches or beats the best tuned cutoffs (+1 to +2 pts over a global cutoff). Tuned rank-dependent cutoffs on the same fixed probabilities land within about 0.5 pt of it.
4. **So D1 and D2 are one system.** The decision layer needs context-aware probabilities, and the competition stage is what produces them. Build both. Keep EF decoding and tuned cutoffs as two options and let cross-validation (and the country-holdout check) pick.

Reasons to still prefer EF decoding: it adapts per entity (candidate count, confidence spread) with nothing tuned, and when the prior shifts (France) it can be corrected by re-estimating one number, instead of re-tuning cutoffs we cannot validate.

### 1.2 Structural hypotheses to check first (each changes the plan)

| ID | Hypothesis | How to check in training data | If true | If false |
|---|---|---|---|---|
| H1 | Each S2/S3 ID appears under at most one S1 | Count S1 rows per S2/S3 ID in the ground truth | Use exclusivity: argmax assignment, competition features (D2) | Drop hard exclusivity; keep the margins as soft features |
| H2 | Any two S1 records are different businesses | Implied by "deduplicated"; spot-check name/address near-duplicates within S1 | Unlimited certified hard negatives, including chain branches (D4) | Use them only as weak negatives |
| H3 | Some source is deduplicated internally (≤1 match per S1 from S2, or from S3) | Distribution of matches per S1 from each source | Hard cap per source during decoding: a big precision gain | S2/S3 internal duplicates exist, so clustering them first (D6) becomes attractive |
| H4 | Matched records always share the same country label | Compare the labels on true pairs | Split the candidate search by country (plain string equality, allowed for an open set) | Keep the search across countries; use country agreement only as a feature |
| H5 | The noise is synthetic, with a finite set of rules | Look at 100 true pairs: regular abbreviations, single-character typos, template-like names | Invest in D5, and use the learned rules to make synthetic data for France (D4) | Lean more on neural models (D7) |
| H6 | "Orphan" S2/S3 records (matching no S1) are rare | Share of S2/S3 IDs absent from the ground truth | Coverage pressure: an unassigned record probably belongs to its best S1, which boosts recall | Orphans are distractors, so the any-match head matters more |
| H7 | Test US/India looks like training US/India | Train a classifier to tell train records from test records | Trust cross-validation | Re-weight training or calibrate per split |

**Other numbers to pull on day one:**
- the singleton rate, and matches per S1 split by source;
- how many records are in each file, train and test (for compute planning);
- the test country mix, i.e. how much is France;
- postal code presence rate, and its agreement rate on true pairs vs hard negatives;
- how often S1 names repeat (a chain rate);
- what share of true pairs have low name similarity (trade-name cases);
- the share of landmark-style addresses;
- the share of records missing each part (house number, street, city, postal code, state).

### 1.3 Where points will be lost: error types mapped to fixes

| Error type | Example | Fails as | Fix |
|---|---|---|---|
| Chain or franchise branches | Starbucks, 5th Ave vs Main St | False match | Address conflict features (D3), rival-score margins (D2), name-rarity feature |
| Generic names | "Sri Balaji Traders", "Main Street Pizza" | False match | Weight rare shared tokens; category conflict; name frequency in S1 |
| Same building, different business | Two shops in one mall or tower | False match | Name decides; name-residual features (D5) |
| Trade names ("DBA") | "XYZ Holdings LLC dba Joe's Pizza" | Missed match, and a candidate-search miss | Split names into aliases and take the best match; address-only candidate search; cross-field features |
| Transliteration | Laxmi / Lakshmi, Shree / Sri / Shri | Missed match | Transliteration key; character models (D7) |
| Landmark addresses | "Near SBI ATM, MG Road" vs "12 MG Road" | Missed match | Parse the landmark separately; street similarity without the landmark |
| Partial addresses | No PIN, no state | Uncertain | Treat "missing" separately from "conflict" (D3); borrow the missing part from another source (D6) |
| Digit typos | Suite 200 vs 210, Plot 12 vs 21 | Either way | A separate "near-conflict" state (edit distance 1 between digit strings) |
| Category conflicts | "Balaji Medicals" vs "Balaji Textiles" | False match | Category-word conflict feature, from a word list mined from the data |
| Unseen country | French legal forms, street types, accents | Both | D4 |

---

## 2. Validation design (fix this before any modelling)

1. **An exact local scorer.** Unit-test it against the brief's example (0.714), singletons, and empty predictions.
2. **Folds grouped by S1 entity** (5 folds).
   - Keep *all* S1 and S2/S3 records present for candidate search and for context features; hide only the labels.
   - Any score used to build context features must come from a model that didn't train on that pair (out-of-fold, "OOF").
3. **Anything that uses labels is fitted inside each fold:** the mined rule dictionary, calibration, cutoffs, statistics computed from labels.
4. **Hold out one country (LOCO)** as the France stand-in: train on US and score India, then the reverse.
   - Track it next to normal cross-validation.
   - Keep a change only if it doesn't hurt LOCO.
   - Compare changes, not absolute levels: LOCO uses half the data, so its scores run low.
5. **Measure noise.** Use a paired bootstrap over entities for every comparison. At N entities, the standard error of the score is roughly 0.45/√N, about 0.6 pt at N = 5,000; paired differences are much tighter. Don't chase public-leaderboard gaps under about 1 pt, since it only scores a subset.
6. **Candidate-search metrics:** pair recall at N, the entity-level ceiling (the score with perfect decisions on the candidates), reduction ratio, and average candidates per S1.
7. **Proposed rule for keeping a change:** CV improves by at least 0.3 pt with bootstrap p < 0.1, **and** LOCO is not worse by more than 0.3 pt.

---

## 3. Options at each stage, and which to pick

### 3.1 Normalisation

| Option | Upside | Downside | Verdict |
|---|---|---|---|
| Hand-written dictionary (Rd→road, Pvt→private) | Fast | Brittle, per-country, misses France | Keep it tiny and documented |
| Dictionary mined from true training pairs (D5) | Recovers the generator's own dictionary | Only covers US/India vocabulary | **Yes** |
| Abbreviation matching with no dictionary: same first letter, and the short token's letters appear in order in the long one (**min length 2**) | Catches Bd/Boulevard, Sté/Société, Pvt/Private, Rd/Road, St/Street/Saint with no list | Some false hits on very short tokens | **Yes**, a core France tool |
| Transliteration key: collapse aa/ee/oo/ii, sh→s, ph→f, w→v, ksh→x, drop an 'h' after a consonant | Laxmi = Lakshmi, Shree ≈ Sri, Krishna = Krushna | Can merge distinct names | **Yes**, as a feature and a search key, never as a replacement |
| LLM normalisation or parsing (Qwen3 up to 8B, offline) | Knows French; handles messy addresses | Cost; output varies run to run; counts toward the 8B limit | Optional: only if regex parsing fails on France |
| Keep raw **and** normalised versions and compute features on both | Normalisation mistakes become signals, not failures | More features | **Yes** |

**Concrete steps:**
- lowercase; NFKD accent folding (standard library `unicodedata`);
- `&` ↔ `and`; punctuation → space, but keep digit groups (`12-3-456` → tokens `12 3 456` *and* the joined `123456`);
- ordinals (`5th` / `fifth`, `1er` / `premier`), with number words normalised for English;
- split trade-name aliases on `dba`, `d/b/a`, `t/a`, `trading as`, `aka`, `formerly`;
- tag each token by role (legal form, business category, place name, number, distinctive core token). Find legal-form and category words by frequency (common trailing tokens), not from a fixed list.

### 3.2 Candidate search ("blocking")

**Target:** about 99% pair recall at no more than 30–60 candidates per S1. Tune this once we see the data size.

| Method | Catches | Misses |
|---|---|---|
| Exact keys: postal code; exact normalised name; transliteration key of the first core token + postal code | Clean cases, cheaply | Anything with a missing or garbled key |
| Letter-trigram TF-IDF on names, top k (sparse matrix product) | Typos, abbreviations, reordered words | Trade names |
| Word TF-IDF / BM25 on name + address | Pairs with many shared tokens | Heavy abbreviation |
| **Address-only** TF-IDF, top k within postal code or city | **Trade names**, where the name is completely different | Landmark-only addresses |
| Dense multilingual embeddings (e5 / bge-m3) + FAISS | Transliteration, meaning, French | Exact numbers |
| Fine-tuned embedding model, trained on true pairs with S1×S1 hard negatives (Sudowoodo-style) | Raises recall at small N | Costs training time. Only if recall falls short |
| Rare-token inverted index (share any token with high IDF) | Distinctive single words | Nothing, but it is noisy |

**Design choices:**
- **Search in both directions.** Retrieve the top S2/S3 records for each S1, *and* the top S1s for each S2/S3 record, then union. The rival-score features in D2 need each record's competing S1s to be present, and many pipelines only search from S1.
- **Combine everything,** then cut to the top N with a cheap LightGBM ranker (about 20 cheap features). **That top N is `candidate_pairs.tsv`.**
- For each method, measure its *unique* recall (true pairs only it finds). Drop methods that add nothing unique.
- Split by country only if H4 holds.

### 3.3 Pair features, by family

| Family | Examples | Targets |
|---|---|---|
| Name string similarity | Jaro-Winkler, Levenshtein ratio, token sort/set/partial ratio (rapidfuzz), on raw, normalised and core-only names | Typos, word order |
| Weighted token similarity | TF-IDF cosine (letters and words), BM25, SoftTF-IDF (TF-IDF with fuzzy token matching), Monge-Elkan with abbreviation-aware token similarity | Abbreviations, reordering |
| Rarity | Total and max IDF of shared tokens; total and max IDF of *unshared* tokens (a distinctive mismatch); how many S1 records share the core name (chain indicator) | Chains, generic names |
| Legal form / category | Legal form: match, conflict, or missing; category-word conflict | Medicals vs Textiles |
| Address parts (D3) | Each of postal code, house number, unit, street, locality, city, state marked match, near-conflict, conflict, one missing, or both missing, plus a similarity value; landmark similarity; overlap of the sets of numbers; digit edit distance | Chains, partial addresses |
| Cross-field | Name tokens that appear in the other record's address (building names, trade names) | Trade names, malls |
| Embeddings | Multilingual cosine on name, address, and the full record | Transliteration, France |
| Rule-based leftover (D5) | Unexplained leftover tokens and their IDF weight; which edit rules were used, by type | Everything; a sharp signal |
| Cross-encoder score (D7) | Fine-tuned relevance score | Hard cases |
| Context (D2) | Rank in the S1's list and in the record's list, margins, "mutual best", z-score within the list | Chains, singletons |
| Collective (D6) | Support from sibling records, size of the record's cluster | Records with several matches |
| Meta | Source pair (S1–S2 vs S1–S3), missing-field indicators, lengths | Noise that differs by source |
| Fellegi–Sunter weights | Per-field match odds by agreement level, fitted inside each fold | A cheap, classic prior |

**Avoid:** a one-hot country column; features based on raw token identity (they won't carry over to France); anything computed from labels outside the fold.

### 3.4 Matchers

| Option | Verdict |
|---|---|
| LightGBM / CatBoost | **Main model.** Consider *monotone constraints* on the main similarity features: they act as a regulariser that keeps behaviour sensible on an unseen country (test with LOCO). |
| Cross-encoder (Ditto-style: `[COL] name [VAL] … [COL] address [VAL] …`) | **Yes, as a feature.** Multilingual base: bge-reranker-v2-m3 (568M, Apache), mdeberta-v3-base (MIT), or Qwen3-Reranker-0.6B (Apache). Train on the candidate-search negatives plus S1×S1 negatives, with Ditto-style augmentation (drop a span, shuffle tokens). Score each pair in both orders and average. |
| Embedding model (two separate encoders) | Candidate search only. |
| Zero-shot LLM up to 8B | No. Fine-tuned small models beat zero-shot LLMs at this ([Peeters et al., EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-81.pdf); [Fine-tuning LLMs for EM](https://arxiv.org/pdf/2409.08185)). |
| Fine-tuned LLM judge (Qwen3-4B/8B + LoRA) | Stretch goal, **only on uncertain pairs**. Ask in **"select" style**: give it S1 plus its top 5 candidates and ask which ones match. [Match, Compare, or Select?](https://arxiv.org/pdf/2405.16884) reports that selecting or comparing beats judging pairs one at a time, and it fits the exclusivity structure. |

### 3.5 Context / collective stage
- **Round 2:** LightGBM on round-1 OOF scores plus context features (D2) and collective features (D6).
- **Assignment** (if H1 holds): each S2/S3 record goes to its best S1, then each S1 is decoded. Alternative: greedy global pass. Take the highest-probability (S1, record) pair, lock it, remove that record from other S1s, repeat.
- **Set-level model** (a transformer over an S1's whole candidate list): novel, but round 2 captures most of it. Defer.

### 3.6 Decision layer
In this order:
1. Probabilities that already include context: the round-2 output, then isotonic calibration, optionally per rank.
2. An entity-level has-match head.
3. EF decoding **or** tuned rank-dependent cutoffs, chosen by CV and LOCO.
4. A hard cap per source if H3 holds.
5. A prior correction per country (D4).

---

## 4. The novel directions in detail

Each one gives the mechanism, why it fits, the risks, how to test it, and the effort.

### D1. Metric-aware decision layer
- **Has-match head.** Target: the S1 has at least one match. Features:
  - top-1 and top-2 probability, and the gap between them;
  - number of candidates above 0.1 / 0.3 / 0.5, and the entropy of the list;
  - whether the top candidate's own top S1 is **someone else**, a strong singleton signal;
  - how complete the S1 record is, and how rare its name is.
- **Optional count model** (ordinal: 0 / 1 / 2 / 3+ matches) to guide how many to pick.
- **EF decoding.** Over the context-aware probabilities, compute the exact expected F0.5 of predicting the top k, for every k. The count of true matches is a Poisson-binomial, so a small dynamic program does it. Condition on "at least one match" and weight by the head's probability. Predicting nothing scores the head's P(no match). Already built and tested in `research/sims/decode_sim2.py` (run from that folder); port it when there is real data.
- **Prior correction for France.** The test singleton rate and match rate in France may differ from training. Re-estimate the class prior on the unlabelled France candidates with the Saerens–Latinne–Decaestecker EM method, then rescale the probabilities. Cheap, and it moves the decision boundary where we can't tune one. Test: does it help the held-out country in LOCO?
- **Leaderboard probe (costs 1 submission).** An all-empty submission scores exactly the **singleton rate of the public test split**, France included. That directly calibrates the head's prior. Optionally, compare two submissions that differ only in France predictions to isolate France performance. Use this sparingly: the public leaderboard is a subset.
- **Risks:** overfitting cutoffs to cross-validation. Mitigate by tuning on OOF across all folds and checking against LOCO.

### D2. Competition / exclusivity stage
- **Mechanism.** Round-1 OOF scores → context features → round-2 model. Features for a pair (s, r):
  - r's rank in s's list, and s's rank in r's list;
  - `score(s,r) − best score of r against any other S1` (margin against the strongest rival), and `score(s,r) − s's next-best score`;
  - a **mutual-best** flag (r is s's top, and s is r's top), the classic strong precision signal;
  - z-score of the score within s's list, and within r's list;
  - how many S1s score r above 0.5.
- **Why it fits:** exclusivity (H1) means rivals compete for the same record. Chains are exactly the case where a single pair looks good but a rival looks better.
- **Risk:** leakage if round-1 scores are not OOF; the rival lists need search in both directions (3.2).
- **Test:** round 2 vs round 1 on CV and LOCO. Break out the chain subset (S1 names that repeat).
- **Effort:** about a day once round 1 exists.

### D3. Conflict-aware structured comparison
- **Parsing without tying to a country** (detect by pattern, not by label):
  - **postal codes:** a standalone 5-digit token (US ZIP, French CP), `\d{5}-\d{4}`, or a 6-digit token (Indian PIN, sometimes written `560 001`);
  - **house or door numbers:** leading number, `No.`, `#`, `H.No`, `D.No`, `Plot`, `Flat`, `Suite`/`Ste`, `Apt`, `Unit`, `bis`/`ter`, and municipal forms like `1-2-345/A`;
  - **landmark phrases:** near, opp, opposite, behind, beside, next to, adjacent to, in front of, à côté de, en face de, près de;
  - **street-type words:** mined from the data, plus abbreviation matching;
  - **localities:** Nagar, Colony, Layout, Sector N, Phase N, Block X, "5th Cross, 3rd Main", ZI/ZA/ZAC.
- **Encoding:** a category per part {match, near-conflict, conflict, one-missing, both-missing}, plus a similarity value. LightGBM handles categories and missing values natively.
- **Why:** it separates "no information" from "contradicting information". That separation is the precision lever F0.5 rewards.
- **Test:** false-match rate among the hard negatives, and ablation on CV and LOCO. Inspect parser coverage by hand on 100 records per country, including test France (looking at unlabelled test inputs is ordinary data exploration).

### D4. Generalisation-first design for France
Assume France could be up to a third of the test set. If a model that overfits US/India scores 0.60 on France where a robust one scores 0.80, that is about **6–7 pts overall**, likely larger than any other single idea.
- LOCO validation from day 1 (section 2).
- Only dictionary-free building blocks: abbreviation matching, transliteration key, accent folding, legal forms and categories found by frequency, relative (similarity) features.
- Country statistics (IDF, name frequency) grouped **by the country string**, so an unseen label gets its own statistics automatically. This complies with the open-set rule; unit-test with a made-up country label.
- Multilingual encoders only.
- Monotone constraints in LightGBM (3.4).
- **Guaranteed negatives from S1:** pairs of S1 records from the same country with similar names or addresses are certified non-matches. On *test* France they are free hard negatives with no labels needed.
- **Synthetic French positives:** apply the edit rules learned in D5 (operator frequencies per source) to test-France S1 records:
  - drop the legal form;
  - shorten street types with the generic rule (first letter + consonants: boulevard→bd/blvd, avenue→av);
  - add typos; reorder tokens; drop the postal code at the learned rate.

  Then fine-tune the cross-encoder or LightGBM on these plus the certified negatives: self-supervised adaptation to the new domain.
- ⚠️ Training on test inputs is a **grey area** under "only the provided training data". Ask the organisers and keep it behind a flag. Everything else in D4 is clean.
- French cases to use as **unit tests** for the dictionary-free functions (not as hard-coded rules):
  - legal forms: SARL, SAS, SASU, EURL, SNC, SCI, & Cie, Ets, Sté/Société;
  - street types: rue, av, bd, pl, ch, imp, rte, quai, fbg;
  - Saint/Ste with hyphens, Cedex, "Paris 8e" = 75008, BP boxes, bis/ter.
  - Note the three-way "Ste" collision: Suite (US), Sainte, Société. Abbreviation matching treats it as "compatible", not "equal", which is correct.

### D5. Reverse-engineering the noise generator
- **Mining the rules.** For each true training pair:
  1. Normalise and tokenise both names (and addresses).
  2. Find the best token alignment (Hungarian algorithm on a Jaro-Winkler similarity matrix).
  3. Log substitutions (a→b with a≠b) and unaligned (dropped) tokens.
  4. Compare with hard-negative pairs (PMI) to keep only real equivalences.

  Outputs: a substitution dictionary, a set of tokens that can be dropped without penalty (legal suffixes, "the", …), typo statistics (edit type and position), and a reorder rate, **per source**.
- **"Unexplained leftover" feature** (a simple stand-in for a noisy-channel likelihood ratio).
  - Align the names with cost tiers: exact = dictionary equivalent = abbreviation-compatible = same transliteration key < typo (edit distance 1, length ≥ 5) < unexplained.
  - Measure what remains: count of unexplained tokens, their IDF weight, and the max IDF of an unexplained token, per side.
  - Add **rule counts by type** so the model can learn each source's noise signature.
- **Why:** if the generator is synthetic, a real match can be explained with **zero** leftover. Non-matches almost never can, except chains, where D3 steps in.
- **Test:** precision of "leftover = 0" on training data, then feature ablation. Mine the dictionary inside each fold.
- **Bonus:** the learned rule frequencies drive the synthetic data in D4.

### D6. Using all three sources together
- **Record-vs-record model** (S2–S3, S2–S2, S3–S3), with labels free from the ground truth: records under the same S1 are matches, records under different S1s are not. Reuse the pair features, symmetrically.
- **Sibling support:** for (s, r), take the max and mean of P(r, r′) over records r′ that the OOF round-1 scores **confidently** assign to s (p > 0.8).
- **Inconsistency pruning:** if two accepted records for one S1 have very low P(r, r′), drop the weaker one.
- **Profile merging and re-scoring** (R-Swoosh-like): build s's profile from s plus its confident matches (every name variant, the union of address parts, e.g. a PIN code from S2). Then compute the best score across the profile for every remaining candidate. Only merge above p > 0.9, and never chain merges together.
- **Cluster first, then link** (if H3 shows many S2/S3 duplicates): make strict record clusters (high cutoff, size cap), then match clusters to S1 using aggregated features.
- **Leakage trap:** the "confident members" must come from OOF predictions, **never** from labels. Otherwise training and test features don't match.
- **Test:** gains on entities with at least 2 matches, and the change in false-match rate.

### D7. Neural models
- **Cross-encoder:** as in 3.4. Train with hard-negative curriculum (easy first, then candidate-search hard negatives, then S1×S1 chain negatives).
  - On the M1 Pro, fine-tuning a base-size model on MPS works but is slow. Use Kaggle (2×T4, about 30 GPU-hours/week) or Colab.
  - Inference: about 500k pairs at 128 tokens takes roughly 5–10 min on a T4 for a base model.
- **Offline address parsing with an LLM** (Qwen3-4B on vLLM), only if regex parsing fails on France. Roughly 60k records × 60 output tokens ≈ 3.6M tokens, about 1–2 GPU-hours.
- **LLM judge on uncertain pairs** (Qwen3-8B + LoRA, select-style prompting) for pairs with probability between 0.3 and 0.8. Its logit becomes a feature in the last-stage model. This is also the **only legal source of world knowledge**: Bombay = Mumbai, Gurgaon = Gurugram, Bengaluru = Bangalore, French abbreviations.
- **Character-level model** (ByT5-small, Apache) as a transliteration-robust cross-encoder. Experimental.

---

## 5. Rejected or deprioritised

| Idea | Why not |
|---|---|
| Merging everything that links (transitive closure / union-find) | Chain businesses link into one huge cluster and precision collapses. D6 gets the benefit without it. |
| Graph neural network over the record graph | Days of work for a small gain over hand-built graph features. |
| Zero-shot LLM as the main matcher | Weaker than fine-tuned small models; expensive; limited to 8B anyway. |
| Splink / Fellegi–Sunter as the main model | Assumes fields are independent; we have labels. Keep its match weights as features. |
| libpostal, gazetteers, geocoders, public abbreviation lists | External data, so disqualification risk. libpostal is offline but trained on OpenStreetMap: too ambiguous to risk. |
| One-hot country / rules per country | Violates the open-set rule and doesn't carry over to France. |
| ID-number or row-order signals | Check only that validation isn't fooled; top packages are audited. |

---

## 6. Compliance and risk register

- **External data:** nothing beyond the provided files and the model weights. Hand-written normalisation rules: keep them few, generic, and documented in the methodology write-up.
- **Models:** MIT or Apache 2.0, at most 8B. **Allowed:** bge-m3 (MIT), bge-reranker-v2-m3 (Apache), multilingual-e5 (MIT), gte-multilingual (Apache), mdeberta-v3 (MIT), xlm-roberta (MIT), Qwen3 / Qwen3-Embedding / Qwen3-Reranker (Apache), ByT5 (Apache). **Not allowed:** Llama, Gemma, Qwen2.5-3B/72B, Ministral-8B. Re-check every model card when we use it.
- **Open-set country:** handle countries only by string equality or grouping, with a fallback for unseen labels. Unit test: rename "India" to "Xland" and the pipeline must still run and produce every row.
- **Honest candidate file:** `candidate_pairs.tsv` = exactly the set the final model scores. Assert matches ⊆ candidates, no duplicates, and every test S1 present. Run `utils/validate_submission.py` before every upload.
- **Reproducibility:** one entry script; fixed seeds; pinned `requirements.txt`; cached intermediates; documented runtime and hardware.
- **Libraries** (all permissive): pandas/polars, rapidfuzz, scikit-learn, lightgbm, faiss-cpu, sentence-transformers, transformers, peft, vllm (GPU only). Accent folding via the standard library's `unicodedata`.

## 7. Compute plan
- **M1 Pro 16 GB:** feature engineering, TF-IDF, LightGBM, small embedding models (e5-small), decoding. About 1M pairs × 150 float32 features ≈ 600 MB, which is fine.
- **Kaggle or Colab GPU:** cross-encoder fine-tuning, bge-m3 over large pools, LLM LoRA or parsing.
- Cache every stage to parquet so later stages can iterate without recomputing.

---

## 8. Experiment backlog (in order; each has a hypothesis and a pass/fail check)

| # | Experiment | Hypothesis / purpose | Pass if |
|---|---|---|---|
| E0 | Data checks H1–H7 and the day-one numbers | Decides D2, D3, D6 and the source caps | Written into this file |
| E1 | Scorer, validator wrapper, trivial baselines (all-empty; exact-name match) | All-empty score = singleton rate, the floor | Scorer passes unit tests |
| E2 | Candidate search v1: letter TF-IDF + postal code + both directions; recall-at-N curves | The ceiling is set here | Recall ≥ 97% at N ≤ 50 |
| E3 | LightGBM v1 on basic similarities + tuned rank cutoffs → **first leaderboard upload** | Establish a baseline | Valid submission, CV logged |
| E4 | Leaderboard probe: all-empty submission | Test singleton rate for the has-match prior | 1 submission used |
| E5 | D3 parts marked match/conflict/missing + rarity + category conflict | Precision gains on chains and generic names | CV +, LOCO not − |
| E6 | D2 round 2 + D1 head + EF decoding | Context-aware probabilities make decoding work | Beats E3 decoding by ≥ 1 pt |
| E7 | LOCO harness; ablations for transfer; monotone constraints; France unit tests | Robustness for France | LOCO gap narrows |
| E8 | D5 mined dictionary + leftover feature | Reverse-engineered noise | "Leftover = 0" precision > 0.95 |
| E9 | D7 cross-encoder → feature | Hard cases, transliteration | CV + ≥ 0.5 pt |
| E10 | D6 record-vs-record model, sibling support, profile merging | Entities with several matches | CV + on the ≥2-match subset, false matches not up |
| E11 | Dense or fine-tuned embedding search (only if E2 recall is short) | Raise the ceiling | Recall + ≥ 1 pt at the same N |
| E12 | D4 self-supervised adaptation for France (**only if the organisers allow it**) | Domain adaptation | LOCO + (simulating the setup with the held-out country's S1) |
| E13 | LLM judge on uncertain pairs (stretch) | World knowledge | CV + on the uncertain band |
| E14 | Seed/fold ensembling, final calibration, package + methodology write-up | Final submission | Validator PASS; reproducible from scratch |

## 9. Questions for the organisers (ask early)
1. Does the 8B limit apply to each model or to the whole ensemble (e.g. a 0.6B cross-encoder plus an 8B LLM)?
2. May unlabelled **test** records be used for fitting (TF-IDF, self-supervised adaptation, pseudo-labels)?
3. May a pretrained model's built-in knowledge be used (e.g. city-name aliases)? Presumably yes.
4. How many leaderboard submissions are allowed per day?

## 10. References
- Foursquare Location Matching, 1st place: [Kaggle write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution). Multi-stage LightGBM (with a first LightGBM cutting to the top 40 per id) → BERT-family models added in later stages.
- Peeters, Steiner, Bizer, *Entity Matching using Large Language Models*, [EDBT 2025](https://openproceedings.org/2025/conf/edbt/paper-81.pdf).
- *Fine-tuning Large Language Models for Entity Matching*, [arXiv 2409.08185](https://arxiv.org/pdf/2409.08185).
- Wang et al., *Match, Compare, or Select?*, [arXiv 2405.16884](https://arxiv.org/pdf/2405.16884).
- Li et al., *Ditto: Deep Entity Matching with Pre-Trained Language Models* (VLDB 2021). Wang et al., *Sudowoodo* (ICDE 2023).
- Fellegi & Sunter (1969). Cohen, Ravikumar & Fienberg, SoftTF-IDF (2003). Benjelloun et al., Swoosh (2009).
- Jansche (2007); Dembczyński et al. (2011); Ye et al. (2012): optimising F-measures and choosing the best prediction set.
- Saerens, Latinne & Decaestecker (2002): adjusting classifier outputs to a new class prior.
