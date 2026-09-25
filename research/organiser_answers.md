# Organiser answers and problem-statement update (received 2026-09-25)

Source: `~/Downloads/6ab674645103d_emails_comms_amazon_ml_challenge_2026-2.pdf` (updated statement) + consolidated answers to team questions.

## Update to the problem statement (top of the PDF)
> **Update: candidate_pairs.tsv is part of your final submission**
> 1. **Blocking has to scale.** Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option. Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.
> 2. **Candidate generation counts toward the final ranking.** We will review your candidate_pairs.tsv and the code that produces it when deciding final rankings, alongside your matching_results.tsv score. **The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard.**

## Answers (as received)
1. One S1 entity may match many S2/S3 records.
2. No external data. Prohibited: external DBs, APIs, geocoding/entity lookup, internet augmentation, and packages bundling geo/postal/business data (libpostal, geocoders, postal-code or gazetteer datasets). Allowed: pure-algorithm libraries (RapidFuzz, jellyfish, scikit-learn, LightGBM, pandas), general-language pretrained NLP/embedding models within the limits, any algorithm using only the provided records, **small hand-written normalisation dictionaries**.
3. **candidate_pairs.tsv = the final candidate set actually fed to the matching model; in a cascade, the input to the FIRST scoring model.** Every id in matching_results.tsv must appear in candidate_pairs.tsv.
4. matching_results.tsv is uploaded for the leaderboard; candidate_pairs.tsv goes in the final zip.
5. (Eligibility: MCA 2027 graduates are eligible.)
6. Include **every** S2/S3 id believed to be the same business; don't keep only the top one of content-identical records. The scorer credits each correct id and penalises wrong ones.
7. Pretrained open-weight models allowed if MIT/Apache-2.0, ≤ 8B params, run offline, **fine-tuned only on provided data**. **The limit is per model**; every model (embedder, reranker, matcher, **any preprocessing/augmentation model**) must independently comply, and its licence is checked. Hosted LLM APIs are not allowed.
8. Same as 2, plus: **large static country/state/city tables are prohibited**; fastText and wordfreq are fine.
9. AI coding assistants are allowed.
10. Same as 7.
11. **Unsupervised statistics on the test files (TF-IDF, token frequencies, blocking index) are allowed. Self-training and generating synthetic pairs from the provided records are also fine.**
12. AWS Entity Resolution is prohibited.
13. AI coding assistance is allowed.
14. Row order of the submission can be changed.
15. Country may be used as a feature and the script may be detected; treat country as an open set.
16. No page limit on the methodology document.
17. France is test-only and appears in **both** the public and private splits.

## Still unconfirmed
- Whether an S2/S3 record can match more than one S1 **in test**. Not stated explicitly; it follows from "S1 is deduplicated" and is empirically 0 of 7.64M in training. We treat it as a hard rule in the decision layer but **not** as a pruning rule in blocking, to stay safe if test differs.
