import polars as pl, sys, re, collections
sys.path.insert(0, "kaggle/blocking_v2"); from translit import translit
INDIC = r"[ऀ-ൿ]"
j = pl.scan_parquet("research/eda/cache/train_pairs_joined.parquet").select("business_name", "business_name_r") \
      .filter(pl.col("business_name_r").str.contains(INDIC) & ~pl.col("business_name").str.contains(INDIC)).collect()
print("Indic-name true pairs:", j.height)
tok = lambda s: re.findall(r"[^\s.,()\-]+", s)
cnt, n_al, vocab = collections.defaultdict(collections.Counter), 0, collections.Counter()
for a, b in j.iter_rows():
    ta, tb = [w.lower() for w in tok(a)], [w for w in tok(b) if re.search(INDIC, w)]
    for w in tb: vocab[w] += 1
    if len(ta) == len(tok(b)):
        n_al += 1
        for x, y in zip(ta, tok(b)):
            if re.search(INDIC, y): cnt[y][x] += 1
print("same-length (alignable) pairs:", n_al, " distinct Indic word types:", len(vocab))
lex = {w: c.most_common(1)[0] for w, c in cnt.items()}
cons = [c[1] / sum(cnt[w].values()) for w, c in lex.items() if sum(cnt[w].values()) >= 5]
print(f"word types learned: {len(lex)}; with >=5 obs: {len(cons)}; mean consistency of top mapping: {sum(cons)/len(cons):.3f}")
tot = sum(vocab.values()); cov = sum(v for w, v in vocab.items() if w in lex)
print(f"token coverage of lexicon over all Indic name tokens: {cov/tot:.3f}")
print("examples (Indic -> learned English | rule translit):")
for w, _ in vocab.most_common(400)[::20]:
    if w in lex: print(f"  {w:22s} -> {lex[w][0]:16s} | {translit(w)}")
tv = collections.Counter()
for f in ("test_s2", "test_s3"):
    for s in pl.scan_parquet(f"research/eda/cache/{f}.parquet").select("business_name").filter(pl.col("business_name").str.contains(INDIC)).collect()["business_name"]:
        for w in tok(s):
            if re.search(INDIC, w): tv[w] += 1
tt = sum(tv.values()); oov = {w: c for w, c in tv.items() if w not in lex}
print(f"\nTEST Indic name tokens {tt}, types {len(tv)}; OOV types {len(oov)}; OOV token share {sum(oov.values())/tt:.4f}")
print("top OOV:", [(w, translit(w), c) for w, c in sorted(oov.items(), key=lambda x: -x[1])[:15]])
# address-side Indic words (states etc.)
av = collections.Counter()
for s in pl.scan_parquet("research/eda/cache/train_s2.parquet").select("business_address").filter(pl.col("business_address").str.contains(INDIC)).collect()["business_address"]:
    for w in re.findall(r"[ऀ-ൿ‌‍]+(?:\s[ऀ-ൿ‌‍]+)*", s): av[w] += 1
print(f"\naddress Indic phrase types (train S2): {len(av)}; top: {av.most_common(8)}")
