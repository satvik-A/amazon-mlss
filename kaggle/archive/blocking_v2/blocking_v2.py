"""Indic scripts -> Latin, dependency-free.
The 9 major Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share one layout: the same offset inside each 0x80 block is the same sound.
So one offset table transliterates all of them."""
BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
VOW = {0x05:"a",0x06:"aa",0x07:"i",0x08:"ii",0x09:"u",0x0A:"uu",0x0B:"ri",0x0C:"li",0x0D:"e",0x0E:"e",0x0F:"e",
       0x10:"ai",0x11:"o",0x12:"o",0x13:"o",0x14:"au",0x60:"rri",0x61:"lli"}
CON = dict(zip(range(0x15, 0x3A), "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n n p ph b bh m y r r l l l v sh sh s h".split()))
CON.update({0x58:"q",0x59:"kh",0x5A:"gh",0x5B:"z",0x5C:"r",0x5D:"rh",0x5E:"f",0x5F:"y"})
MAT = {0x3E:"aa",0x3F:"i",0x40:"ii",0x41:"u",0x42:"uu",0x43:"ri",0x44:"rri",0x45:"e",0x46:"e",0x47:"e",0x48:"ai",
       0x49:"o",0x4A:"o",0x4B:"o",0x4C:"au",0x62:"li",0x63:"lli"}
SIGN = {0x01:"n",0x02:"n",0x03:"h"}
VIRAMA, NUKTA = 0x4D, 0x3C

def _off(ch):
    o = ord(ch)
    if 0x0900 <= o < 0x0D80: return o & 0x7F
    return None

def translit(s: str) -> str:
    if not s: return s
    out, pend = [], False           # pend: a consonant is waiting for its inherent 'a'
    for ch in s:
        off = _off(ch)
        if off is None:          # leaving the word: drop the word-final inherent 'a'
            pend = False
            out.append(ch); continue
        if off in CON:
            if pend: out.append("a")
            out.append(CON[off]); pend = True
        elif off in MAT:
            out.append(MAT[off]); pend = False
        elif off == VIRAMA: pend = False
        elif off == NUKTA: continue
        elif off in VOW:
            if pend: out.append("a"); pend = False
            out.append(VOW[off])
        elif off in SIGN:
            if pend: out.append("a"); pend = False
            out.append(SIGN[off])
        elif 0x66 <= off <= 0x6F:
            if pend: out.append("a"); pend = False
            out.append(str(off - 0x66))
        else:
            if pend: out.append("a"); pend = False
    # word-final inherent 'a' is dropped (schwa deletion): राम -> ram
    return "".join(out)


# ============================ blocking v2 ============================
import glob, os, time, gc, sys
import polars as pl, numpy as np
LOCAL = not os.path.exists("/kaggle")
T0 = time.time(); WD = "." if LOCAL else "/kaggle/working"; OUT = open(f"{WD}/results_v2.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)[0])
rd = lambda n: pl.read_parquet(f"{IN}/{n}.parquet")

INDIC = r"[ऀ-ൿ]"
LEG = r"\b(inc|incorporated|llc|ltd|limited|pvt|private|public|corp|corporation|co|company|lp|llp|pllc|pc|the|and|of|l|c|p|a|dba|fka|aka|formerly|sarl|sas|sa|eurl|sasu|sci|ei|snc|cie|et|de|du|la|le|les|des)\b"
WEB = r"(^@|www\.|\.(com|in|net|org|co|fr|biz|info)\b|#\s*\d+)"

def prep(df):
    """Transliterate Indic text (only rows that contain it), then build normalized views."""
    df = df.with_columns(pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
    for c in ("business_name", "business_address"):
        m = df[c].str.contains(INDIC)
        if m.any():
            df = df.with_columns(pl.when(pl.col(c).str.contains(INDIC)).then(pl.col(c).map_elements(translit, return_dtype=pl.String)).otherwise(pl.col(c)).alias(c))
    clean = lambda c: pl.col(c).str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
    nm = clean("business_name").str.replace_all(WEB, " ").str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEG, " ").str.replace_all(r"\s+", " ").str.strip_chars()
    ad = clean("business_address").str.replace(r"^none$", "")
    return df.select("entity_id", pl.col("country").cast(pl.Categorical),
        nm.str.extract_all(r"[a-z0-9]{2,}").list.unique(maintain_order=True).alias("nw"),
        ad.str.extract_all(r"[a-z]{3,}").alias("aw"),
        ad.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != "")).alias("ad"))

def skel(e):
    """consonant skeleton of a lowercase ascii word expression (vectorized)"""
    e = e.str.replace_all("ksh", "x").str.replace_all("ph", "f").str.replace_all(r"([bcdgjkpstz])h", "$1") \
         .str.replace_all("ck", "k").str.replace_all("q", "k").str.replace_all("c", "k").str.replace_all("w", "v").str.replace_all("z", "j")
    e = e.str.replace_all(r"\B[aeiouy]", "")
    for ch in "bdfgjklmnprstvx": e = e.str.replace_all(ch + ch, ch)
    return e
_leg_sk = set(pl.DataFrame({"w": "private limited pvt ltd public incorporated inc company corporation corp llp llc sarl".split()})
              .select(skel(pl.col("w")))["w"].to_list()) | {"pr", "l", "li"}

def tokens(p, idcol):
    """one row per (id, token-hash, arm). arms: 0 name, 1 addr, 2 skeleton, 3 name-pair, 4 exact keys"""
    d = p.with_columns(pl.col("nw").list.eval(skel(pl.element())).list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().is_in(list(_leg_sk)))).alias("sk"))
    h4 = pl.col("nw").list.head(4)
    prs = [pl.when(h4.list.len() > j).then(pl.min_horizontal(h4.list.get(i, null_on_oob=True), h4.list.get(j, null_on_oob=True)) + "|" +
                                          pl.max_horizontal(h4.list.get(i, null_on_oob=True), h4.list.get(j, null_on_oob=True))) for i in range(4) for j in range(i+1, 4)]
    arms = [
        (0, pl.col("nw").list.eval(pl.lit("n:") + pl.element())),
        (1, pl.concat_list(pl.col("aw").list.eval(pl.lit("a:") + pl.element()), pl.col("ad").list.eval(pl.lit("#:") + pl.element()))),
        (2, pl.col("sk").list.eval(pl.lit("k:") + pl.element())),
        (3, pl.concat_list(*prs).list.drop_nulls().list.eval(pl.lit("p:") + pl.element())),
        (4, pl.concat_list(pl.lit("N:") + pl.col("nw").list.sort().list.join(" "),
                           pl.lit("C:") + pl.col("nw").list.join(""),
                           pl.lit("C:") + pl.col("nw").list.head(2).list.join(""),
                           pl.lit("K:") + pl.col("sk").list.sort().list.join(" "))),
    ]
    out = []
    for a, e in arms:
        t = d.select(pl.col(idcol).alias("id"), "country", e.list.unique().alias("t")).explode("t").drop_nulls("t") \
             .filter(pl.col("t").str.len_chars() > 3)
        out.append(t.select("id", "country", pl.col("t").hash().alias("h"), pl.lit(a, dtype=pl.UInt8).alias("arm")))
    return pl.concat(out)

def sigs(p, idcol):
    """sibling signatures: exact set of address numbers; exact name skeleton key"""
    s = p.with_columns(pl.col("nw").list.eval(skel(pl.element())).list.eval(pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().is_in(list(_leg_sk)))).alias("sk"))
    num = s.select(pl.col(idcol).alias("id"), "country", (pl.lit("#") + pl.col("ad").list.unique().list.sort().list.join("-")).alias("sig"),
                   pl.col("ad").list.join("").str.len_chars().alias("nd")).filter(pl.col("nd") >= 3).drop("nd")
    nam = s.select(pl.col(idcol).alias("id"), "country", (pl.lit("K") + pl.col("sk").list.sort().list.join(" ")).alias("sig"),
                   pl.col("sk").list.len().alias("n")).filter(pl.col("n") >= 2).drop("n")
    return pl.concat([num, nam]).with_columns(pl.col("sig").hash().alias("sig"))

CAP = {0: 50, 1: 50, 2: 50, 3: 30, 4: 30}
def retrieve(Sp, Rt, dfc, chunk=1500):
    St = tokens(Sp, "sid").join(dfc, on=["country", "h"])
    res = []
    for c0 in range(0, Sp.height, chunk):
        ch = St.filter((pl.col("id") >= c0) & (pl.col("id") < c0 + chunk))
        j = ch.join(Rt, on=["country", "h", "arm"], suffix="_r")
        a = j.group_by(["id", "arm", "id_r"]).agg(pl.col("idf").sum().alias("sc"))
        a = a.with_columns(pl.col("sc").rank("ordinal", descending=True).over(["id", "arm"]).alias("rk"))
        a = a.filter(pl.col("rk") <= pl.col("arm").replace_strict(CAP, return_dtype=pl.UInt32))
        f = a.group_by(["id", "id_r"]).agg((1.0 / (20 + pl.col("rk"))).sum().alias("rrf"),
                                           *[(pl.col("arm") == k).any().alias(f"a{k}") for k in range(5)])
        res.append(f.with_columns(pl.col("rrf").rank("ordinal", descending=True).over("id").alias("frk")))
    return pl.concat(res)

def expand(cand, Rsig, M=10, maxgrp=12):
    """add every R record sharing a sibling signature with one of the top-M candidates"""
    grp = Rsig.group_by(["country", "sig"]).agg(pl.col("id").alias("mem")).filter(pl.col("mem").list.len() <= maxgrp)
    top = cand.filter(pl.col("frk") <= M).select("id", "id_r")
    s = top.join(Rsig.rename({"id": "id_r"}), on="id_r").join(grp, on=["country", "sig"]).select("id", pl.col("mem")).explode("mem").rename({"mem": "id_r"}).unique()
    return s.join(cand.select("id", "id_r"), on=["id", "id_r"], how="anti")

def build_pool(R):
    t = time.time(); P = prep(R.with_row_index("rid").rename({"rid": "rid"}).select("rid", "entity_id", "business_name", "business_address", "country"))
    P = P.with_columns(pl.Series("rid", np.arange(R.height, dtype=np.uint32)))
    log(f"  prep pool {R.height}: {time.time()-t:.0f}s")
    Rt = tokens(P, "rid"); log(f"  pool tokens {Rt.height}")
    dfc = Rt.group_by(["country", "h", "arm"]).len().rename({"len": "df"})
    N = P.group_by("country").len().rename({"len": "N"})
    dfc = dfc.join(N, on="country").with_columns((pl.col("N") / pl.col("df")).log().alias("idf")) \
             .filter(pl.when(pl.col("arm") == 4).then(pl.col("df") <= 200).otherwise(pl.col("df") <= 5000)).select("country", "h", "arm", "idf")
    Rt = Rt.join(dfc.select("country", "h", "arm"), on=["country", "h", "arm"]).rename({"id": "id_r"})
    log(f"  pool postings after df cap {Rt.height}  {time.time()-T0:.0f}s")
    return P, Rt, dfc, sigs(P, "rid")

# ------------------------------ TRAIN: recall ------------------------------
s1 = rd("train_s1"); R = pl.concat([rd("train_s2"), rd("train_s3")])
if LOCAL: s1 = s1.head(100000); R = R.head(400000)
g = rd("gt_rows").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids") \
    .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "r"})
log(f"TRAIN S1={s1.height} R={R.height}")
P, Rt, dfc, Rsig = build_pool(R)
samp = s1.sample(2000 if LOCAL else 30000, seed=11)
Sp = prep(samp).with_columns(pl.Series("sid", np.arange(samp.height, dtype=np.uint32)))
cand = retrieve(Sp, Rt, dfc); log(f"retrieved {cand.height}  {time.time()-T0:.0f}s")
ex = expand(cand, Rsig); log(f"expansion adds {ex.height}")
idmap_s = Sp.select(pl.col("sid").alias("id"), pl.col("entity_id").alias("s1"))
idmap_r = P.select(pl.col("rid").alias("id_r"), pl.col("entity_id").alias("r"))
Rraw = R.select(pl.col("entity_id").alias("r"), pl.col("business_name").fill_null("").str.contains(INDIC).alias("indic"),
                (pl.col("business_address").is_null() | (pl.col("business_address") == "None")).alias("noaddr"),
                pl.col("business_name").fill_null("").str.contains(r"(?i)\.(com|in|net|org)\b|^@|www\.").alias("web"),
                pl.col("business_name").fill_null("").str.contains(r"(?i)f/k/a|d/b/a|t/a|formerly").alias("alias"))
C = cand.join(idmap_s, on="id").join(idmap_r, on="id_r")
E = ex.join(idmap_s, on="id").join(idmap_r, on="id_r").select("s1", "r").with_columns(pl.lit(True).alias("exp"))
Pos = g.filter(pl.col("s1").is_in(samp["entity_id"].implode())).join(Rraw, on="r") \
       .join(samp.select(pl.col("entity_id").alias("s1"), "country"), on="s1") \
       .join(C.select("s1", "r", "frk", *[f"a{k}" for k in range(5)]), on=["s1", "r"], how="left") \
       .join(E, on=["s1", "r"], how="left").with_columns(pl.col("exp").fill_null(False), pl.col("frk").fill_null(10**9))
def rec(h, K, withexp=False):
    if h.height == 0: return float("nan")
    m = (h["frk"] <= K) | (h["exp"] if withexp else False)
    return float(m.mean())
log("\n=== v2 recall (pairs) by fused rank K, and with sibling expansion (+exp from top-10) ===")
exp_per = E.group_by("s1").len()["len"]; log(f"expansion: mean added per S1 {float(exp_per.sum())/samp.height:.2f}")
for K in (5, 10, 20, 30, 50, 100):
    row = [f"K={K:3d} all={rec(Pos, K):.4f} +exp={rec(Pos, K, True):.4f}"]
    for sl in ["indic", "noaddr", "web", "alias"]:
        h = Pos.filter(pl.col(sl)); row.append(f"{sl}={rec(h, K):.3f}/{rec(h, K, True):.3f}")
    for c in ["US", "India"]:
        h = Pos.filter(pl.col("country") == c); row.append(f"{c}={rec(h, K):.4f}/{rec(h, K, True):.4f}")
    log("  ".join(row))
log("\n=== entity completeness (ALL matches of the S1 retrieved) ===")
for K in (20, 30, 50):
    e = Pos.group_by("s1").agg(((pl.col("frk") <= K) | pl.col("exp")).all().alias("ok"))
    log(f"  top{K}+exp: {e['ok'].mean():.4f}    top{K} only: {Pos.group_by('s1').agg((pl.col('frk') <= K).all())['frk'].mean():.4f}")
log("\n=== per-arm: share of found positives hit by arm, and unique finds ===")
F = Pos.filter(pl.col("frk") < 10**9)
for k, nme in enumerate(["name", "addr", "skeleton", "namepair", "keys"]):
    oth = [f"a{j}" for j in range(5) if j != k]
    log(f"  {nme:9s} hit {F[f'a{k}'].mean():.3f}  unique {F.filter(pl.col(f'a{k}') & ~pl.any_horizontal(*oth)).height}")
log(f"  expansion-only finds: {Pos.filter((pl.col('frk') == 10**9) & pl.col('exp')).height}")
miss = Pos.filter((pl.col("frk") > 30) & ~pl.col("exp")).join(R.select(pl.col("entity_id").alias("r"), "business_name", "business_address"), on="r") \
          .join(s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("s1_addr")), on="s1")
log("\n=== sample of remaining misses (not in top30, not expanded) ===")
for r in miss.sample(min(25, miss.height), seed=1).iter_rows(named=True):
    log(f"  rk={r['frk'] if r['frk'] < 10**9 else '-'} | {r['n1']} | {str(r['s1_addr'])[:55]}  ==>  {r['business_name']} | {str(r['business_address'])[:55]}")
# candidate volume stats on train for later comparison with France
vol = C.filter(pl.col("frk") <= 30).group_by("s1").agg(pl.len().alias("n"), (pl.sum_horizontal(*[pl.col(f"a{k}").cast(pl.Int8) for k in range(5)]) >= 3).sum().alias("strong"))
log(f"\ntrain: mean candidates(top30) {vol['n'].mean():.1f}; mean 'strong' (>=3 arms) {vol['strong'].mean():.2f}; exp/S1 {float(exp_per.sum())/samp.height:.2f}")
del P, Rt, dfc, Rsig, cand, ex, C, E, Pos, R; gc.collect()

# ------------------------------ TEST France sanity ------------------------------
t1 = rd("test_s1"); tR = pl.concat([rd("test_s2"), rd("test_s3")])
for c in (["France"] if not LOCAL else []):
    Rc = tR.filter(pl.col("country") == c); Sc = t1.filter(pl.col("country") == c).sample(10000, seed=3)
    log(f"\nTEST {c}: S1 sample {Sc.height}, pool {Rc.height}")
    P, Rt, dfc, Rsig = build_pool(Rc)
    Sp = prep(Sc).with_columns(pl.Series("sid", np.arange(Sc.height, dtype=np.uint32)))
    cand = retrieve(Sp, Rt, dfc); ex = expand(cand, Rsig)
    vol = cand.filter(pl.col("frk") <= 30).group_by("id").agg(pl.len().alias("n"), (pl.sum_horizontal(*[pl.col(f"a{k}").cast(pl.Int8) for k in range(5)]) >= 3).sum().alias("strong"))
    log(f"{c}: mean candidates(top30) {vol['n'].mean():.1f}; mean 'strong' (>=3 arms) {vol['strong'].mean():.2f}; exp/S1 {ex.height/Sc.height:.2f}")
    top = cand.filter(pl.col("frk") <= 5).join(Sp.select(pl.col("sid").alias("id"), pl.col("entity_id").alias("s1")), on="id") \
              .join(P.select(pl.col("rid").alias("id_r"), pl.col("entity_id").alias("r")), on="id_r") \
              .join(Sc.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("s1_addr")), on="s1") \
              .join(Rc.select(pl.col("entity_id").alias("r"), "business_name", "business_address"), on="r")
    log(f"{c}: sample top-5 candidates for 4 S1s:")
    for sid in top["s1"].unique().sample(4, seed=0).to_list():
        tt = top.filter(pl.col("s1") == sid).sort("frk")
        log(f"  ## {tt['n1'][0]} | {tt['s1_addr'][0]}")
        for r in tt.iter_rows(named=True): log(f"      {r['frk']} {r['business_name']} | {r['business_address']}")
log(f"\nDONE {time.time()-T0:.0f}s")
