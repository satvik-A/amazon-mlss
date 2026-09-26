"""[account 2 · GPU T4 · internet] Embedding search arm, measured against the word blocking (recall lab r10 config).
Per train country: embed ALL S2/S3 records + a 30k S1 sample with intfloat/multilingual-e5-small (MIT, 118M), exact
cosine top-K on GPU. Reports pair recall / S1 completeness of: word arms, embedding arm (K = 5..100), union, and what the
embedding arm adds on the slices the word search misses (no-address copies, Indian-script copies)."""
import glob, json, os, subprocess, sys, time
REF = "5f74c86"
find = lambda pat: sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))
W = os.path.dirname(find("polars-1.44.2*.whl")[0])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W, "polars==1.44.2", "rapidfuzz==3.14.6"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                f"git+https://github.com/satvik-A/amazon-mlss.git@{REF}#subdirectory=code/business_entity_resolution"], check=True)
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
from ber.normalize import Normalizer
from ber.pipeline import norm_chunks
from ber.translit import INDIC_RE
from ber import blocking as B
T0 = time.time(); WD = "/kaggle/working"; N_Q = 30000; KS = (5, 10, 20, 50, 100)
REP = open(f"{WD}/results_embed.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
IN = os.path.dirname(find("train_s1.parquet")[0]); ART = os.path.dirname(find("indic_lexicon.parquet")[0])
MN = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(MN); enc = AutoModel.from_pretrained(MN, torch_dtype=torch.float16).cuda().eval()
log(f"model {MN} on {torch.cuda.get_device_name(0)}  {time.time()-T0:.0f}s")
def text(df):
    n = df["business_name"].fill_null("").to_list(); a = df["business_address"].fill_null("").to_list()
    return [f"query: {x} | {y}" if y and y.strip() not in ("None", "null") else f"query: {x}" for x, y in zip(n, a)]
@torch.no_grad()
def embed(txt, bs=1024):
    order = np.argsort([len(t) for t in txt]); out = torch.empty((len(txt), 384), dtype=torch.float16, device="cuda")
    for i in range(0, len(txt), bs):
        idx = order[i:i + bs]
        b = tok([txt[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1); out[torch.as_tensor(idx, device="cuda")] = torch.nn.functional.normalize(e.float(), dim=-1).half()
    return out
NZ = Normalizer(ART)
B.ADDR_TAIL = 3
gt = pl.read_parquet(f"{IN}/gt_rows.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")) \
       .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("r"))
caps = {**B.CAP, 3: 30, 4: 30, 5: 40, 6: 25, 7: 20}
rows = []
for c in ("India", "US"):
    S = pl.scan_parquet(f"{IN}/train_s1.parquet").filter(pl.col("country") == c).collect().sort("entity_id")
    R = pl.scan_parquet([f"{IN}/train_s2.parquet", f"{IN}/train_s3.parquet"]).filter(pl.col("country") == c).collect().sort("entity_id")
    Q = S.filter(pl.col("entity_id").hash(21) % 1000 < max(1, int(1000 * N_Q / S.height)))
    # --- word blocking (recall lab r10) ---
    RN = norm_chunks(NZ, R).with_columns(pl.Series("id", np.arange(R.height, dtype=np.uint32)))
    idx = B.Index(RN, cap=2000, key_cap=200)
    QN = NZ.transform(Q).with_columns(pl.Series("id", np.arange(Q.height, dtype=np.uint32)))
    cand = idx.query(QN, caps=caps); sig = B.signatures(RN); ex = B.expand(cand, sig)
    cand = pl.concat([cand.select("id", "id_r", "prk", "xrk"), ex.select("id", "id_r").with_columns(pl.lit(None, dtype=pl.UInt32).alias("prk"), pl.lit(None, dtype=pl.UInt32).alias("xrk"))], how="vertical_relaxed")
    cand = cand.filter((pl.col("prk") <= 60) | pl.col("xrk").is_not_null() | (pl.col("prk").is_null() & pl.col("xrk").is_null())).select("id", "id_r").unique()
    del idx, sig, ex, RN; log(f"{c}: word blocking {cand.height/Q.height:.1f}/S1  {time.time()-T0:.0f}s")
    # --- embedding arm ---
    ER = embed(text(R)); log(f"{c}: embedded {R.height} records  {time.time()-T0:.0f}s")
    EQ = embed(text(Q))
    top = []
    for i in range(0, Q.height, 256):
        sc = EQ[i:i + 256] @ ER.T
        top.append(torch.topk(sc, max(KS), dim=1).indices.cpu().numpy())
    top = np.vstack(top); del ER, EQ; torch.cuda.empty_cache()
    log(f"{c}: kNN done  {time.time()-T0:.0f}s")
    g = gt.filter(pl.col("s1").is_in(Q["entity_id"].implode()))
    qid = Q.select(pl.col("entity_id").alias("s1")).with_row_index("id").with_columns(pl.col("id").cast(pl.UInt32))
    rid = R.select(pl.col("entity_id").alias("r")).with_row_index("id_r").with_columns(pl.col("id_r").cast(pl.UInt32))
    g = g.join(qid, on="s1").join(rid, on="r")
    g = g.join(R.select(pl.col("entity_id").alias("r"), pl.col("business_name").fill_null("").str.contains(INDIC_RE).alias("indic"),
                        pl.col("business_address").fill_null("").str.strip_chars().is_in(["", "None"]).alias("noaddr")), on="r")
    g = g.join(cand.with_columns(pl.lit(True).alias("w")), on=["id", "id_r"], how="left").with_columns(pl.col("w").fill_null(False))
    for K in KS:
        E = pl.DataFrame({"id": np.repeat(np.arange(Q.height, dtype=np.uint32), K), "id_r": top[:, :K].ravel().astype(np.uint32)})
        h = g.join(E.with_columns(pl.lit(True).alias("e")), on=["id", "id_r"], how="left").with_columns(pl.col("e").fill_null(False))
        new = E.join(cand, on=["id", "id_r"], how="anti").height / Q.height
        u = h["w"] | h["e"]
        row = dict(country=c, K=K, word_recall=h["w"].mean(), embed_recall=h["e"].mean(), union_recall=u.mean(),
                   word_complete=h.group_by("s1").agg(pl.col("w").all())["w"].mean(),
                   union_complete=h.with_columns(u.alias("u")).group_by("s1").agg(pl.col("u").all())["u"].mean(),
                   added_cands_per_S1=new, word_cands_per_S1=cand.height / Q.height,
                   noaddr_word=h.filter(pl.col("noaddr"))["w"].mean(), noaddr_union=h.filter(pl.col("noaddr")).select(pl.col("w") | pl.col("e")).to_series().mean(),
                   indic_word=h.filter(pl.col("indic"))["w"].mean(), indic_union=h.filter(pl.col("indic")).select(pl.col("w") | pl.col("e")).to_series().mean(),
                   rescued_share_of_word_misses=h.filter(~pl.col("w"))["e"].mean())
        rows.append(row); log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
pl.DataFrame(rows).write_csv(f"{WD}/embed_lab.csv"); log(f"DONE {time.time()-T0:.0f}s")
