"""[pass 4 · GPU T4] Embedding search arm for ONE (split, country): embed every S2/S3 record and every S1 with
intfloat/multilingual-e5-small (MIT, 118M; offline after download, not fine-tuned), exact cosine top-K per S1 on GPU.
Writes emb_<split>_<Country>.parquet [s1, r, erk, esim] for the candidate job (a blocking arm; no pair scorer).
Embed lab (30k train S1): top-10 adds 4.5 cands/S1, India recall 0.9796 -> 0.9880, US 0.9932 -> 0.9945."""
import glob, os, subprocess, sys, time
SPLIT, CTRY, K = "test", "India", 20
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
T0 = time.time(); WD = "/kaggle/working"
REP = open(f"{WD}/results_emb_{SPLIT}_{CTRY}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
IN = os.path.dirname(find(f"{SPLIT}_s1.parquet")[0])
MN = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(MN); enc = AutoModel.from_pretrained(MN, torch_dtype=torch.float16).cuda().eval()
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
S = pl.scan_parquet(f"{IN}/{SPLIT}_s1.parquet").filter(pl.col("country") == CTRY).collect().sort("entity_id")
R = pl.scan_parquet([f"{IN}/{SPLIT}_s2.parquet", f"{IN}/{SPLIT}_s3.parquet"]).filter(pl.col("country") == CTRY).collect().sort("entity_id")
log(f"{SPLIT} {CTRY}: S1 {S.height} R {R.height}")
ER = embed(text(R)); log(f"embedded records  {time.time()-T0:.0f}s")
EQ = embed(text(S)); log(f"embedded S1  {time.time()-T0:.0f}s")
I, V = [], []
for i in range(0, S.height, 256):
    v, ix = torch.topk(EQ[i:i + 256] @ ER.T, K, dim=1)
    I.append(ix.cpu().numpy().astype(np.uint32)); V.append(v.float().cpu().numpy())
I, V = np.vstack(I), np.vstack(V)
log(f"kNN top-{K}  {time.time()-T0:.0f}s")
out = pl.DataFrame({"s1": np.repeat(S["entity_id"].to_numpy(), K), "r": R["entity_id"].to_numpy()[I.ravel()],
                    "erk": np.tile(np.arange(1, K + 1, dtype=np.uint16), S.height), "esim": V.ravel().astype(np.float32)})
out.write_parquet(f"{WD}/emb_{SPLIT}_{CTRY}.parquet")
log(f"wrote emb_{SPLIT}_{CTRY}.parquet {out.height} rows; mean esim@1 {V[:, 0].mean():.3f} @{K} {V[:, -1].mean():.3f}  DONE {time.time()-T0:.0f}s")
