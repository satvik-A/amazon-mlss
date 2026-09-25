"""Score level-1 pairs with the fine-tuned cross-encoders from er-xenc-v1 (one model per T4, in parallel).
Input: level1_<name>.parquet files (s1, r, p, ...) from er-matcher-full (B, C) or the level-1 test job.
Only the uncertain band BAND_LO < p < BAND_HI is scored (the rest is already decided by the GBDT; same rule at test time).
Output: xenc_<name>.parquet [s1, r, lg_qwen, lg_bge]."""
import glob, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
WD = "." if LOCAL else "/kaggle/working"
if not LOCAL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51", "peft>=0.13"], check=True)
import numpy as np, polars as pl
T0 = time.time(); BAND_LO, BAND_HI = float(os.environ.get("BAND_LO", 0.005)), float(os.environ.get("BAND_HI", 0.995))
REP = open(f"{WD}/results_xenc_score.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
MODELS = {k: (os.path.dirname(find(f"model_{k}/config.json")[0]) if find(f"model_{k}/config.json") else
              os.path.dirname(find(f"model_{k}/adapter_config.json")[0])) for k in ("qwen", "bge")} if not LOCAL else {}
FILES = find("level1_*.parquet") if not LOCAL else sorted(glob.glob("../matcher_full/level1_*.parquet"))
log(f"models {MODELS}; files {FILES}; band ({BAND_LO}, {BAND_HI})")

WORKER = r'''
import sys, numpy as np, polars as pl, torch
kind, mdir, inp, outp = sys.argv[1:5]
from transformers import AutoTokenizer
dev = "cuda"
d = pl.read_parquet(inp)
if kind == "qwen":
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(mdir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
    model = PeftModel.from_pretrained(model, mdir).merge_and_unload().to(dev).half().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    INST = "Are the Query business record and the Document business record the same business entity (same business at the same location)?"
    def run(a, b):
        body = tok([f"<Instruct>: {INST}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
        ids = [pre + t[:200] + suf for t in body]; m = max(map(len, ids)); pad = tok.pad_token_id
        lg = model(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                   attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev)).logits[:, -1, :]
        return (lg[:, yes] - lg[:, no]).float()
else:
    from transformers import AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).half().eval()
    def run(a, b):
        return model(**{k: v.to(dev) for k, v in tok(list(a), list(b), padding=True, truncation=True, max_length=192, return_tensors="pt").items()}).logits.squeeze(-1).float()
# length-sorted batches: much less padding
order = np.argsort((d["t_s1"].str.len_chars() + d["t_r"].str.len_chars()).to_numpy())
a, b = d["t_s1"].to_numpy()[order], d["t_r"].to_numpy()[order]
out = np.zeros(len(a), dtype=np.float32); bs = 256
with torch.no_grad():
    for i in range(0, len(a), bs):
        out[order[i:i + bs]] = run(a[i:i + bs].tolist(), b[i:i + bs].tolist()).cpu().numpy()
d.select("s1", "r").with_columns(pl.Series(f"lg_{kind}", out)).write_parquet(outp)
'''

if __name__ == "__main__":
    open(f"{WD}/worker.py", "w").write(WORKER)
    txt = lambda df: df.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("t"))
    for f in FILES:
        name = os.path.basename(f)[len("level1_"):-len(".parquet")]
        d = pl.read_parquet(f, columns=["s1", "r", "p"])
        u = d.filter((pl.col("p") > BAND_LO) & (pl.col("p") < BAND_HI))
        split = "test" if name.startswith("test") else "train"
        S = txt(pl.scan_parquet(f"{IN}/{split}_s1.parquet").filter(pl.col("entity_id").is_in(u["s1"].unique().implode())).collect())
        R = txt(pl.scan_parquet([f"{IN}/{split}_s2.parquet", f"{IN}/{split}_s3.parquet"]).filter(pl.col("entity_id").is_in(u["r"].unique().implode())).collect())
        u = u.join(S.rename({"entity_id": "s1", "t": "t_s1"}), on="s1").join(R.rename({"entity_id": "r", "t": "t_r"}), on="r")
        u.write_parquet(f"{WD}/pairs_{name}.parquet")
        log(f"{name}: {d.height} rows, uncertain band {u.height} ({u.height/max(d.height,1):.3f})  {time.time()-T0:.0f}s")
        t = time.time()
        ps = [subprocess.Popen([sys.executable, f"{WD}/worker.py", k, MODELS[k], f"{WD}/pairs_{name}.parquet", f"{WD}/_{k}_{name}.parquet"],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, k in enumerate(("qwen", "bge"))]
        codes = [p.wait() for p in ps]
        out = u.select("s1", "r")
        for k in ("qwen", "bge"):
            if os.path.exists(f"{WD}/_{k}_{name}.parquet"):
                out = out.join(pl.read_parquet(f"{WD}/_{k}_{name}.parquet"), on=["s1", "r"], how="left")
        out.write_parquet(f"{WD}/xenc_{name}.parquet")
        log(f"  scored in {time.time()-t:.0f}s ({u.height/max(time.time()-t,1):.0f} pairs/s per model pair), exit codes {codes}")
        os.remove(f"{WD}/pairs_{name}.parquet")
    log(f"DONE {time.time()-T0:.0f}s")
