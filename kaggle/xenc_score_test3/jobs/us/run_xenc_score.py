"""Score level-1 pairs with every fine-tuned cross-encoder found in the inputs (er-xenc-v1/v1b/v2/v3; two at a time, one per T4).
Input: level1_<name>.parquet files (s1, r, p, ...) from er-matcher-full (B, C) or the level-1 test job.
Only the uncertain band BAND_LO < p < BAND_HI is scored (the rest is already decided by the GBDT; same rule at test time).
Output: xenc_<name>.parquet [s1, r, lg_<model tag> for every fine-tuned model found in the inputs]."""
import glob, os, subprocess, sys, time
REF = "__REF__"
LOCAL = not os.path.exists("/kaggle")
WD = "." if LOCAL else "/kaggle/working"
if not LOCAL:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # peft vs old torchao
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "peft>=0.13"], check=True)   # newest: Qwen3.5 support
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention", "causal-conv1d"], check=False)
import numpy as np, polars as pl
T0 = time.time(); BAND_LO, BAND_HI = float(os.environ.get("BAND_LO", 0.005)), float(os.environ.get("BAND_HI", 0.995))
REP = open(f"{WD}/results_xenc_score.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REP.write(s + "\n"); REP.flush()
find = lambda f: sorted(glob.glob(f"/kaggle/input/**/{f}", recursive=True))
IN = "../../research/eda/cache" if LOCAL else os.path.dirname(find("train_s1.parquet")[0])
def _models():
    """every model_<tag>/ directory in the inputs (er-xenc-v1, v1b, v2, v3 ...): {tag: dir}"""
    out = {}
    for f in find("model_*/config.json") + find("model_*/adapter_config.json"):
        d = os.path.dirname(f)
        run = os.path.basename(os.path.dirname(d)).replace("er-xenc-", "")     # kernel of origin (v1, v4, ...) keeps tags unique
        out[f"{run}_{os.path.basename(d)[len('model_'):]}".replace("-", "_").replace(".", "_")] = d
    return out
MODELS = _models() if not LOCAL else {}
FILES = find("level1_*.parquet") if not LOCAL else sorted(glob.glob("../matcher_full/level1_*.parquet"))
log(f"models {MODELS}; files {FILES}; band ({BAND_LO}, {BAND_HI})")

WORKER = r'''
import json, os, sys, numpy as np, polars as pl, torch
tag, mdir, inp, outp = sys.argv[1:5]
from transformers import AutoTokenizer
dev = "cuda"
d = pl.read_parquet(inp)
ad = os.path.join(mdir, "adapter_config.json")
cfg = json.load(open(os.path.join(mdir, "config.json"))) if os.path.exists(os.path.join(mdir, "config.json")) else {}
yesno = os.path.exists(ad) or any("CausalLM" in a or "ConditionalGeneration" in a for a in cfg.get("architectures", []))
if yesno:   # Qwen3 reranker / Qwen3.5 with the yes/no prompt (LoRA adapter merged, or a full fine-tune)
    base = json.load(open(ad))["base_model_name_or_path"] if os.path.exists(ad) else mdir
    if "Qwen3.5" in base or "qwen3_5" in json.dumps(cfg):
        from transformers import AutoModelForImageTextToText as AM
    else:
        from transformers import AutoModelForCausalLM as AM
    tok = AutoTokenizer.from_pretrained(mdir, padding_side="left")
    model = AM.from_pretrained(base, torch_dtype=torch.float16)
    if os.path.exists(ad):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, mdir).merge_and_unload()
    model = model.to(dev).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    INST = "Are the Query business record and the Document business record the same business entity (same business at the same location)?"
    def run(a, b):
        body = tok([f"<Instruct>: {INST}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
        ids = [pre + t[:200] + suf for t in body]; m = max(map(len, ids)); pad = tok.pad_token_id
        kw = dict(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                  attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev))
        try:
            lg = model(**kw, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            lg = model(**kw).logits[:, -1, :]
        return (lg[:, yes] - lg[:, no]).float()
    bs = 16 if ("4B" in base or "2B" in base) else 64
else:       # encoder / encoder-decoder with a sequence-classification head (bge, ByT5, CANINE, ...)
    from transformers import AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(mdir, trust_remote_code=True, torch_dtype=torch.float16).to(dev).eval()
    def run(a, b):
        return model(**{k: v.to(dev) for k, v in tok(list(a), list(b), padding=True, truncation=True, max_length=192, return_tensors="pt").items()}).logits.squeeze(-1).float()
    bs = 256
order = np.argsort((d["t_s1"].str.len_chars() + d["t_r"].str.len_chars()).to_numpy())   # length-sorted: less padding
a, b = d["t_s1"].to_numpy()[order], d["t_r"].to_numpy()[order]
out = np.zeros(len(a), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(a), bs):
        out[order[i:i + bs]] = run(a[i:i + bs].tolist(), b[i:i + bs].tolist()).cpu().numpy()
d.select("s1", "r").with_columns(pl.Series(f"lg_{tag}", out)).write_parquet(outp)
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
        t = time.time(); tags = sorted(MODELS); codes = {}
        for j in range(0, len(tags), 2):          # two models at a time, one per T4
            ps = {tg: subprocess.Popen([sys.executable, f"{WD}/worker.py", tg, MODELS[tg], f"{WD}/pairs_{name}.parquet", f"{WD}/_{tg}_{name}.parquet"],
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}) for i, tg in enumerate(tags[j:j + 2])}
            codes.update({tg: p.wait() for tg, p in ps.items()})
        out = u.select("s1", "r")
        for tg in tags:
            if os.path.exists(f"{WD}/_{tg}_{name}.parquet"):
                out = out.join(pl.read_parquet(f"{WD}/_{tg}_{name}.parquet"), on=["s1", "r"], how="left")
        out.write_parquet(f"{WD}/xenc_{name}.parquet")
        log(f"  scored {u.height} pairs with {len(tags)} models in {time.time()-t:.0f}s; exit codes {codes}")
        os.remove(f"{WD}/pairs_{name}.parquet")
    log(f"DONE {time.time()-T0:.0f}s")
