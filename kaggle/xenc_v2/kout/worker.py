
import json, math, os, sys, time
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score, log_loss
kind, WD, TRAIN_MIN = sys.argv[1], sys.argv[2], float(sys.argv[3])
tag = kind.replace("hf:", "").replace("/", "_")
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pl.read_parquet(f"{WD}/pairs_train.parquet"); ev = pl.read_parquet(f"{WD}/pairs_eval.parquet")
L = open(f"{WD}/log_{tag}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{kind}] " + s, flush=True); L.write(s + "\n"); L.flush()
from transformers import AutoTokenizer
EVAL_MAX = 10**9
if kind.startswith("qwen"):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    name = "Qwen/Qwen3-Reranker-4B" if kind == "qwen4b" else "Qwen/Qwen3-Reranker-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if kind == "qwen4b":   # 8 GB of fp16 weights on a 16 GB T4: frozen fp16 base + fp32 LoRA, gradient checkpointing
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads(); EVAL_MAX = 20000
    else:
        model = AutoModelForCausalLM.from_pretrained(name).to(dev)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    INST = "Are the Query business record and the Document business record the same business entity (same business at the same location)?"
    def enc(a, b):
        body = tok([f"<Instruct>: {INST}\n<Query>: {x}\n<Document>: {y}" for x, y in zip(a, b)], add_special_tokens=False)["input_ids"]
        ids = [pre + t[:200] + suf for t in body]; m = max(map(len, ids)); pad = tok.pad_token_id
        return dict(input_ids=torch.tensor([[pad] * (m - len(i)) + i for i in ids], device=dev),
                    attention_mask=torch.tensor([[0] * (m - len(i)) + [1] * len(i) for i in ids], device=dev))
    def fwd(b):
        # only the last position's logits (full-sequence x 152k-vocab logits is ~7.6 GB per scoring batch)
        lg = model(**b, logits_to_keep=1).logits[:, -1, :]
        return (lg[:, yes] - lg[:, no]).float()
    def make_trainable():
        global model
        if kind == "qwenfull":        # full fine-tune of the 0.6B (compare with LoRA)
            return 1e-5, 16
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        return (1e-4, 8) if kind == "qwen4b" else (2e-4, 24)
else:
    # "bge" = BAAI/bge-reranker-v2-m3; "hf:<id>" = any HF encoder / encoder-decoder with a sequence-classification head
    from transformers import AutoModelForSequenceClassification
    name = kind[3:] if kind.startswith("hf:") else "BAAI/bge-reranker-v2-m3"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, trust_remote_code=True, ignore_mismatched_sizes=True).to(dev)
    def enc(a, b):
        return {k: v.to(dev) for k, v in tok(list(a), list(b), padding=True, truncation=True, max_length=192, return_tensors="pt").items()}
    def fwd(b):
        return model(**b).logits.squeeze(-1).float()
    def make_trainable():
        return (1e-4, 24) if "t5" in name.lower() else (2e-5, 24)

@torch.no_grad()
def score(d, bs=128):
    model.eval(); out = []
    a, b = d["t_s1"].to_list(), d["t_r"].to_list()
    for i in range(0, len(a), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(fwd(enc(a[i:i + bs], b[i:i + bs])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def report(tag, d, lg):
    p = 1 / (1 + np.exp(-np.clip(lg, -30, 30))); y = d["y"].to_numpy()
    x = d.with_columns(pl.Series("lg", lg), pl.Series("p", p))
    res = {"auc": float(roc_auc_score(y, lg)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}
    top = x.sort("lg", descending=True).group_by("s1", maintain_order=True).first()
    has = x.group_by("s1").agg(pl.col("y").max().alias("any"))
    t = top.join(has, on="s1")
    res["top1_acc_nonsingleton"] = float(t.filter(pl.col("any") == 1)["y"].mean())
    res["singleton_max_p_below_0.5"] = float((t.filter(pl.col("any") == 0)["p"] < 0.5).mean()) if t.filter(pl.col("any") == 0).height else None
    for c in x["country"].unique().to_list():
        xc = x.filter(pl.col("country") == c)
        res[f"auc_{c}"] = float(roc_auc_score(xc["y"].to_numpy(), xc["lg"].to_numpy()))
    log(tag, json.dumps(res)); return res, x

log(f"model {name} on {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")
t0 = time.time()
zs = ev.sample(min(ev.height, 6000), seed=0)
r0, _ = report("ZERO-SHOT", zs, score(zs)); log(f"zero-shot scored {zs.height} in {time.time()-t0:.0f}s")
lr, bs = make_trainable()
params = [p for p in model.parameters() if p.requires_grad]
log(f"trainable params {sum(p.numel() for p in params)/1e6:.1f}M")
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
lossf = torch.nn.BCEWithLogitsLoss()
a, b, y = tr["t_s1"].to_list(), tr["t_r"].to_list(), torch.tensor(tr["y"].to_numpy(), dtype=torch.float32, device=dev)
steps, seen, t1, run = 0, 0, time.time(), []
model.train()
for i in range(0, len(a), bs):
    if time.time() - t1 > TRAIN_MIN * 60: break
    for g in opt.param_groups: g["lr"] = lr * min(1.0, (steps + 1) / 200)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        loss = lossf(fwd(enc(a[i:i + bs], b[i:i + bs])), y[i:i + bs])
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
    steps += 1; seen += len(a[i:i + bs]); run.append(loss.item())
    if steps % 200 == 0:
        log(f"step {steps} seen {seen} loss {np.mean(run[-200:]):.4f} {seen/(time.time()-t1):.1f} pairs/s")
log(f"trained {steps} steps, {seen} pairs ({seen/len(a):.2f} epoch) in {(time.time()-t1)/60:.1f} min")
ev = ev if ev.height <= EVAL_MAX else ev.filter(pl.col("s1").is_in(ev["s1"].unique().sort().head(int(EVAL_MAX / 18)).implode()))
t2 = time.time(); lg = score(ev); log(f"eval scored {ev.height} pairs in {time.time()-t2:.0f}s ({ev.height/(time.time()-t2):.0f} pairs/s)")
r1, x = report("FINE-TUNED", ev, lg)
x.select("s1", "r", "y", "country", "sc", "rk", "lg").write_parquet(f"{WD}/eval_{tag}.parquet")
model.save_pretrained(f"{WD}/model_{tag}"); tok.save_pretrained(f"{WD}/model_{tag}")
json.dump({"zero_shot": r0, "fine_tuned": r1, "pairs_seen": seen, "steps": steps, "model": name}, open(f"{WD}/metrics_{tag}.json", "w"), indent=1)
