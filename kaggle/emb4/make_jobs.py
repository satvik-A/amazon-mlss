"""Embedding-arm jobs. Usage: python make_jobs.py <owner> <prefix> <jobs dir> [splits=train,test]"""
import json, os, sys
owner, prefix, jdir = sys.argv[1], sys.argv[2], sys.argv[3]
splits = sys.argv[4].split(",") if len(sys.argv) > 4 else ["train", "test"]
here = os.path.dirname(os.path.abspath(__file__))
data = ["satvikaderla/amazon-ml-er-2026-data"] if owner == "satvikaderla" else [f"{owner}/er-bundle"]
for split, countries in {"train": ["US", "India"], "test": ["US", "India", "France"]}.items():
    if split not in splits: continue
    for c in countries:
        name = f"{split}_{c}".lower(); d = f"{here}/{jdir}/{name}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/run_emb.py", "w").write(open(f"{here}/run_emb.py").read().replace("__SPLIT__", split).replace("__CTRY__", c))
        slug = f"{prefix}-{name.replace('_', '-')}"
        json.dump({"id": f"{owner}/{slug}", "title": slug, "code_file": "run_emb.py", "language": "python", "kernel_type": "script",
                   "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "dataset_sources": data, "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
        print(d, slug)
