"""Create one Kaggle kernel folder per (split, country): kaggle/cands_full/jobs/<split>_<country>/ (script with config filled in + metadata).
Usage: python make_jobs.py <git ref>   then  kaggle kernels push -p kaggle/cands_full/jobs/<name>"""
import json, os, sys
ref = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
for split, countries in {"train": ["US", "India"], "test": ["US", "India", "France"]}.items():
    for c in countries:
        name = f"{split}_{c}".lower(); d = f"{here}/jobs/{name}"; os.makedirs(d, exist_ok=True)
        src = open(f"{here}/run_cands_full.py").read().replace("__SPLIT__", split).replace("__COUNTRY__", c).replace("__REF__", ref)
        open(f"{d}/run_cands_full.py", "w").write(src)
        json.dump({"id": f"satvikaderla/er-cands-{name}", "title": f"er-cands-{name}", "code_file": "run_cands_full.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
                   "enable_internet": True, "dataset_sources": ["satvikaderla/amazon-ml-er-2026-data"], "competition_sources": [],
                   "kernel_sources": ["satvikaderla/er-artifacts"]}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
        print(d)
