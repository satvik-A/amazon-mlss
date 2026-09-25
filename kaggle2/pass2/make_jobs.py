"""Second pass on account 2 (satvik006, CPU, offline): r9 blocking -> full-mode pools -> matcher -> submission.
Builds kaggle2/pass2/jobs/<name>/ from the account-1 scripts (which carry an offline install path). Usage: make_jobs.py <ref>"""
import json, os, sys
ref = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(os.path.dirname(here))
DS = ["satvik006/er-bundle", "satvik006/er-src"]
def meta(slug, code, sources):
    return {"id": f"satvik006/{slug}", "title": slug, "code_file": code, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": DS, "competition_sources": [], "kernel_sources": [f"satvik006/{s}" for s in sources]}
def write(name, code, src, slug, sources):
    d = f"{here}/jobs/{name}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/{code}", "w").write(src)
    json.dump(meta(slug, code, sources), open(f"{d}/kernel-metadata.json", "w"), indent=1); print(d, slug)
C = open(f"{root}/kaggle/cands_full/run_cands_full.py").read()
for split, countries in {"train": ["US", "India"], "test": ["US", "India", "France"]}.items():
    for c in countries:
        n = f"{split}_{c}".lower()
        write(f"cands2_{n}", "run_cands_full.py", C.replace("__SPLIT__", split).replace("__COUNTRY__", c).replace("__REF__", ref),
              f"er2-cands2-{n.replace('_', '-')}", [])
write("matcher_full2", "run_matcher_full.py", open(f"{root}/kaggle/matcher_full/run_matcher_full.py").read().replace("__REF__", ref),
      "er2-matcher-full2", ["er2-cands2-train-us", "er2-cands2-train-india"])
write("submit2", "run_submit.py", open(f"{root}/kaggle/submit/run_submit.py").read().replace("__REF__", ref),
      "er2-submit2", ["er2-cands2-test-us", "er2-cands2-test-india", "er2-cands2-test-france", "er2-matcher-full2"])
