"""One job per train country on account 3 (satvik0006). Usage: python make_jobs.py <git ref>"""
import json, os, sys
ref = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
for c in ("India", "US"):
    d = f"{here}/jobs/{c.lower()}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/run_global4.py", "w").write(open(f"{here}/run_global4.py").read().replace("__REF__", ref).replace("__CTRY__", c))
    slug = f"er3-global4-{c.lower()}"
    json.dump({"id": f"satvik0006/{slug}", "title": slug, "code_file": "run_global4.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
               "dataset_sources": ["satvik0006/er-bundle", "satvik0006/er3-matcher4"], "competition_sources": [],
               "kernel_sources": [f"satvik0006/er3-cands4-train-{c.lower()}"]}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print(d)
