"""One global-assignment job per train country (account 2). Usage: python make_jobs.py <git ref>"""
import json, os, sys
ref = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
for c in ("India", "US"):
    d = f"{here}/jobs/{c.lower()}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/run_global.py", "w").write(open(f"{here}/run_global.py").read().replace("__REF__", ref).replace("__CTRY__", c))
    slug = f"er2-global-{c.lower()}"
    json.dump({"id": f"satvik006/{slug}", "title": slug, "code_file": "run_global.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
               "dataset_sources": ["satvik006/er-bundle"], "competition_sources": [],
               "kernel_sources": [f"satvik006/er2-cands2-train-{c.lower()}", "satvik006/er2-matcher-full2"]}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print(d)
