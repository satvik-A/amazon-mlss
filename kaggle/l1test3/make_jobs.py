"""One level-1 test job per country (pass 3): kaggle/l1test3/jobs/<country>/. Usage: python make_jobs.py <git ref>"""
import json, os, sys
ref = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
for c in ("US", "India", "France"):
    d = f"{here}/jobs/{c.lower()}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/run_submit.py", "w").write(open(f"{here}/run_submit.py").read().replace("__REF__", ref).replace("__ONLY__", c))
    slug = f"er-l1test3-{c.lower()}"
    json.dump({"id": f"satvikaderla/{slug}", "title": slug, "code_file": "run_submit.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
               "dataset_sources": ["satvikaderla/amazon-ml-er-2026-data"], "competition_sources": [],
               "kernel_sources": ["satvikaderla/er-artifacts", f"satvikaderla/er-cands3-test-{c.lower()}", "satvikaderla/er-matcher-full3"]},
              open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print(d)
