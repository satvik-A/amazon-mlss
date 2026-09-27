"""Account 3 (satvik0006): per train country, K scoring parts (shards k::K) + one combine job (competition + sibling
features over all S1s). Usage: python make_jobs.py <git ref>   -> jobs/<country>_p<k>, jobs/<country>_combine"""
import json, os, sys
ref = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
src = open(f"{here}/run_global4.py").read()
meta = lambda slug, ds, ks: {"id": f"satvik0006/{slug}", "title": slug, "code_file": "run_global4.py", "language": "python", "kernel_type": "script",
                             "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                             "dataset_sources": ds, "competition_sources": [], "kernel_sources": ks}
for c, K in (("India", 2), ("US", 3)):
    parts = []
    for name, part in [*[(f"p{k}", f"{k}/{K}") for k in range(K)], ("combine", "combine")]:
        d = f"{here}/jobs/{c.lower()}_{name}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/run_global4.py", "w").write(src.replace("__REF__", ref).replace("__CTRY__", c).replace("__PART__", part))
        slug = f"er3-global4-{c.lower()}-{name}"
        if name == "combine":
            m = meta(slug, ["satvik0006/er-bundle"], [f"satvik0006/{p}" for p in parts])
        else:
            m = meta(slug, ["satvik0006/er-bundle", "satvik0006/er3-matcher4"], [f"satvik0006/er3-cands4-train-{c.lower()}"]); parts.append(slug)
        json.dump(m, open(f"{d}/kernel-metadata.json", "w"), indent=1)
        print(d)
