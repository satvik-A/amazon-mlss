"""One kernel folder per blocking variant for account 2 (satvik006): kaggle2/recall_lab/jobs/<name>/"""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [
    {"name": "r0_base", "cap": 5000, "key_cap": 200, "caps": {}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6]},
    {"name": "r1_aux_x2", "cap": 5000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6]},
    {"name": "r2_df20k", "cap": 20000, "key_cap": 400, "caps": {}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6]},
    {"name": "r3_df2k", "cap": 2000, "key_cap": 200, "caps": {}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6]},
    {"name": "r4_prk100_tri25", "cap": 5000, "key_cap": 200, "caps": {"6": 25}, "keep_prk": 100, "arms": [0, 3, 4, 5, 6]},
    # round 2: the no-address arm (7)
    {"name": "r5_arm7", "cap": 5000, "key_cap": 200, "caps": {}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7]},
    {"name": "r6_arm7_cap20", "cap": 5000, "key_cap": 200, "caps": {"7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7]},
    {"name": "r7_arm7_aux2_df2k", "cap": 2000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25, "7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7]},
    # round 3: + arm 8 (core-name key x address number / word)
    {"name": "r8_arm78", "cap": 5000, "key_cap": 200, "caps": {}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7, 8]},
    {"name": "r9_arm78_aux2_df2k", "cap": 2000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25, "7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7, 8]},
    # round 4 (internet, ber from GitHub): address-word window = first words + the last 3 (city / state), vs r9 on the same code
    {"name": "r9b_base", "ref": "__REF__", "cap": 2000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25, "7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7, 8], "tail": 0},
    {"name": "r10_tail3", "ref": "__REF__", "cap": 2000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25, "7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7, 8], "tail": 3},
    {"name": "r11_tail3_places", "ref": "__REF__", "cap": 2000, "key_cap": 200, "caps": {"3": 30, "4": 30, "5": 40, "6": 25, "7": 20}, "keep_prk": 60, "arms": [0, 3, 4, 5, 6, 7, 8], "tail": 3, "places": 1},
]
import subprocess, sys
REF = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
ONLY = sys.argv[1:]
for v in VARIANTS:
    if ONLY and v["name"] not in ONLY: continue
    if v.get("ref") == "__REF__": v = dict(v, ref=REF)
    d = f"{here}/jobs/{v['name']}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/run_recall_lab.py", "w").write(open(f"{here}/run_recall_lab.py").read().replace("'__VARIANT__'", repr(json.dumps(v))).replace('"__VARIANT__".startswith', '"x".startswith'))
    slug = "er2-recall-" + v["name"].replace("_", "-")
    json.dump({"id": f"satvik006/{slug}", "title": slug, "code_file": "run_recall_lab.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": bool(v.get("ref")),
               "dataset_sources": ["satvik006/er-bundle", "satvik006/er-src"], "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print(d, slug)
