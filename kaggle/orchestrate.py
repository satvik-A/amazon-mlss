"""Overnight orchestrator: launches each Kaggle job as soon as its upstream jobs are COMPLETE and a session slot is free.
Kaggle limits: 5 CPU, 2 GPU, (TPU queue) concurrent sessions. State persists in kaggle/.orchestrator_state.json.
Events (one line each) go to stdout and kaggle/orchestrator.log.  Usage: python kaggle/orchestrate.py"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = f"{ROOT}/.venv/bin/kaggle"; PY = f"{ROOT}/.venv/bin/python"
STATE = f"{ROOT}/kaggle/.orchestrator_state.json"; LOG = f"{ROOT}/kaggle/orchestrator.log"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}
ACTIVE = {"RUNNING", "QUEUED", "NEW", "PENDING"}
CANDS2 = ["er-cands2-train-us", "er-cands2-train-india", "er-cands2-test-us", "er-cands2-test-india", "er-cands2-test-france"]
CPU_JOBS = {"er-blocking-v5", "er-cands-train-us", "er-cands-train-india", "er-cands-test-us", "er-cands-test-india",
            "er-matcher-full", "er-submit", "er-stack", "er-matcher-full2", "er-submit2", "er-submit3", *CANDS2}
GPU_JOBS = {"er-xenc-v1", "er-xenc-v1b", "er-xenc-v2", "er-xenc-v3", "er-xenc-v4", "er-xenc-score", "er-xenc-score-test"}
XENC = ["er-xenc-v1", "er-xenc-v1b", "er-xenc-v2", "er-xenc-v3", "er-xenc-v4"]
# pass 3 (blocking ADDR_TAIL=3): per-country level-1 test + cross-encoder jobs in parallel, final merge without recomputation
C3 = ["us", "india", "france"]
CANDS3 = [f"er-cands3-{s}" for s in ("train-us", "train-india", "test-us", "test-india", "test-france")]
P3_CPU = {*CANDS3, "er-matcher-full3", "er-stack3", *[f"er-l1test3-{c}" for c in C3], "er-final3"}
P3_GPU = {"er-xenc-score3", *[f"er-xenc-score-test3-{c}" for c in C3]}
CPU_JOBS |= P3_CPU; GPU_JOBS |= P3_GPU
# pass 4 (looser blocking, best-F cut-off; candidate-set size no longer a constraint)
CANDS4 = [f"er-cands4-{s}" for s in ("train-us", "train-india", "test-us", "test-india", "test-france")]
P4_CPU = {*CANDS4, "er-matcher-full4", "er-stack4", *[f"er-l1test4-{c}" for c in C3], "er-final4"}
P4_GPU = {"er-xenc-score4", *[f"er-xenc-score-test4-{c}" for c in C3]}
EMB4 = [f"er-emb4-{s}" for s in ("train-india", "train-us", "test-india", "test-us", "test-france")]
P4_GPU |= set(EMB4)
CPU_JOBS |= P4_CPU; GPU_JOBS |= P4_GPU

# name -> (resource, deps (must be COMPLETE), launch command builder), in priority order
def launch(folder, acc=None, sources=None):
    cmd = [PY, f"{ROOT}/kaggle/launch.py", f"{ROOT}/{folder}"]
    if acc: cmd += ["--acc", acc]
    if sources is not None: cmd += ["--sources", ",".join(sources)]
    return cmd
def prebuilt(folder, acc):
    return [K, "kernels", "push", "-p", f"{ROOT}/{folder}/build", "--accelerator", acc]

JOBS = [
    ("er-matcher-full", "cpu", ["er-cands-train-us", "er-cands-train-india"], lambda st: launch("kaggle/matcher_full")),
    ("er-submit", "cpu", ["er-matcher-full", "er-cands-test-us", "er-cands-test-india", "er-cands-test-france"],
     lambda st: launch("kaggle/submit")),
    # cross-encoder scores for B/C: needs the matcher's level-1 files and at least the bge run; uses every finished bake-off
    ("er-xenc-score", "gpu", ["er-matcher-full2", "er-xenc-v1", "er-xenc-v4"],
     lambda st: launch("kaggle/xenc_score", "NvidiaTeslaT4", ["er-xenc-v1", "er-xenc-v4", "er-matcher-full2"])),   # bge + MuRIL + CANINE (Qwen 4B / 3.5 too slow on T4)
    ("er-stack", "cpu", ["er-xenc-score"], lambda st: launch("kaggle/stack", sources=["er-matcher-full2", "er-xenc-score"])),
    # stacked submission: cross-encoder scores on the TEST uncertain band (level-1 files from er-submit2), then submit3
    ("er-xenc-score-test", "gpu", ["er-submit2", "er-stack", "er-xenc-v1", "er-xenc-v4"], lambda st: launch("kaggle/xenc_score_test", "NvidiaTeslaT4")),
    ("er-submit3", "cpu", ["er-xenc-score-test", "er-stack", "er-matcher-full2"], lambda st: launch("kaggle/submit3")),
    ("er-xenc-v2", "gpu", [], lambda st: prebuilt("kaggle/xenc_v2", "NvidiaTeslaT4")),
    # second pass (blocking arms 7 + 8): gated by the file kaggle/.cands2_go (created once the config is chosen + jobs2 built)
    *[(n, "cpu", [], (lambda f: lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/cands_full/jobs2/{f}"])(n[len("er-cands2-"):].replace("-", "_")))
      for n in CANDS2],
    ("er-matcher-full2", "cpu", ["er-cands2-train-us", "er-cands2-train-india"], lambda st: launch("kaggle/matcher_full2")),
    ("er-submit2", "cpu", ["er-matcher-full2", "er-cands2-test-us", "er-cands2-test-india", "er-cands2-test-france"], lambda st: launch("kaggle/submit2")),
    ("er-xenc-v3", "gpu", [], lambda st: prebuilt("kaggle/xenc_v3", "NvidiaTeslaT4")),
    # ---- pass 3 ----
    ("er-cands3-test-france", "cpu", [], lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/cands_full/jobs3/test_france"]),
    ("er-matcher-full3", "cpu", ["er-cands3-train-us", "er-cands3-train-india"], lambda st: launch("kaggle/matcher_full3")),
    ("er-xenc-score3", "gpu", ["er-matcher-full3"], lambda st: launch("kaggle/xenc_score3", "NvidiaTeslaT4")),
    ("er-stack3", "cpu", ["er-xenc-score3"], lambda st: launch("kaggle/stack3")),
    *[(f"er-l1test3-{c}", "cpu", ["er-matcher-full3", f"er-cands3-test-{c}"], (lambda c: lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/l1test3/jobs/{c}"])(c)) for c in C3],
    *[(f"er-xenc-score-test3-{c}", "gpu", [f"er-l1test3-{c}", "er-stack3"], (lambda c: lambda st: launch(f"kaggle/xenc_score_test3/jobs/{c}", "NvidiaTeslaT4"))(c)) for c in C3],
    ("er-final3", "cpu", ["er-stack3", *[f"er-l1test3-{c}" for c in C3], *[f"er-xenc-score-test3-{c}" for c in C3]], lambda st: launch("kaggle/final3")),
    # ---- pass 4: embedding arm (GPU) -> candidates (word arms + embedding arm) -> ... ----
    *[(n, "gpu", [], (lambda f: lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/emb4/jobs/{f}", "--accelerator", "NvidiaTeslaT4"])(n[len("er-emb4-"):].replace("-", "_"))) for n in EMB4],
    *[(n, "cpu", [f"er-emb4-{n[len('er-cands4-'):]}"], (lambda f: lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/cands_full/jobs4d/{f}"])(n[len("er-cands4-"):].replace("-", "_"))) for n in CANDS4],
    ("er-matcher-full4", "cpu", ["er-cands4-train-us", "er-cands4-train-india"], lambda st: launch("kaggle/matcher_full4")),
    *[(f"er-l1test4-{c}", "cpu", ["er-matcher-full4", f"er-cands4-test-{c}"], (lambda c: lambda st: [K, "kernels", "push", "-p", f"{ROOT}/kaggle/l1test4/jobs/{c}"])(c)) for c in C3],
    ("er-xenc-score4", "gpu", ["er-matcher-full4"], lambda st: launch("kaggle/xenc_score4", "NvidiaTeslaT4")),
    ("er-stack4", "cpu", ["er-xenc-score4"], lambda st: launch("kaggle/stack4")),
    *[(f"er-xenc-score-test4-{c}", "gpu", [f"er-l1test4-{c}"], (lambda c: lambda st: launch(f"kaggle/xenc_score_test4/jobs/{c}", "NvidiaTeslaT4"))(c)) for c in C3],
    ("er-final4", "cpu", ["er-stack4", *[f"er-l1test4-{c}" for c in C3], *[f"er-xenc-score-test4-{c}" for c in C3]], lambda st: launch("kaggle/final4")),
]
# the scoring job waits for v1b to finish (so the Qwen models are included) unless v1b failed
WAIT_FOR = {"er-xenc-score": ["er-xenc-v4"]}
GATES = {n: f"{ROOT}/kaggle/.cands2_go" for n in CANDS2}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def status(j):
    try:
        out = subprocess.run([K, "kernels", "status", f"satvikaderla/{j}"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "?"
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    return m.group(1) if m else ("MISSING" if "404" in out or "not found" in out.lower() else "?")


def stack_passed(launched) -> bool:
    """True if er-stack's level 2 beat level 1 on holdout C by >= 0.002 (else skip the test scoring + submit3)."""
    if "stack:delta" not in launched:
        d = f"{ROOT}/kaggle/stack/kout"; os.makedirs(d, exist_ok=True)
        subprocess.run([K, "kernels", "output", "satvikaderla/er-stack", "-p", d, "-o", "--file-pattern", r".*\.(json|txt)$"], capture_output=True, text=True)
        try:
            launched["stack:delta"] = json.load(open(f"{d}/decision_stack.json")).get("delta_C", -1)
        except Exception:
            return False
        log(f"STACK gate: delta on C {launched['stack:delta']:+.4f} -> {'ACCEPT' if launched['stack:delta'] >= 0.002 else 'REJECT (no stacked submission)'}")
    if launched["stack:delta"] < 0.002:
        for n in ("er-xenc-score-test", "er-submit3"):
            launched.setdefault(n, "skipped (stack rejected)")
        return False
    return True


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"launched": {}}
    launched = st["launched"]
    prev = {}
    names = sorted(CPU_JOBS | GPU_JOBS | {"er-cands-test-france"})
    log("orchestrator started")
    while True:
        cur = {j: status(j) for j in names}
        for j, s in cur.items():
            if prev.get(j) != s and s != "?":
                log(f"{j}: {s}")
        prev.update({k: v for k, v in cur.items() if v != "?"})
        gpu_busy = sum(prev.get(j) in ACTIVE for j in GPU_JOBS)
        cpu_busy = sum(prev.get(j) in ACTIVE for j in CPU_JOBS)
        for name, res, deps, cmd in JOBS:
            if name in launched:
                continue
            if prev.get(name) in ACTIVE:             # already running (launched by hand)
                launched[name] = "external"; continue
            if any(prev.get(d) in {"ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"} for d in deps):
                if not launched.get(name + ":blocked"):
                    log(f"BLOCKED {name}: upstream failed ({[d for d in deps if prev.get(d) != 'COMPLETE']})"); launched[name + ":blocked"] = 1
                continue
            if not all(prev.get(d) == "COMPLETE" for d in deps):
                continue
            if any(prev.get(w) in ACTIVE for w in WAIT_FOR.get(name, [])):
                continue
            if name in GATES and not os.path.exists(GATES[name]):
                continue
            if name == "er-xenc-score-test" and not stack_passed(launched):
                continue
            # Kaggle counts GPU sessions in the batch limit of 5 too
            if (res == "gpu" and gpu_busy >= 2) or cpu_busy + gpu_busy >= 5:
                continue
            # keep a slot for tonight's stacked submission until it is launched
            if (name in P3_CPU or name in P3_GPU) and "er-submit3" not in launched and cpu_busy + gpu_busy >= 4:
                continue
            c = cmd(prev)
            r = subprocess.run(c, capture_output=True, text=True, cwd=ROOT)
            msg = (r.stdout + r.stderr).strip().splitlines()
            ok = any("successfully pushed" in m for m in msg)
            log(f"LAUNCH {name}: {'ok' if ok else 'FAILED'} :: {msg[-1] if msg else ''}")
            if ok:
                launched[name] = time.strftime("%H:%M")
                gpu_busy += res == "gpu"; cpu_busy += res == "cpu"
            json.dump(st, open(STATE, "w"), indent=1)
        # submission files: download + official validator, once per submit job
        for sj, folder in (("er-submit", "submit"), ("er-submit2", "submit2"), ("er-submit3", "submit3"), ("er-final3", "final3"), ("er-final4", "final4")):
            if prev.get(sj) == "COMPLETE" and not launched.get(f"{sj}:fetched"):
                d = f"{ROOT}/kaggle/{folder}/kout"; os.makedirs(d, exist_ok=True)
                subprocess.run([K, "kernels", "output", f"satvikaderla/{sj}", "-p", d, "-o", "--file-pattern", r".*\.(tsv|txt)$"], capture_output=True, text=True)
                v = subprocess.run([PY, f"{ROOT}/student_resource/utils/validate_submission.py", "-m", f"{d}/matching_results.tsv",
                                    "-c", f"{d}/candidate_pairs.tsv", "-t", f"{ROOT}/student_resource/dataset/test"], capture_output=True, text=True)
                log(f"SUBMISSION {sj} fetched to kaggle/{folder}/kout; validator: {(v.stdout.strip().splitlines() or ['?'])[-1]}")
                if os.path.exists(f"{d}/l1/matching_results.tsv"):
                    v = subprocess.run([PY, f"{ROOT}/student_resource/utils/validate_submission.py", "-m", f"{d}/l1/matching_results.tsv",
                                        "-c", f"{d}/l1/candidate_pairs.tsv", "-t", f"{ROOT}/student_resource/dataset/test"], capture_output=True, text=True)
                    log(f"SUBMISSION {sj} level-1 fallback in kaggle/{folder}/kout/l1; validator: {(v.stdout.strip().splitlines() or ['?'])[-1]}")
                launched[f"{sj}:fetched"] = 1; json.dump(st, open(STATE, "w"), indent=1)
        json.dump(st, open(STATE, "w"), indent=1)
        done = all(n in launched for n, *_ in JOBS) and all(prev.get(n) in TERMINAL for n, *_ in JOBS)
        if done:
            log("orchestrator: all jobs launched and finished"); break
        time.sleep(120)


if __name__ == "__main__":
    main()
