"""Account-2 (satvik006) orchestrator for the second pass. Overnight orchestrator: launches each Kaggle job as soon as its upstream jobs are COMPLETE and a session slot is free.
Kaggle limits: 5 CPU, 2 GPU, (TPU queue) concurrent sessions. State persists in kaggle/.orchestrator_state.json.
Events (one line each) go to stdout and kaggle/orchestrator.log.  Usage: python kaggle/orchestrate.py"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = f"{ROOT}/.venv/bin/kaggle"; PY = f"{ROOT}/.venv/bin/python"
STATE = f"{ROOT}/kaggle2/.orchestrator2_state.json"; LOG = f"{ROOT}/kaggle2/orchestrator2.log"
os.environ["KAGGLE_API_TOKEN"] = open(os.path.expanduser("~/.kaggle2/access_token")).read().strip()
OWNER = "satvik006"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"}
ACTIVE = {"RUNNING", "QUEUED", "NEW", "PENDING"}
CANDS2 = ["er2-cands2-train-us", "er2-cands2-train-india", "er2-cands2-test-us", "er2-cands2-test-india", "er2-cands2-test-france"]
CPU_JOBS = {*CANDS2, "er2-matcher-full2", "er2-submit2"}
GPU_JOBS = set()
def push(name):
    return [K, "kernels", "push", "-p", f"{ROOT}/kaggle2/pass2/jobs/{name}"]
JOBS = [
    *[(n, "cpu", [], (lambda f: lambda st: push(f))("cands2_" + n[len("er2-cands2-"):].replace("-", "_"))) for n in CANDS2],
    ("er2-matcher-full2", "cpu", ["er2-cands2-train-us", "er2-cands2-train-india"], lambda st: push("matcher_full2")),
    ("er2-submit2", "cpu", ["er2-matcher-full2", "er2-cands2-test-us", "er2-cands2-test-india", "er2-cands2-test-france"], lambda st: push("submit2")),
]
WAIT_FOR = {}
GATES = {}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


def status(j):
    try:
        out = subprocess.run([K, "kernels", "status", f"{OWNER}/{j}"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "?"
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    return m.group(1) if m else ("MISSING" if "404" in out or "not found" in out.lower() else "?")


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"launched": {}}
    launched = st["launched"]
    prev = {}
    names = sorted(CPU_JOBS | GPU_JOBS)
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
            if (res == "gpu" and gpu_busy >= 2) or (res == "cpu" and cpu_busy >= 5):
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
        for sj, folder in (("er2-submit2", "submit2_acc2"),):
            if prev.get(sj) == "COMPLETE" and not launched.get(f"{sj}:fetched"):
                d = f"{ROOT}/kaggle/{folder}/kout"; os.makedirs(d, exist_ok=True)
                subprocess.run([K, "kernels", "output", f"{OWNER}/{sj}", "-p", d, "-o", "--file-pattern", r".*\.(tsv|txt)$"], capture_output=True, text=True)
                v = subprocess.run([PY, f"{ROOT}/student_resource/utils/validate_submission.py", "-m", f"{d}/matching_results.tsv",
                                    "-c", f"{d}/candidate_pairs.tsv", "-t", f"{ROOT}/student_resource/dataset/test"], capture_output=True, text=True)
                log(f"SUBMISSION {sj} fetched to kaggle/{folder}/kout; validator: {(v.stdout.strip().splitlines() or ['?'])[-1]}")
                launched[f"{sj}:fetched"] = 1; json.dump(st, open(STATE, "w"), indent=1)
        json.dump(st, open(STATE, "w"), indent=1)
        done = all(n in launched for n, *_ in JOBS) and all(prev.get(n) in TERMINAL for n, *_ in JOBS)
        if done:
            log("orchestrator: all jobs launched and finished"); break
        time.sleep(120)


if __name__ == "__main__":
    main()
