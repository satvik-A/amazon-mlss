"""Build + push a Kaggle job with the code pinned: copies <folder>/*.py and kernel-metadata.json into <folder>/build/,
replacing __REF__ with the git ref, then runs `kaggle kernels push`.
Usage: python kaggle/launch.py kaggle/<job> [ref] [--acc NvidiaTeslaT4|NvidiaTeslaP100|Tpu1VmV38] [--sources k1,k2]
--sources replaces kernel_sources (slugs without the owner), e.g. to include only finished upstream kernels."""
import glob, os, shutil, subprocess, sys
args = sys.argv[1:]
acc = None
if "--acc" in args:
    i = args.index("--acc"); acc = args[i + 1]; del args[i:i + 2]
srcs = None
if "--sources" in args:
    i = args.index("--sources"); srcs = args[i + 1].split(","); del args[i:i + 2]
folder = args[0].rstrip("/")
ref = args[1] if len(args) > 1 else subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
b = f"{folder}/build"; shutil.rmtree(b, ignore_errors=True); os.makedirs(b)
for f in glob.glob(f"{folder}/*.py"):
    open(f"{b}/{os.path.basename(f)}", "w").write(open(f).read().replace("__REF__", ref))
import json
meta = json.load(open(f"{folder}/kernel-metadata.json"))
if srcs is not None:
    meta["kernel_sources"] = [f"satvikaderla/{x}" for x in srcs if x]
json.dump(meta, open(f"{b}/kernel-metadata.json", "w"), indent=1)
print(f"ref {ref}")
k = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "kaggle")
subprocess.run([k, "kernels", "push", "-p", b] + (["--accelerator", acc] if acc else []), check=False)
