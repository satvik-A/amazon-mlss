"""Build + push a Kaggle job with the code pinned: copies <folder>/*.py and kernel-metadata.json into <folder>/build/,
replacing __REF__ with the git ref, then runs `kaggle kernels push`.
Usage: python kaggle/launch.py kaggle/<job> [ref] [--acc NvidiaTeslaT4|NvidiaTeslaP100]"""
import glob, os, shutil, subprocess, sys
args = sys.argv[1:]
acc = None
if "--acc" in args:
    i = args.index("--acc"); acc = args[i + 1]; del args[i:i + 2]
folder = args[0].rstrip("/")
ref = args[1] if len(args) > 1 else subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
b = f"{folder}/build"; shutil.rmtree(b, ignore_errors=True); os.makedirs(b)
for f in glob.glob(f"{folder}/*.py"):
    open(f"{b}/{os.path.basename(f)}", "w").write(open(f).read().replace("__REF__", ref))
shutil.copy(f"{folder}/kernel-metadata.json", b)
print(f"ref {ref}")
k = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "kaggle")
subprocess.run([k, "kernels", "push", "-p", b] + (["--accelerator", acc] if acc else []), check=False)
