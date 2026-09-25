"""Build + push a Kaggle job with the code pinned: copies <folder>/*.py and kernel-metadata.json into <folder>/build/,
replacing __REF__ with the git ref, then runs `kaggle kernels push`. Usage: python kaggle/launch.py kaggle/<job> [ref]"""
import glob, os, shutil, subprocess, sys
folder = sys.argv[1].rstrip("/")
ref = sys.argv[2] if len(sys.argv) > 2 else subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
b = f"{folder}/build"; shutil.rmtree(b, ignore_errors=True); os.makedirs(b)
for f in glob.glob(f"{folder}/*.py"):
    open(f"{b}/{os.path.basename(f)}", "w").write(open(f).read().replace("__REF__", ref))
shutil.copy(f"{folder}/kernel-metadata.json", b)
print(f"ref {ref}")
k = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "kaggle")
subprocess.run([k, "kernels", "push", "-p", b], check=False)
