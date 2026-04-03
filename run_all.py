import subprocess
import sys
import time

steps = [
    ("data_prep.py",    "preparing data"),
    ("rankings.py",     "computing rankings"),
    ("ml_pipeline.py",  "training models"),
    ("ml_viz.py",       "generating ml plots"),
    ("rankings_viz.py", "generating ranking plots"),
]

for script, label in steps:
    print(f"\n[{label}]")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"failed on {script} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"done in {elapsed:.1f}s")

print("\nall steps complete")
