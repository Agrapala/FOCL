"""
FLoBC - Master Runner
======================
Run everything in order:

    python run_all.py

Stages:
  1. run_paper_experiments.py  - Paper section 5.1-5.4 (4 experiments, all charts)
  2. prove_objectives.py       - All 4 research objectives with metrics + charts
  3. generate_mechanism_doc.py - Word document (block diagram reference)
"""

import sys, os, subprocess, time

ROOT   = os.path.dirname(os.path.abspath(__file__))
PY     = sys.executable
DIV    = "=" * 64


def run(label, script):
    print(f"\n{DIV}")
    print(f"  RUNNING: {label}")
    print(DIV)
    t0  = time.time()
    ret = subprocess.run([PY, os.path.join(ROOT, script)], cwd=ROOT)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n  FAILED: {script}  (exit {ret.returncode})")
        sys.exit(1)
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    os.chdir(ROOT)
    total_start = time.time()

    run("Paper Experiments  (section 5.1 Benchmark, 5.2 Ratio, "
        "5.3 Reward-Penalty, 5.4 Sync)",
        "run_paper_experiments.py")

    run("Research Objective Proof  (Obj 1: Platform, Obj 2: Accuracy, "
        "Obj 3: Byzantine, Obj 4: Credentials)",
        "prove_objectives.py")

    run("Mechanism Word Document",
        "generate_mechanism_doc.py")

    total = time.time() - total_start
    print(f"\n{DIV}")
    print(f"  ALL DONE  -  total time: {total/60:.1f} min")
    print(f"")
    print(f"  Outputs:")
    print(f"    results/paper_experiments.json     paper experiment metrics")
    print(f"    results/objectives_proof.json      all 4 objective proofs")
    print(f"    results/blockchain_proof.json      full blockchain export")
    print(f"    results/FLoBC_PoCL_Mechanism.docx  Word document")
    print(f"    dashboard/paper_*.png              4 paper experiment charts")
    print(f"    dashboard/obj1_platform.png        Obj 1: platform evidence")
    print(f"    dashboard/obj2_improvement.png     Obj 2: accuracy gain")
    print(f"    dashboard/obj3_byzantine.png       Obj 3: malicious rejection")
    print(f"    dashboard/obj4_credentials.png     Obj 4: credential management")
    print(DIV + "\n")
