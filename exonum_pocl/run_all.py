"""
FLoBC-PoCL Master Runner — Exonum Blockchain
=============================================
Run everything from the exonum_pocl folder:

    cd C:\\Users\\SASINI\\Desktop\\research\\flobc\\exonum_pocl
    python run_all.py

Or from the flobc root:
    python exonum_pocl/run_all.py

Steps:
  0. Dependency check
  1. Quick sanity test (1 round, all 3 consensus modes)
  2. Full experiments (pBFT / PoCL-pBFT / PoS + HP tuning)
  3. Generate charts (if matplotlib available)
  4. Print final summary table
"""

import sys
import os
import time
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DIV = "=" * 70


def header(title):
    print(f"\n{DIV}\n  {title}\n{DIV}")


def check_deps():
    header("STEP 0 — Dependency Check")
    required = ["numpy"]
    optional = ["matplotlib"]

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK  {pkg}")
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([
                sys.executable, "-m", "pip",
                "install", pkg, "--break-system-packages", "--quiet",
            ])

    has_mpl = True
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  OK  {pkg} (charts enabled)")
        except ImportError:
            has_mpl = False
            print(f"  --  {pkg} not found — charts disabled")
            print(f"      pip install matplotlib")

    return has_mpl


def sanity_test():
    header("STEP 1 — Sanity Test (1 round, all 3 consensus modes)")
    from fl.engine import FLoCBPoCL, HyperParams

    hp = HyperParams(lr=5e-4, epochs_per_rnd=1, batch_size=64,
                     hidden_dims=[256, 128, 64])
    for mode in ["pbft", "pocl_pbft", "pos"]:
        print(f"\n  Testing {mode.upper()}...", end=" ", flush=True)
        try:
            eng = FLoCBPoCL(hp=hp, consensus_mode=mode, verbose=False)
            res = eng.train(n_rounds=1)
            acc = res["global_acc_final"]
            ok  = res["chain_valid"]
            print(f"OK  acc={acc:.4f}  chain_valid={ok}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    print("\n  All sanity tests passed")


def run_experiments():
    header("STEP 2 — Full Experiment Suite")
    ret = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_experiments.py")],
        cwd=ROOT,
    )
    if ret.returncode != 0:
        print("  Experiments FAILED. Check errors above.")
        sys.exit(1)
    print("  All experiments complete")


def generate_charts():
    header("STEP 3 — Chart Generation")
    plot_script = os.path.join(ROOT, "dashboard", "plot_results.py")
    ret = subprocess.run([sys.executable, plot_script], cwd=ROOT)
    if ret.returncode != 0:
        print("  Chart generation had issues (matplotlib installed?)")
    else:
        print("  Charts saved to dashboard/")


def print_final_summary():
    header("STEP 4 — Final Summary")
    master_path = os.path.join(ROOT, "results", "master_results.json")
    if not os.path.exists(master_path):
        print("  results/master_results.json not found — skipping summary")
        return

    with open(master_path) as f:
        master = json.load(f)

    meta = master.get("meta", {})
    exps = master.get("experiments", {})

    print(f"\n  Blockchain    : {meta.get('blockchain', '?')}")
    print(f"  Hospital Nodes: {meta.get('n_hospital_nodes', 4)}")
    print(f"  Total runtime : {meta.get('total_time_sec', '?')}s")

    print(f"\n  {'Mechanism':<20} {'G.Acc%':>8} {'Delay ms':>10} "
          f"{'Gas(avg)':>10} {'ChainLen':>10} {'Valid':>7}")
    print(f"  {'─'*70}")
    for name, res in exps.items():
        print(f"  {name:<20} "
              f"{res['global_acc_final']*100:>7.2f}% "
              f"{res['avg_consensus_delay_ms']:>9.3f}ms "
              f"{res['avg_gas_equivalent']:>10.1f} "
              f"{res['chain_length']:>10} "
              f"{'OK' if res['chain_valid'] else 'ERR':>7}")

    print(f"\n  Node A Target >=89%:")
    for name, res in exps.items():
        na = float(res.get("per_node_acc_final", {}).get("0", 0)) * 100
        ok = "OK" if na >= 89.0 else "MISS"
        print(f"    {name:<22} NodeA={na:.2f}% {ok}")

    print(f"\n  FL Improvement (best tuned run):")
    best_exp = exps.get("PoCL-pBFT-Tuned", list(exps.values())[-1])
    nnames   = ["Node A", "Node B", "Node C", "Node D"]
    for i, nm in enumerate(nnames):
        imp = best_exp.get("fl_improvement_pct", {}).get(str(i), "?")
        if isinstance(imp, (int, float)):
            flag = ">=5pp OK" if imp >= 5 else "<5pp MISS"
            print(f"    {nm}: {imp:+.2f}pp  ({flag})")
        else:
            print(f"    {nm}: {imp}")

    gs = master.get("ipfs_gas_savings", {})
    print(f"\n  IPFS Gas Savings    : {gs.get('savings_pct', '?')}%")
    print(f"  Model size on IPFS  : CID only (48 bytes) vs "
          f"{gs.get('model_size_bytes','?')} bytes full model")

    best_hp = master.get("best_hyperparams", {})
    print(f"\n  Best Hyperparameters (from grid search):")
    for k, v in best_hp.items():
        print(f"    {k}: {v}")

    print(f"\n  Output files:")
    results_dir = os.path.join(ROOT, "results")
    dash_dir    = os.path.join(ROOT, "dashboard")
    for fname in sorted(os.listdir(results_dir)):
        print(f"    results/{fname}")
    for fname in sorted(os.listdir(dash_dir)):
        if fname.endswith(".png"):
            print(f"    dashboard/{fname}")

    print(f"\n{DIV}")
    print(f"  FLoBC-PoCL on Exonum — COMPLETE")
    print(f"  pBFT + PoCL-pBFT + PoS | 4-node healthcare FL | CNN")
    print(DIV + "\n")


if __name__ == "__main__":
    os.chdir(ROOT)
    t_start = time.time()

    has_mpl = check_deps()
    sanity_test()
    run_experiments()
    if has_mpl:
        generate_charts()
    print_final_summary()

    print(f"\n  Total wall-clock time: {time.time()-t_start:.1f}s\n")
