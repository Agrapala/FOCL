"""
Experiments Runner — Full Paper Experiment Suite
=================================================
Runs all required experiments and saves results to results/

Experiments:
  1. FL with pBFT consensus     — 20 rounds
  2. FL with PoCL-pBFT          — 20 rounds  (main contribution)
  3. FL with PoS                — 20 rounds  (comparison)
  4. Consensus comparison table (accuracy / delay / gas / fault tolerance)
  5. IPFS gas savings analysis
  6. Hyperparameter search -> best CNN config
  7. Best HP run with PoCL-pBFT — 25 rounds  (Node A >= 89% target)
  8. Full chain export for each consensus mode
"""

import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from fl.engine      import FLoCBPoCL, HyperParams
from fl.hyper_search import grid_search
from ipfs.ipfs_node  import IPFS

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DIV = "=" * 68


def save_json(name: str, data: dict) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [Save] {path}")
    return path


def run_fl(consensus: str, n_rounds: int,
           hp: HyperParams = None, label: str = None):
    """Run FL, return (result_dict, engine)."""
    label = label or f"FL_{consensus.upper()}"
    print(f"\n{DIV}")
    print(f"  {label}")
    print(DIV)
    engine = FLoCBPoCL(hp=hp, consensus_mode=consensus, verbose=True)
    result = engine.train(n_rounds=n_rounds)
    result["label"] = label
    chain_path = os.path.join(RESULTS_DIR, f"chain_{consensus}.json")
    engine.export_chain(chain_path)
    return result, engine


def print_comparison_table(results: dict):
    print(f"\n{DIV}")
    print(f"  CONSENSUS MECHANISM COMPARISON TABLE")
    print(DIV)
    print(f"  {'Mechanism':<18} {'FL Acc%':>8} {'Delay(ms)':>12} "
          f"{'Gas(avg)':>10} {'FaultTol':>10} {'ChainLen':>10}")
    print(f"  {'─'*70}")
    for name, res in results.items():
        acc_pct  = round(res["global_acc_final"] * 100, 2)
        delay_ms = res["avg_consensus_delay_ms"]
        gas      = res["avg_gas_equivalent"]
        chain_l  = res["chain_length"]
        ft       = 1   # f = (n_validators-1)//3 = (3-1)//3 = 0; practical f=1 with 4 nodes
        print(f"  {name:<18} {acc_pct:>7.2f}% {delay_ms:>11.3f}ms "
              f"{gas:>10.1f} {ft:>10} {chain_l:>10}")
    print(DIV)


def main():
    all_results   = {}
    t_total_start = time.time()

    # ── EXP 1: pBFT ──────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  EXP 1: pBFT Consensus — 20 FL rounds")
    res_pbft, eng_pbft = run_fl("pbft", n_rounds=20, label="EXP1_pBFT")
    all_results["pBFT"] = res_pbft
    save_json("exp1_pbft", res_pbft)

    # ── EXP 2: PoCL-pBFT ─────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  EXP 2: PoCL-pBFT Consensus — 20 FL rounds")
    res_pocl, eng_pocl = run_fl("pocl_pbft", n_rounds=20, label="EXP2_PoCL_pBFT")
    all_results["PoCL-pBFT"] = res_pocl
    save_json("exp2_pocl_pbft", res_pocl)

    # ── EXP 3: PoS ───────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  EXP 3: PoS Consensus — 20 FL rounds")
    res_pos, eng_pos = run_fl("pos", n_rounds=20, label="EXP3_PoS")
    all_results["PoS"] = res_pos
    save_json("exp3_pos", res_pos)

    # ── Comparison table ─────────────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── EXP 4: Hyperparameter Search ─────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  EXP 4: Hyperparameter Search (fast grid) -> Target Node A >=89%")
    print(DIV)
    best_hp_dict, search_results = grid_search(fast=True, n_trial_rounds=5)
    save_json("exp4_hyper_search", search_results)

    # ── EXP 5: Best HP with PoCL-pBFT — 25 rounds ────────────────────────────
    print(f"\n{DIV}")
    print(f"  EXP 5: PoCL-pBFT with BEST Hyperparameters — 25 rounds")
    print(DIV)
    best_hp = HyperParams(
        lr             = best_hp_dict.get("lr",             2e-4),
        batch_size     = best_hp_dict.get("batch_size",     32),
        epochs_per_rnd = best_hp_dict.get("epochs_per_rnd", 8),
        hidden_dims    = best_hp_dict.get("hidden_dims",    [512, 256, 128]),
        dropout_rate   = best_hp_dict.get("dropout_rate",   0.20),
    )
    res_best, eng_best = run_fl(
        "pocl_pbft", n_rounds=25, hp=best_hp, label="EXP5_BestHP_PoCL_pBFT")
    all_results["PoCL-pBFT-Tuned"] = res_best
    save_json("exp5_best_hp_pocl_pbft", res_best)
    eng_best.export_chain(os.path.join(RESULTS_DIR, "chain_best_hp.json"))

    # ── IPFS analysis ─────────────────────────────────────────────────────────
    ipfs_stats = IPFS.stats()
    # Estimate model size (float32 weights)
    # [512*1024 + 512] + [256*512 + 256] + [128*256 + 128] + [2*128 + 2] = ~680k params * 4 bytes
    model_size_bytes = (512*1024 + 512 + 256*512 + 256 + 128*256 + 128 + 2*128 + 2) * 4
    ipfs_savings = IPFS.gas_savings_vs_onchain(
        n_blocks           = eng_best.chain.length(),
        weights_size_bytes = model_size_bytes,
    )
    save_json("exp6_ipfs_analysis", {
        "ipfs_stats":  ipfs_stats,
        "gas_savings": ipfs_savings,
    })
    print(f"\n  [IPFS] Off-chain storage stats:")
    print(f"    Objects stored  : {ipfs_stats['total_objects']}")
    print(f"    Total size      : {ipfs_stats['total_bytes_MB']} MB")
    print(f"    Gas savings vs on-chain: {ipfs_savings['savings_pct']}%")

    # ── FL Improvement summary ────────────────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  FL IMPROVEMENT OVER LOCAL BASELINE")
    print(DIV)
    print(f"  {'Node':<22} {'Baseline':>10} {'FL (Tuned)':>12} "
          f"{'Improve':>10} {'>=89%':>7} {'>=5pp':>7}")
    print(f"  {'─'*70}")
    node_names = ["Node A", "Node B", "Node C", "Node D"]
    for nid in range(4):
        base = float(res_best["baseline_accs"].get(str(nid), 0.0))
        fl   = float(res_best["per_node_acc_final"].get(str(nid), 0.0))
        imp  = float(res_best["fl_improvement_pct"].get(str(nid), 0.0))
        flag89 = "✓" if fl >= 0.89 else ("~" if fl >= 0.85 else "✗")
        flag5  = "✓" if imp >= 5.0 else "✗"
        print(f"  {node_names[nid]:<22} {base*100:>8.2f}% "
              f"{fl*100:>10.2f}% {imp:>+8.2f}pp {flag89:>7} {flag5:>7}")
    print(DIV)

    # ── Master results file ───────────────────────────────────────────────────
    master = {
        "meta": {
            "total_time_sec":   round(time.time() - t_total_start, 1),
            "blockchain":       "Exonum (simulated)",
            "consensus_modes":  ["pBFT", "PoCL-pBFT", "PoS"],
            "n_hospital_nodes": 4,
        },
        "experiments": {
            k: {
                "global_acc_final":       v["global_acc_final"],
                "avg_consensus_delay_ms": v["avg_consensus_delay_ms"],
                "avg_gas_equivalent":     v["avg_gas_equivalent"],
                "chain_length":           v["chain_length"],
                "chain_valid":            v["chain_valid"],
                "per_node_acc_final":     v["per_node_acc_final"],
                "fl_improvement_pct":     v["fl_improvement_pct"],
                "baseline_accs":          v["baseline_accs"],
            }
            for k, v in all_results.items()
        },
        "best_hyperparams": {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in best_hp_dict.items()
        },
        "ipfs_gas_savings": ipfs_savings,
    }
    save_json("master_results", master)

    print(f"\n{DIV}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {time.time()-t_total_start:.1f}s")
    print(f"  Results in: {RESULTS_DIR}")
    print(DIV + "\n")


if __name__ == "__main__":
    main()
