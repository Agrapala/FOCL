"""
run_pneumonia.py  —  STAGE 2: All Experiments
==============================================
Run order:
  1. python train_local_nodes.py       Stage 1: best local models per hospital
  2. python run_pneumonia.py           Stage 2: all FL experiments (this file)
  3. python evaluate_objectives.py     Stage 3: verify research objectives
  4. python dashboard/plot_pneumonia.py         generate charts
"""

import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.pneumonia_loader       import (load_all_nodes, build_splits,
                                         print_summary, HOSPITAL_NODES)
from core.flobc_pneumonia_engine import FloBCPneumonia, SyncScheme

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

HOSP_NAMES = {nid: cfg["name"] for nid, cfg in HOSPITAL_NODES.items()}
NID_LIST   = ["A", "B", "C", "D"]
DIV        = "=" * 68


def load_local_weights():
    weights = {}
    for nid in NID_LIST:
        path = os.path.join(ROOT, "results", f"local_best_{nid}.npy")
        if os.path.exists(path):
            weights[nid] = np.load(path)
            print(f"  ✓ Node {nid} ({HOSP_NAMES[nid]}) warm-start loaded")
        else:
            print(f"  ⚠ Node {nid}: no Stage 1 weights — random init")
    return weights


def main():
    np.random.seed(42)

    print(f"\n{DIV}")
    print("  FLoBC × Pneumonia — Stage 2: Federated Learning + Blockchain")
    print("  4 Hospitals | Real X-ray Data | Blockchain Validated")
    print(DIV)

    node_data = load_all_nodes()
    per_node_train, X_val, y_val, X_test, y_test, per_node_test = build_splits(
        node_data, val_ratio=0.15, test_ratio=0.10, seed=42)
    print_summary(per_node_train, X_val, y_val, X_test, y_test)

    print(f"\n{DIV}")
    print("  Loading Stage 1 warm-start weights")
    print(DIV)
    local_weights = load_local_weights()

    # ══════════════════════════════════════════════════════════════════
    # EXP 1 — All 4 hospitals healthy | BSP | 30 rounds
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{DIV}")
    print("  EXPERIMENT 1 — All 4 Hospitals Healthy | BSP | 30 rounds")
    print(DIV)

    fw1 = FloBCPneumonia(
        per_node_train=per_node_train,
        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        hospital_names=HOSP_NAMES,
        sync_scheme=SyncScheme.BSP,
        n_validators=3,
        lr=0.008, batch_size=32, local_epochs=15,
        verbose=True, local_init_weights=local_weights,
    )
    t0   = time.time()
    res1 = fw1.train(n_rounds=30)
    t1   = round(time.time() - t0, 1)

    DIV2 = '─' * 68
    print(f"\n  {DIV2}")
    print("  EXPERIMENT 1 RESULTS")
    print(f"  {DIV2}")
    print(f"  Start accuracy  : {res1['accuracy_log'][0]:.4f}")
    print(f"  Final accuracy  : {res1['final_accuracy']:.4f}  "
          f"(+{res1['final_accuracy']-res1['accuracy_log'][0]:.4f})")
    print(f"  Chain length    : {res1['chain_length']} blocks")
    print(f"  Chain valid     : {res1['chain_valid']}")
    print(f"  Runtime         : {t1}s")
    print("  Final trust scores:")
    for tid, score in sorted(res1["final_trust"].items()):
        bar = "█" * int(score * 50)
        print(f"    Node {NID_LIST[tid]} ({HOSP_NAMES[NID_LIST[tid]]:<22}): "
              f"{score:.4f}  {bar}")

    fw1.export_chain(os.path.join(ROOT, "results", "blockchain_exp1_BSP.json"))
    fw1.print_chain(max_blocks=5)

    # ══════════════════════════════════════════════════════════════════
    # EXP 2 — Byzantine: Hospital Jaffna sends noisy updates
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{DIV}")
    print("  EXPERIMENT 2 — Byzantine: Hospital Jaffna (Node D) noisy")
    print(DIV)

    fw2 = FloBCPneumonia(
        per_node_train=per_node_train,
        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        hospital_names=HOSP_NAMES,
        sync_scheme=SyncScheme.BSP,
        n_validators=3,
        noise_profile={"D": 0.45},
        lr=0.008, batch_size=32, local_epochs=15,
        verbose=False, local_init_weights=local_weights,
    )
    res2 = fw2.train(n_rounds=30)

    print(f"  Final accuracy (with Byzantine node): {res2['final_accuracy']:.4f}")
    print(f"  Chain valid: {fw2.chain.is_chain_valid()}")
    print("  Final trust scores (Jaffna should be lowest):")
    for tid, score in sorted(res2["final_trust"].items()):
        nid = NID_LIST[tid]
        bar = "█" * int(score * 50)
        tag = "  ← PENALISED" if nid == "D" else ""
        print(f"    Node {nid} ({HOSP_NAMES[nid]:<22}): {score:.4f}  {bar}{tag}")
    fw2.export_chain(os.path.join(ROOT, "results", "blockchain_exp2_byzantine.json"))

    # ══════════════════════════════════════════════════════════════════
    # EXP 3 — Sync scheme comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{DIV}")
    print("  EXPERIMENT 3 — Sync Scheme Comparison | 20 rounds")
    print(DIV)

    sync_configs = [
        ("BSP",    SyncScheme.BSP, 1.0, 0.0),
        ("SSP",    SyncScheme.SSP, 1.0, 0.2),
        ("BAP100", SyncScheme.BAP, 1.0, 0.0),
        ("BAP60",  SyncScheme.BAP, 0.6, 0.0),
    ]
    sync_results = {}
    for label, scheme, bap_r, ssp_s in sync_configs:
        fw = FloBCPneumonia(
            per_node_train=per_node_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
            hospital_names=HOSP_NAMES,
            sync_scheme=scheme, n_validators=3,
            bap_majority_ratio=bap_r, ssp_slack_ratio=ssp_s,
            lr=0.008, batch_size=32, local_epochs=12,
            verbose=False, local_init_weights=local_weights,
        )
        res = fw.train(n_rounds=20)
        acc = res["final_accuracy"]
        bar = "█" * int(acc * 50)
        print(f"  {label:<8}  acc={acc:.4f}  blocks={res['chain_length']}  {bar}")
        sync_results[label] = {
            "final_accuracy": round(acc, 4),
            "chain_length":   res["chain_length"],
            "accuracy_log":   [round(a, 4) for a in res["accuracy_log"]],
        }

    # ══════════════════════════════════════════════════════════════════
    # TAMPER DETECTION DEMO
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{DIV}")
    print("  DEMO — Blockchain Tamper Detection")
    print(DIV)
    fw_t = FloBCPneumonia(
        per_node_train=per_node_train,
        X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        hospital_names=HOSP_NAMES, n_validators=2,
        verbose=False, local_init_weights=local_weights,
    )
    fw_t.train(n_rounds=5)
    print(f"  Chain valid (before): {fw_t.chain.is_chain_valid()}")
    if fw_t.chain.length() > 2:
        fw_t.chain._chain[2].merkle_root = "TAMPERED" * 5
        after = fw_t.chain.is_chain_valid()
        print(f"  Chain valid (after) : {after}")
        print(f"  {'✓ Tampering DETECTED!' if not after else '✗ Not detected'}")

    # ── Save results ───────────────────────────────────────────────────────
    summary = {
        "hospitals": HOSP_NAMES,
        "experiment_1_BSP": {
            "warmstart_acc":  round(res1["accuracy_log"][0], 4),
            "final_accuracy": round(res1["final_accuracy"], 4),
            "chain_length":   res1["chain_length"],
            "chain_valid":    res1["chain_valid"],
            "runtime_sec":    t1,
            "accuracy_log":   [round(a, 4) for a in res1["accuracy_log"]],
            "local_train_log":{k: [round(v,4) for v in vs]
                               for k,vs in res1["local_train_log"].items()},
            "local_val_log":  {k: [round(v,4) for v in vs]
                               for k,vs in res1["local_val_log"].items()},
            "trust_log":      {str(k): [round(v,4) for v in vs]
                               for k,vs in res1["trust_log"].items()},
            "final_trust":    {f"Node_{NID_LIST[k]}": round(v,4)
                               for k,v in res1["final_trust"].items()},
        },
        "experiment_2_byzantine": {
            "final_accuracy": round(res2["final_accuracy"], 4),
            "chain_valid":    fw2.chain.is_chain_valid(),
            "final_trust":    {f"Node_{NID_LIST[k]}_{HOSP_NAMES[NID_LIST[k]]}": round(v,4)
                               for k,v in res2["final_trust"].items()},
        },
        "experiment_3_sync": sync_results,
    }
    out = os.path.join(ROOT, "results", "pneumonia_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{DIV}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(DIV)
    print(f"  Exp 1 BSP   : {res1['accuracy_log'][0]:.4f} → {res1['final_accuracy']:.4f}")
    print(f"  Exp 2 Byz.  : {res2['final_accuracy']:.4f}")
    print(f"  Results → results/pneumonia_results.json")
    print(f"\n  Next:  python evaluate_objectives.py")
    print(f"  Then:  python dashboard/plot_pneumonia.py")
    print(DIV + "\n")


if __name__ == "__main__":
    main()
