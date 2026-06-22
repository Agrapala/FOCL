"""
train_local_nodes.py  —  STAGE 1
==================================
Find the BEST local model for each hospital node independently,
BEFORE federated learning begins.

Each hospital trains on its own private X-ray data (NORMAL + PNEUMONIA)
for many epochs, with early stopping, and saves the best weights.
These best weights are used to WARM-START the FL pipeline in Stage 2,
so the global model starts from a much better position than random.

Process per node
----------------
1. Load real NORMAL + PNEUMONIA images from the node's local folder.
2. Split into local train (75%) / local val (25%) — stays private.
3. Train MLP for up to MAX_EPOCHS with mini-batch SGD.
4. Track validation accuracy each epoch — keep the BEST weights seen.
5. Early stop if no improvement for PATIENCE epochs.
6. Save best weights to  results/local_best_<node_id>.npy
7. Print final local train accuracy + best local val accuracy.

Run:
    cd C:\\Users\\SASINI\\Desktop\\research\\flobc
    python train_local_nodes.py
"""

import os
import sys
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.pneumonia_loader       import load_all_nodes, HOSPITAL_NODES, build_splits
from core.flobc_pneumonia_engine import PneumoniaModel

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

# ── Hyper-parameters for local pre-training ───────────────────────────────
MAX_EPOCHS  = 60       # maximum epochs per node
PATIENCE    = 8        # early-stop patience
BATCH_SIZE  = 32
LR          = 0.01     # learning rate
LOCAL_VAL   = 0.20     # 20% of each node's data held for local validation
HIDDEN_DIM  = 256      # MLP hidden layer size

NID_NAMES = {nid: cfg["name"] for nid, cfg in HOSPITAL_NODES.items()}
NID_TO_TID = {"A": 0, "B": 1, "C": 2, "D": 3}

DIV = "=" * 64


def train_one_node(node_id: str,
                   X: np.ndarray,
                   y: np.ndarray) -> dict:
    """
    Train a local MLP for one hospital node with early stopping.

    Returns dict with best_weights, best_val_acc, final_train_acc, epochs_run.
    """
    name = NID_NAMES[node_id]
    n    = len(X)

    # ── Local train / val split (private — never shared) ──────────────────
    perm  = np.random.permutation(n)
    n_val = max(4, int(n * LOCAL_VAL))
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]
    X_tr,  y_tr  = X[perm[n_val:]], y[perm[n_val:]]

    print(f"\n  Node {node_id}  [{name}]")
    print(f"    Local train : {len(X_tr)} samples  "
          f"(normal={( y_tr==0).sum()}  pneum={(y_tr==1).sum()})")
    print(f"    Local val   : {len(X_val)} samples  "
          f"(normal={(y_val==0).sum()}  pneum={(y_val==1).sum()})")

    # ── Build model ────────────────────────────────────────────────────────
    input_dim = X_tr.shape[1]
    model     = PneumoniaModel(input_dim, HIDDEN_DIM, output_dim=2)

    best_val_acc    = 0.0
    best_weights    = model.flatten().copy()
    patience_count  = 0
    epochs_run      = 0
    history         = []

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        perm_e = np.random.permutation(len(X_tr))
        X_sh   = X_tr[perm_e]
        y_sh   = y_tr[perm_e]

        for start in range(0, len(X_tr), BATCH_SIZE):
            Xb = X_sh[start : start + BATCH_SIZE]
            yb = y_sh[start : start + BATCH_SIZE]
            if len(Xb) == 0:
                continue
            model.sgd_step(Xb, yb, lr=LR)

        val_acc   = model.accuracy(X_val, y_val)
        train_acc = model.accuracy(X_tr,  y_tr)
        history.append({"epoch": epoch, "train": train_acc, "val": val_acc})

        if val_acc > best_val_acc + 1e-4:
            best_val_acc   = val_acc
            best_weights   = model.flatten().copy()
            patience_count = 0
            marker = " ← best"
        else:
            patience_count += 1
            marker = ""

        # Progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or patience_count >= PATIENCE:
            print(f"    Epoch {epoch:3d}/{MAX_EPOCHS}  "
                  f"train={train_acc:.4f}  val={val_acc:.4f}{marker}")

        if patience_count >= PATIENCE:
            print(f"    Early stop at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    # ── Restore best and get final training accuracy ───────────────────────
    model.unflatten(best_weights)
    final_train_acc = model.accuracy(X_tr, y_tr)

    print(f"    ✓ Best local val accuracy : {best_val_acc:.4f}")
    print(f"    ✓ Final train accuracy    : {final_train_acc:.4f}")
    print(f"    ✓ Epochs run              : {epochs_run}")

    return {
        "node_id":        node_id,
        "hospital_name":  name,
        "best_weights":   best_weights,
        "best_val_acc":   best_val_acc,
        "final_train_acc": final_train_acc,
        "epochs_run":     epochs_run,
        "history":        history,
        "n_train":        len(X_tr),
        "n_val":          len(X_val),
        "input_dim":      input_dim,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    print(f"\n{DIV}")
    print("  STAGE 1  —  Local Pre-Training at Each Hospital Node")
    print("  Each hospital finds its BEST local model independently.")
    print("  Best weights will warm-start the FL pipeline in Stage 2.")
    print(DIV)

    # ── Load all real images ──────────────────────────────────────────────
    node_data = load_all_nodes()

    # ── Train each node ───────────────────────────────────────────────────
    t0      = time.time()
    results = {}
    local_weights: dict = {}   # { node_id: np.ndarray }

    print(f"\n{DIV}")
    print("  LOCAL TRAINING")
    print(DIV)

    for node_id in ["A", "B", "C", "D"]:
        X, y = node_data[node_id]
        res  = train_one_node(node_id, X, y)
        results[node_id] = res

        # Save best weights to disk
        save_path = os.path.join(ROOT, "results", f"local_best_{node_id}.npy")
        np.save(save_path, res["best_weights"])
        local_weights[node_id] = res["best_weights"]
        print(f"    Saved → {save_path}")

    elapsed = round(time.time() - t0, 1)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  STAGE 1 SUMMARY")
    print(DIV)
    print(f"  {'Node':<6}  {'Hospital':<20}  {'Train Acc':>10}  "
          f"{'Val Acc':>9}  {'Epochs':>7}  {'Samples':>8}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*10}  {'─'*9}  {'─'*7}  {'─'*8}")

    for nid in ["A", "B", "C", "D"]:
        r = results[nid]
        print(f"  {nid:<6}  {r['hospital_name']:<20}  "
              f"{r['final_train_acc']:>10.4f}  "
              f"{r['best_val_acc']:>9.4f}  "
              f"{r['epochs_run']:>7}  "
              f"{r['n_train']:>8}")

    print(f"\n  Total Stage 1 time : {elapsed}s")
    print(f"\n  Next step → run Stage 2:")
    print(f"    python run_pneumonia.py")
    print(DIV + "\n")

    # ── Save metadata ─────────────────────────────────────────────────────
    meta = {}
    for nid, r in results.items():
        meta[nid] = {
            "hospital_name":   r["hospital_name"],
            "best_val_acc":    round(r["best_val_acc"], 4),
            "final_train_acc": round(r["final_train_acc"], 4),
            "epochs_run":      r["epochs_run"],
            "n_train":         r["n_train"],
            "n_val":           r["n_val"],
            "input_dim":       r["input_dim"],
            "weights_file":    f"results/local_best_{nid}.npy",
            "history":         r["history"],
        }

    meta_path = os.path.join(ROOT, "results", "local_training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")


if __name__ == "__main__":
    main()
