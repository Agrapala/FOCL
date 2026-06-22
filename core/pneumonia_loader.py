"""
pneumonia_loader.py  —  REAL DATA ONLY (no synthesis)
======================================================
Loads REAL chest X-ray images from both classes for all 4 hospital nodes.

Folder structure (in your project):
  flobc/data/Node_A/NORMAL_NODE_A/    ← real NORMAL X-rays  (label = 0)
  flobc/data/Node_A/PNEUMONIA/        ← real PNEUMONIA X-rays (label = 1)
  flobc/data/Node_B/NORMAL_NODE_B/
  flobc/data/Node_B/PNEUMONIA/
  flobc/data/Node_C/NORMAL_NODE_C/
  flobc/data/Node_C/PNEUMONIA/
  flobc/data/Node_D/NORMAL_NODE_D/
  flobc/data/Node_D/PNEUMONIA/

Label encoding:
  0 = NORMAL    (real X-ray images from NORMAL_NODE_X folder)
  1 = PNEUMONIA (real X-ray images from PNEUMONIA folder)

Output:
  X  shape (N, IMG_SIZE * IMG_SIZE)  float32  values in [0, 1]
  y  shape (N,)                      int32    0 or 1
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE = 64      # 64×64 pixels → 4096 features per image

# Root of YOUR flobc project data folder
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Each node: normal folder name + PNEUMONIA folder name
HOSPITAL_NODES: Dict[str, Dict] = {
    "A": {
        "name":        "Hospital_Galle",
        "normal_dir":  "NORMAL_NODE_A",
        "pneumo_dir":  "PNEUMONIA",
        "folder":      "Node_A",
        "address":     "0xAAA0000000000000000000000000000000000001",
    },
    "B": {
        "name":        "Hospital_Colombo",
        "normal_dir":  "NORMAL_NODE_B",
        "pneumo_dir":  "PNEUMONIA",
        "folder":      "Node_B",
        "address":     "0xBBB0000000000000000000000000000000000002",
    },
    "C": {
        "name":        "Hospital_Kandy",
        "normal_dir":  "NORMAL_NODE_C",
        "pneumo_dir":  "PNEUMONIA",
        "folder":      "Node_C",
        "address":     "0xCCC0000000000000000000000000000000000003",
    },
    "D": {
        "name":        "Hospital_Jaffna",
        "normal_dir":  "NORMAL_NODE_D",
        "pneumo_dir":  "PNEUMONIA",
        "folder":      "Node_D",
        "address":     "0xDDD0000000000000000000000000000000000004",
    },
}


# ── Image loader ───────────────────────────────────────────────────────────

def _load_folder(folder_path: str, label: int) -> Tuple[list, list]:
    """
    Load all JPEG/JPG/PNG images from folder_path.
    Returns (flat_arrays_list, labels_list).
    Each image is resized to IMG_SIZE×IMG_SIZE, converted to grayscale,
    normalised to [0,1], and flattened to a 1-D vector.
    """
    from PIL import Image

    imgs, labels = [], []
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in Path(folder_path).iterdir()
                    if p.suffix.lower() in exts])

    for p in paths:
        try:
            arr = np.array(
                Image.open(str(p)).convert("L").resize((IMG_SIZE, IMG_SIZE)),
                dtype=np.float32
            ) / 255.0
            imgs.append(arr.ravel())
            labels.append(label)
        except Exception:
            pass   # skip corrupted files silently

    return imgs, labels


# ── Public API ─────────────────────────────────────────────────────────────

def load_node_data(node_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ALL real NORMAL + PNEUMONIA images for one hospital node.

    Returns
    -------
    X : np.ndarray  shape (N, IMG_SIZE²)  float32
    y : np.ndarray  shape (N,)            int32   0=NORMAL 1=PNEUMONIA
    """
    cfg        = HOSPITAL_NODES[node_id]
    node_dir   = os.path.join(DATA_ROOT, cfg["folder"])
    normal_dir = os.path.join(node_dir, cfg["normal_dir"])
    pneumo_dir = os.path.join(node_dir, cfg["pneumo_dir"])

    if not os.path.isdir(normal_dir):
        raise FileNotFoundError(f"NORMAL folder missing: {normal_dir}")
    if not os.path.isdir(pneumo_dir):
        raise FileNotFoundError(f"PNEUMONIA folder missing: {pneumo_dir}")

    n_imgs, n_labels = _load_folder(normal_dir, label=0)
    p_imgs, p_labels = _load_folder(pneumo_dir, label=1)

    all_X = np.array(n_imgs + p_imgs, dtype=np.float32)
    all_y = np.array(n_labels + p_labels, dtype=np.int32)

    # shuffle so classes are not in a block
    idx = np.random.permutation(len(all_X))
    all_X, all_y = all_X[idx], all_y[idx]

    print(f"  Node {node_id} [{cfg['name']}]  "
          f"NORMAL={len(n_imgs)}  PNEUMONIA={len(p_imgs)}  "
          f"total={len(all_X)}")
    return all_X, all_y


def load_all_nodes() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load real data for all 4 hospital nodes. Returns {node_id: (X, y)}."""
    print("\n" + "=" * 62)
    print("  Loading REAL X-ray Data  —  4 Hospital Nodes")
    print("  NORMAL images  : NORMAL_NODE_X folders (real X-rays)")
    print("  PNEUMONIA images: PNEUMONIA folders     (real X-rays)")
    print("  NO synthetic data is used anywhere.")
    print("=" * 62)
    data = {}
    for nid in ["A", "B", "C", "D"]:
        X, y = load_node_data(nid)
        data[nid] = (X, y)
    print("=" * 62 + "\n")
    return data


def build_splits(
    node_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_ratio:  float = 0.15,
    test_ratio: float = 0.10,
    seed:       int   = 42,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Split each node's data into train / val / test.

    Strategy
    --------
    • Each hospital keeps its own TRAIN slice — never shared with others.
      This is the core FL privacy guarantee.
    • Validation slices from all nodes are pooled into one shared val set.
      These are used by the BC validators to judge model updates.
    • Test slices from all nodes are pooled into one final test set.
      The global model is evaluated on this after each FL round.
    • Each hospital's OWN test slice is also kept separately in
      `per_node_test` so per-institution "before vs after federation"
      accuracy can be measured on data that hospital never trained on.

    Returns
    -------
    per_node_train : { node_id : (X_train, y_train) }
    X_val, y_val   : pooled validation numpy arrays
    X_test, y_test : pooled test numpy arrays
    per_node_test  : { node_id : (X_test_i, y_test_i) }
    """
    np.random.seed(seed)
    per_node_train = {}
    per_node_test:  Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    val_Xs,  val_ys  = [], []
    test_Xs, test_ys = [], []

    for nid, (X, y) in node_data.items():
        n    = len(X)
        perm = np.random.permutation(n)

        n_test = max(2, int(n * test_ratio))
        n_val  = max(2, int(n * val_ratio))

        test_idx = perm[:n_test]
        val_idx  = perm[n_test : n_test + n_val]
        tr_idx   = perm[n_test + n_val :]

        per_node_train[nid] = (X[tr_idx], y[tr_idx])
        per_node_test[nid]  = (X[test_idx], y[test_idx])
        val_Xs.append(X[val_idx]);   val_ys.append(y[val_idx])
        test_Xs.append(X[test_idx]); test_ys.append(y[test_idx])

    X_val  = np.vstack(val_Xs);   y_val  = np.hstack(val_ys)
    X_test = np.vstack(test_Xs);  y_test = np.hstack(test_ys)

    # shuffle the pooled sets
    for Xp, yp in [(X_val, y_val), (X_test, y_test)]:
        p = np.random.permutation(len(Xp))
        Xp[:] = Xp[p];  yp[:] = yp[p]

    return per_node_train, X_val, y_val, X_test, y_test, per_node_test


def print_summary(per_node, X_val, y_val, X_test, y_test):
    print("\n" + "─" * 62)
    print("  DATASET SUMMARY")
    print("─" * 62)
    for nid, (X, y) in per_node.items():
        name = HOSPITAL_NODES[nid]["name"]
        n    = HOSPITAL_NODES[nid]["normal_dir"]
        p    = "PNEUMONIA"
        print(f"  Node {nid} ({name}):")
        print(f"    Train samples : {len(X)}")
        print(f"    Normal (0)    : {(y==0).sum()}")
        print(f"    Pneumonia (1) : {(y==1).sum()}")
    print(f"\n  Validation set (pooled) : {len(X_val)} samples")
    print(f"    Normal    : {(y_val==0).sum()}")
    print(f"    Pneumonia : {(y_val==1).sum()}")
    print(f"\n  Test set (pooled)       : {len(X_test)} samples")
    print(f"    Normal    : {(y_test==0).sum()}")
    print(f"    Pneumonia : {(y_test==1).sum()}")
    print(f"\n  Feature dimension : {X_val.shape[1]}  "
          f"(= {IMG_SIZE}×{IMG_SIZE} flattened pixels)")
    print("─" * 62 + "\n")
