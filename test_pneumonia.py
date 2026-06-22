"""
test_pneumonia.py
==================
Quick sanity check before running the full experiment.

    cd C:\\Users\\SASINI\\Desktop\\research\\flobc
    python test_pneumonia.py
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("  FLoBC Pneumonia — Pre-flight Check")
print("="*60)

# 1. Pillow
try:
    from PIL import Image
    print("  ✓ Pillow available")
except ImportError:
    print("  ✗ Pillow missing  →  pip install Pillow")
    sys.exit(1)

# 2. NumPy
try:
    import numpy as np
    print("  ✓ NumPy available")
except ImportError:
    print("  ✗ NumPy missing  →  pip install numpy")
    sys.exit(1)

# 3. Data root
from core.pneumonia_loader import DATA_ROOT, HOSPITAL_NODES
if not os.path.isdir(DATA_ROOT):
    print(f"  ✗ DATA_ROOT not found: {DATA_ROOT}")
    sys.exit(1)
print(f"  ✓ DATA_ROOT: {DATA_ROOT}")

# 4. Each hospital folder — check both NORMAL and PNEUMONIA
print()
all_ok = True
for nid, cfg in HOSPITAL_NODES.items():
    node_dir   = os.path.join(DATA_ROOT, cfg["folder"])
    normal_dir = os.path.join(node_dir, cfg["normal_dir"])
    pneumo_dir = os.path.join(node_dir, cfg["pneumo_dir"])

    n_imgs = (glob.glob(os.path.join(normal_dir, "*.jpeg")) +
              glob.glob(os.path.join(normal_dir, "*.jpg")) +
              glob.glob(os.path.join(normal_dir, "*.png")))

    p_imgs = (glob.glob(os.path.join(pneumo_dir, "*.jpeg")) +
              glob.glob(os.path.join(pneumo_dir, "*.jpg")) +
              glob.glob(os.path.join(pneumo_dir, "*.png")))

    ok_n = os.path.isdir(normal_dir) and len(n_imgs) > 0
    ok_p = os.path.isdir(pneumo_dir) and len(p_imgs) > 0

    status = "✓" if (ok_n and ok_p) else "✗"
    print(f"  {status} Node {nid} [{cfg['name']}]")
    print(f"       NORMAL    ({cfg['normal_dir']}): {len(n_imgs)} images  "
          f"{'✓' if ok_n else '✗ MISSING'}")
    print(f"       PNEUMONIA ({cfg['pneumo_dir']}): {len(p_imgs)} images  "
          f"{'✓' if ok_p else '✗ MISSING'}")

    if not (ok_n and ok_p):
        all_ok = False

if not all_ok:
    print("\n  ✗ Some folders are missing or empty. Fix them before continuing.")
    sys.exit(1)

# 5. Trial load — 30 images per class from Node A only
print()
print("  Loading Node A (30 images per class — trial) ...")

from PIL import Image as PILImage
from pathlib import Path

def _load_small(folder, label, max_n=30):
    imgs, labels = [], []
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in Path(folder).iterdir()
                    if p.suffix.lower() in exts])[:max_n]
    for p in paths:
        try:
            arr = np.array(
                PILImage.open(str(p)).convert("L").resize((64, 64)),
                dtype=np.float32) / 255.0
            imgs.append(arr.ravel())
            labels.append(label)
        except Exception:
            pass
    return imgs, labels

cfg_a      = HOSPITAL_NODES["A"]
normal_dir = os.path.join(DATA_ROOT, cfg_a["folder"], cfg_a["normal_dir"])
pneumo_dir = os.path.join(DATA_ROOT, cfg_a["folder"], cfg_a["pneumo_dir"])

n_imgs, n_lbl = _load_small(normal_dir, 0, 30)
p_imgs, p_lbl = _load_small(pneumo_dir, 1, 30)

X = np.array(n_imgs + p_imgs, dtype=np.float32)
y = np.array(n_lbl  + p_lbl,  dtype=np.int32)

print(f"  ✓ X shape  : {X.shape}   (should be (60, 4096))")
print(f"  ✓ y shape  : {y.shape}")
print(f"  ✓ NORMAL   : {(y==0).sum()}")
print(f"  ✓ PNEUMONIA: {(y==1).sum()}")
print(f"  ✓ Range    : [{X.min():.3f}, {X.max():.3f}]")

# 6. Quick model smoke test
from core.flobc_pneumonia_engine import PneumoniaModel
m   = PneumoniaModel(X.shape[1])
acc = m.accuracy(X, y)
print(f"  ✓ Untrained model accuracy: {acc:.4f}  (expect ~0.50)")

print()
print("="*60)
print("  All checks passed!")
print("  Run in order:")
print("    1.  python train_local_nodes.py")
print("    2.  python run_pneumonia.py")
print("    3.  python dashboard/plot_pneumonia.py")
print("="*60 + "\n")
