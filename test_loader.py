"""
test_loader.py
==============
Quick sanity check — confirms the pneumonia images load correctly
before running the full experiment.

Run:
    cd C:\\Users\\SASINI\\Desktop\\research\\flobc
    python test_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*55)
print("  FLoBC Pneumonia Loader — Sanity Check")
print("="*55)

# 1. Check Pillow is available
try:
    from PIL import Image
    print("  ✓ Pillow available")
except ImportError:
    print("  ✗ Pillow NOT found. Install with:   pip install Pillow")
    sys.exit(1)

# 2. Check data path exists
from core.pneumonia_loader import DATA_ROOT, HOSPITAL_NODES
if not os.path.isdir(DATA_ROOT):
    print(f"  ✗ DATA_ROOT not found: {DATA_ROOT}")
    print("    Edit DATA_ROOT in core/pneumonia_loader.py to point to your data folder.")
    sys.exit(1)
print(f"  ✓ DATA_ROOT found: {DATA_ROOT}")

# 3. Check each node folder
import glob
all_ok = True
for node_id, cfg in HOSPITAL_NODES.items():
    pneu_dir = os.path.join(DATA_ROOT, cfg["folder"], "PNEUMONIA")
    if not os.path.isdir(pneu_dir):
        print(f"  ✗ Node {node_id} ({cfg['name']}): PNEUMONIA folder missing at {pneu_dir}")
        all_ok = False
        continue
    imgs = glob.glob(os.path.join(pneu_dir, "*.jpeg")) + \
           glob.glob(os.path.join(pneu_dir, "*.jpg"))
    print(f"  ✓ Node {node_id} ({cfg['name']}): {len(imgs)} JPEG images found")

if not all_ok:
    sys.exit(1)

# 4. Load one node as a trial
print("\n  Loading Node A as trial (first 50 images only)...")
from core.pneumonia_loader import MAX_PER_NODE
import core.pneumonia_loader as pl
_orig = pl.MAX_PER_NODE
pl.MAX_PER_NODE = 50  # use small number for speed

X, y = pl.load_node_data("A")
pl.MAX_PER_NODE = _orig  # restore

import numpy as np
print(f"  ✓ Loaded X shape : {X.shape}")
print(f"  ✓ Labels shape   : {y.shape}")
print(f"  ✓ PNEUMONIA (1)  : {y.sum()}")
print(f"  ✓ NORMAL (0)     : {(y==0).sum()}")
print(f"  ✓ Value range    : [{X.min():.3f}, {X.max():.3f}]")

print("\n" + "="*55)
print("  All checks passed! Ready to run:")
print("    python run_pneumonia.py")
print("="*55 + "\n")
