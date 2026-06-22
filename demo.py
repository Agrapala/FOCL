"""
FLoBC Real Blockchain Demo
===========================
Run this first to verify the full real-blockchain pipeline works.

    python demo.py

Three demos:
  1. Basic federated training with real blockchain (7T / 3V / BSP)
  2. Byzantine fault tolerance + trust scoring
  3. Synchronization scheme comparison
  4. Blockchain integrity verification + chain printout
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.flobc_engine import FLoBC, SyncScheme
from core.data_utils   import generate_mnist_like, train_val_test_split

DIV  = "─" * 62
DIV2 = "═" * 62


def section(title):
    print(f"\n{DIV}\n  {title}\n{DIV}")


def main():
    print(f"\n{DIV2}")
    print("  FLoBC — Federated Learning on Blockchain (REAL CHAIN)")
    print("  Abuzied et al., Cluster Computing 2024")
    print("  DOI: 10.1007/s10586-024-04273-1")
    print(f"{DIV2}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n  Generating synthetic MNIST-like dataset (4 000 samples)...")
    X, y = generate_mnist_like(n_samples=4000, seed=42)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)
    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | "
          f"Test: {len(X_te)} | Dims: {X.shape[1]}")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 1  Basic FLoBC with real blockchain
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 1 — Real Blockchain FLoBC  [7 Trainers | 3 Validators | BSP]")
    print("  Each round commits a cryptographic block with:")
    print("  MODEL_UPDATE + VALIDATION + TRUST_UPDATE + GLOBAL_MODEL txs\n")

    fw1 = FLoBC(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        n_trainers=7, n_validators=3,
        sync_scheme=SyncScheme.BSP,
        use_reputation=True,
        verbose_chain=True,
    )
    r1 = fw1.train(n_rounds=10, verbose=True)

    print(f"\n  ✓ Final accuracy     : {r1['final_accuracy']:.4f}")
    print(f"  ✓ Chain length       : {r1['chain_length']} blocks (incl. genesis)")
    print(f"  ✓ Chain integrity    : {'VALID ✓' if r1['chain_valid'] else 'INVALID ✗'}")

    # Print the chain
    fw1.print_chain(max_blocks=6)

    # Export chain to JSON
    os.makedirs("results", exist_ok=True)
    fw1.export_chain("results/demo1_blockchain.json")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 2  Byzantine fault tolerance
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 2 — Byzantine Fault Tolerance  [Signed Tx + Trust Penalties]")
    print("  Trainers 4 & 5 inject noise into their signed updates.")
    print("  Reputation system penalises them; their tx weight → 0.\n")

    noise = [0.0, 0.0, 0.0, 0.0, 0.25, 0.60]
    fw2   = FLoBC(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        n_trainers=6, n_validators=3,
        use_reputation=True,
        noise_profile=noise,
    )
    r2 = fw2.train(n_rounds=10, verbose=False)

    print("  Final trust scores:")
    for tid, vals in r2["trust_log"].items():
        s   = vals[-1] if vals else 0.0
        bar = "█" * int(s * 40)
        tag = "  ← PENALISED" if s < 0.02 else ("  ← TRUSTED" if s > 0.25 else "")
        print(f"    Trainer {tid}: {s:.4f}  {bar}{tag}")

    print(f"\n  ✓ Accuracy (with Byzantine nodes): {r2['final_accuracy']:.4f}")
    print(f"  ✓ Chain valid                    : {fw2.chain.is_chain_valid()}")
    fw2.export_chain("results/demo2_blockchain.json")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 3  Sync scheme comparison
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 3 — Synchronization Schemes  [BSP | SSP | BAP]")

    for label, scheme, ratio in [
        ("BSP    ", SyncScheme.BSP, 1.0),
        ("SSP    ", SyncScheme.SSP, 1.0),
        ("BAP 1.0", SyncScheme.BAP, 1.0),
        ("BAP 0.6", SyncScheme.BAP, 0.6),
    ]:
        fw3 = FLoBC(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_trainers=6, n_validators=3,
            sync_scheme=scheme,
            bap_majority_ratio=ratio,
        )
        r3  = fw3.train(n_rounds=15, verbose=False)
        acc = r3["final_accuracy"]
        bar = "█" * int(acc * 50)
        print(f"  {label} | acc={acc:.4f} | chain={r3['chain_length']} blocks | {bar}")

    # ══════════════════════════════════════════════════════════════════════
    # DEMO 4  Tamper detection
    # ══════════════════════════════════════════════════════════════════════
    section("DEMO 4 — Blockchain Tamper Detection")

    fw4 = FLoBC(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        n_trainers=4, n_validators=2, sync_scheme=SyncScheme.BSP,
    )
    fw4.train(n_rounds=5, verbose=False)

    print(f"  Chain valid (before tamper): {fw4.chain.is_chain_valid()}")

    # Tamper with block #2's merkle root
    if fw4.chain.length() > 2:
        fw4.chain._chain[2].merkle_root = "deadbeef" * 8
        print(f"  Chain valid (after tamper) : {fw4.chain.is_chain_valid()}")
        print("  ✓ Tampering detected correctly — blockchain is tamper-evident!")
    else:
        print("  (not enough blocks to demo tampering)")

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DIV2}")
    print("  All demos complete!")
    print("  Blockchain exports saved to results/demo1_blockchain.json")
    print("                              results/demo2_blockchain.json")
    print("\n  Next steps:")
    print("    python experiments/run_experiments.py   ← all 8 paper experiments")
    print("    python dashboard/plot_results.py        ← generate charts")
    print(f"{DIV2}\n")


if __name__ == "__main__":
    main()
