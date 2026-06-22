"""
run_network.py  —  FLoBC Real Network Orchestrator
====================================================
Starts ALL components in the correct order and runs the full FL training
using real HTTP communication between trainers and validators.

Component startup sequence:
  1. Rust pBFT blockchain node   (subprocess, port 8100)
  2. Flask validator HTTP servers (threads, ports 7000-7002)
  3. Wire API server              (thread, port 8080)
  4. FL training via HTTP         (main thread)

Usage:
    python network/run_network.py

Prerequisites:
  pip install flask requests flask-cors

For the Rust node (optional but recommended):
  cd rust_blockchain && cargo build --release
  The binary will be at: rust_blockchain/target/release/flobc-blockchain.exe

If Rust is not built, the system falls back to the Python blockchain
while still using real HTTP for trainer-validator communication.

Open in browser while running:
    dashboard/client.html   (JavaScript dashboard)
"""

import sys, os, time, subprocess, signal, json, threading
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.pneumonia_loader  import load_all_nodes, build_splits, HOSPITAL_NODES
from network.http_engine    import HTTPFloBCEngine

HOSP         = {nid: cfg["name"] for nid, cfg in HOSPITAL_NODES.items()}
RUST_BINARY  = os.path.join(ROOT, "rust_blockchain", "target", "release",
                             "flobc-blockchain.exe")
RUST_URL     = "http://127.0.0.1:8100"

DOWNSAMPLE   = 2      # 64x64 -> 32x32
MAX_TRAIN    = 500
FL_ROUNDS    = 20
LOCAL_EPOCHS = 5
BATCH        = 512
LR           = 0.008

DIV = "=" * 68


def _ds(X):
    return X.reshape(-1, 64, 64)[:, ::DOWNSAMPLE, ::DOWNSAMPLE] \
             .reshape(len(X), -1).astype(np.float32)


def start_rust_node():
    """
    Launch the Rust pBFT blockchain node as a subprocess.
    Returns the process or None if binary not found.
    """
    if not os.path.exists(RUST_BINARY):
        alt = RUST_BINARY.replace(".exe", "")   # Linux/Mac
        if not os.path.exists(alt):
            print(f"  [Rust] Binary not found at {RUST_BINARY}")
            print(f"  [Rust] Build it with:")
            print(f"         cd rust_blockchain && cargo build --release")
            print(f"  [Rust] Continuing without Rust node (Python blockchain active)")
            return None
        binary = alt
    else:
        binary = RUST_BINARY

    print(f"  [Rust] Starting pBFT blockchain node ...")
    proc = subprocess.Popen(
        [binary],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for it to start (max 5s)
    import requests
    for _ in range(10):
        time.sleep(0.5)
        try:
            r = requests.get(f"{RUST_URL}/health", timeout=0.5)
            if r.status_code == 200:
                print(f"  [Rust] pBFT node online at {RUST_URL}")
                return proc
        except Exception:
            pass

    print(f"  [Rust] Node did not start in time — continuing without it")
    proc.terminate()
    return None


def check_dependencies():
    missing = []
    for pkg in ["flask", "requests", "flask_cors"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with:  pip install {' '.join(missing)}")
        print()


def main():
    np.random.seed(42)
    t_total = time.time()

    print(f"\n{DIV}")
    print("  FLoBC  —  Real Network Mode")
    print("  Rust pBFT blockchain  |  Flask HTTP API  |  ECDSA secp256k1")
    print(f"  FL Rounds: {FL_ROUNDS} | Local Epochs: {LOCAL_EPOCHS} | "
          f"Batch: {BATCH} | 32x32 features")
    print(DIV)

    check_dependencies()

    # ── 1. Start Rust pBFT blockchain node ────────────────────────────────
    print("\n  Step 1: Rust pBFT Blockchain Node")
    rust_proc = start_rust_node()

    # ── 2. Load + preprocess data ─────────────────────────────────────────
    print("\n  Step 2: Loading real X-ray data ...")
    node_data = load_all_nodes()
    per_node_train, X_val, y_val, X_test, y_test, _ = build_splits(
        node_data, val_ratio=0.15, test_ratio=0.10, seed=42)

    rng = np.random.default_rng(42)
    for nid in per_node_train:
        X_tr, y_tr = per_node_train[nid]
        if len(X_tr) > MAX_TRAIN:
            idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        per_node_train[nid] = (_ds(X_tr), y_tr)
    X_val, X_test = _ds(X_val), _ds(X_test)

    print(f"  Feature dim : {X_val.shape[1]}  (32x32 downsampled)")
    print(f"  Val samples : {len(X_val)} | Test: {len(X_test)}")

    # ── 3. Create HTTP engine + start servers ──────────────────────────────
    print("\n  Step 3: Starting Flask HTTP servers ...")
    engine = HTTPFloBCEngine(
        per_node_train=per_node_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        n_validators=3,
        lr=LR, batch_size=BATCH, local_epochs=LOCAL_EPOCHS,
        verbose=True,
    )
    engine.start_servers()

    # ── 4. Print dashboard instructions ────────────────────────────────────
    print(f"\n{DIV}")
    print(f"  SERVICES RUNNING:")
    print(f"    Rust pBFT node  : {RUST_URL}{'  (active)' if rust_proc else '  (offline - Python fallback)'}")
    print(f"    Validator HTTP  : http://127.0.0.1:7000, :7001, :7002")
    print(f"    Wire API        : http://127.0.0.1:8080")
    print(f"")
    print(f"  OPEN DASHBOARD:")
    print(f"    File -> Open -> dashboard/client.html")
    print(f"    (Live updates every 3 seconds from Wire API)")
    print(f"")
    print(f"  WIRE API ENDPOINTS:")
    print(f"    GET http://127.0.0.1:8080/status")
    print(f"    GET http://127.0.0.1:8080/chain")
    print(f"    GET http://127.0.0.1:8080/trust")
    print(f"    GET http://127.0.0.1:8080/accuracy")
    print(f"    GET http://127.0.0.1:8080/nodes")
    print(f"")
    print(f"  RUST NODE ENDPOINTS:")
    print(f"    GET  http://127.0.0.1:8100/stats")
    print(f"    GET  http://127.0.0.1:8100/chain")
    print(f"    POST http://127.0.0.1:8100/block/propose")
    print(DIV)
    print()

    # ── 5. Run FL training (HTTP mode) ────────────────────────────────────
    print(f"  Step 4: FL Training with HTTP Communication ...")
    results = engine.train(n_rounds=FL_ROUNDS)

    # ── 6. Save results ────────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = {
        "mode":          "HTTP network",
        "fl_rounds":     FL_ROUNDS,
        "final_accuracy": round(results["final_accuracy"], 4),
        "accuracy_log":  [round(a, 4) for a in results["accuracy_log"]],
        "trust_log":     {str(k): [round(v, 4) for v in vs]
                          for k, vs in results["trust_log"].items()},
        "chain_valid":   results["chain_valid"],
        "rust_node":     rust_proc is not None,
        "crypto":        "ECDSA secp256k1",
        "consensus":     "pBFT PoS >2/3",
        "runtime_s":     round(time.time() - t_total, 1),
    }
    out_path = os.path.join(ROOT, "results", "network_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{DIV}")
    print(f"  NETWORK FL COMPLETE")
    print(f"  Final accuracy  : {results['final_accuracy']:.4f}")
    print(f"  Chain valid     : {results['chain_valid']}")
    print(f"  Rust node used  : {rust_proc is not None}")
    print(f"  Runtime         : {time.time()-t_total:.1f}s")
    print(f"  Results saved   : {out_path}")
    print(DIV)
    print()
    print("  Servers still running — Wire API and dashboard remain live.")
    print("  Press Ctrl+C to stop all servers.")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n  Shutting down ...")
        if rust_proc:
            rust_proc.terminate()


if __name__ == "__main__":
    main()
