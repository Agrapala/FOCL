"""
wire_api.py  —  FLoBC Wire API Server
=======================================
A standalone Flask server that acts as the read-only query interface
for all external consumers (JavaScript dashboard, monitoring tools, etc.).

This is the "Wire API" described in the paper's modular view (Fig. 1):
  "Validators also expose a Wire API for answering queries about the
   blockchain state (e.g., what's the latest model version?)"

The Wire API aggregates state from:
  - The Python FL engine (accuracy log, trust scores)
  - The Rust blockchain node  (chain, blocks, transactions)
  - The Python validator servers (node identities)

Runs on port 8080 (separate from the Rust node on 8100 and
Flask validators on 7000-7002).

Start with:
    python network/wire_api.py

Or called automatically by run_network.py.
"""

import sys, os, json, time, threading
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUST_BC_URL = "http://127.0.0.1:8100"
WIRE_PORT   = 8080


class WireAPIServer:
    """
    Aggregating Wire API server.
    State is injected by the FL engine after each round via update_*() methods,
    or fetched live from the Rust node via proxy endpoints.
    """

    def __init__(self, rust_bc_url: str = RUST_BC_URL, port: int = WIRE_PORT):
        self.rust_bc_url = rust_bc_url
        self.port        = port
        self._state      = {
            "accuracy_log":  [],
            "trust_scores":  {},
            "nodes":         [],
            "round":         0,
            "started_at":    time.time(),
        }

        try:
            from flask import Flask
            from flask_cors import CORS
            self.app      = Flask("wire_api")
            CORS(self.app)
            self._flask_ok = True
        except ImportError:
            self._flask_ok = False
            print("  [WireAPI] Flask/flask-cors not installed.")
            return

        self._setup_routes()

    # ──────────────────────────────────────────────────────────────────────
    # State update methods  (called by FL engine each round)
    # ──────────────────────────────────────────────────────────────────────

    def update_accuracy(self, log: list):
        self._state["accuracy_log"] = list(log)

    def update_trust(self, scores: dict):
        self._state["trust_scores"] = {str(k): float(v) for k, v in scores.items()}

    def update_round(self, r: int):
        self._state["round"] = r

    def register_node(self, info: dict):
        self._state["nodes"].append(info)

    # ──────────────────────────────────────────────────────────────────────
    # Routes
    # ──────────────────────────────────────────────────────────────────────

    def _setup_routes(self):
        from flask import jsonify, request
        import requests as _req

        app  = self.app
        self_ = self

        @app.route("/", methods=["GET"])
        def index():
            return jsonify({
                "service":    "FLoBC Wire API",
                "version":    "1.0",
                "endpoints":  [
                    "GET /health",
                    "GET /status",
                    "GET /chain",
                    "GET /chain/<index>",
                    "GET /chain/valid",
                    "GET /trust",
                    "GET /accuracy",
                    "GET /nodes",
                    "GET /stats",
                    "GET /transactions",
                ],
            })

        @app.route("/health", methods=["GET"])
        def health():
            rust_ok = False
            try:
                r = _req.get(f"{self_.rust_bc_url}/health", timeout=1.0)
                rust_ok = r.status_code == 200
            except Exception:
                pass
            return jsonify({
                "wire_api":         "ok",
                "rust_node_online": rust_ok,
                "rust_url":         self_.rust_bc_url,
                "uptime_s":         round(time.time() - self_._state["started_at"], 1),
            })

        @app.route("/status", methods=["GET"])
        def status():
            log = self_._state["accuracy_log"]
            return jsonify({
                "round":          self_._state["round"],
                "n_nodes":        len(self_._state["nodes"]),
                "final_accuracy": round(log[-1], 4) if log else 0.0,
                "max_accuracy":   round(max(log), 4) if log else 0.0,
                "trust":          self_._state["trust_scores"],
            })

        @app.route("/chain", methods=["GET"])
        def chain():
            """Proxy chain data from the Rust node."""
            try:
                r = _req.get(f"{self_.rust_bc_url}/chain", timeout=5.0)
                return jsonify(r.json())
            except Exception as e:
                return jsonify({"error": str(e), "source": "rust_node"}), 503

        @app.route("/chain/valid", methods=["GET"])
        def chain_valid():
            try:
                r = _req.get(f"{self_.rust_bc_url}/chain/valid", timeout=3.0)
                return jsonify(r.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 503

        @app.route("/chain/<int:index>", methods=["GET"])
        def chain_block(index):
            try:
                r = _req.get(f"{self_.rust_bc_url}/chain/{index}", timeout=3.0)
                return jsonify(r.json()), r.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 503

        @app.route("/transactions", methods=["GET"])
        def transactions():
            try:
                r = _req.get(f"{self_.rust_bc_url}/transactions", timeout=5.0)
                return jsonify(r.json())
            except Exception as e:
                return jsonify({"error": str(e)}), 503

        @app.route("/trust", methods=["GET"])
        def trust():
            """Current trust scores from the FL engine."""
            # Prefer live data from engine; fall back to Rust Wire API
            scores = self_._state["trust_scores"]
            if not scores:
                try:
                    r = _req.get(f"{self_.rust_bc_url}/wire/trust", timeout=2.0)
                    scores = r.json()
                except Exception:
                    pass
            return jsonify(scores)

        @app.route("/accuracy", methods=["GET"])
        def accuracy():
            """Accuracy log from the FL engine."""
            log = self_._state["accuracy_log"]
            if not log:
                try:
                    r = _req.get(f"{self_.rust_bc_url}/wire/accuracy", timeout=2.0)
                    data = r.json()
                    log  = data.get("accuracy_log", [])
                except Exception:
                    pass
            return jsonify({
                "accuracy_log": log,
                "final":  round(log[-1], 4) if log else 0.0,
                "max":    round(max(log), 4) if log else 0.0,
                "rounds": len(log),
            })

        @app.route("/nodes", methods=["GET"])
        def nodes():
            local = self_._state["nodes"]
            if not local:
                try:
                    r = _req.get(f"{self_.rust_bc_url}/wire/nodes", timeout=2.0)
                    local = r.json()
                except Exception:
                    pass
            return jsonify(local)

        @app.route("/stats", methods=["GET"])
        def stats():
            try:
                r = _req.get(f"{self_.rust_bc_url}/stats", timeout=3.0)
                return jsonify(r.json())
            except Exception as e:
                log = self_._state["accuracy_log"]
                return jsonify({
                    "error":           str(e),
                    "wire_api_status": "ok",
                    "final_accuracy":  round(log[-1], 4) if log else 0.0,
                })

    # ──────────────────────────────────────────────────────────────────────
    # Start
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        if not self._flask_ok:
            return None

        def _run():
            self.app.run(
                host="0.0.0.0", port=self.port,
                debug=False, use_reloader=False, threaded=True,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.4)
        print(f"  [HTTP] Wire API      → http://127.0.0.1:{self.port}")
        return t

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting FLoBC Wire API server on http://0.0.0.0:8080")
    print("Proxying chain queries to Rust node at http://127.0.0.1:8100")
    server = WireAPIServer()
    if server._flask_ok:
        server.app.run(host="0.0.0.0", port=WIRE_PORT, debug=False)
    else:
        print("Flask not available. Install: pip install flask flask-cors")
