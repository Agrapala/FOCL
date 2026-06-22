"""
validator_node.py  —  HTTP Flask Validator Server
==================================================
Each BCValidator in FLoBC runs as a real HTTP server (one Flask app per
validator).  Python trainer nodes POST their weight updates to each validator
via HTTP — replacing the in-process function calls of the baseline engine.

This matches the paper's architecture:
  Trainer  →  HTTP POST /validate  →  Validator
  Trainer  →  HTTP GET  /model     →  Validator (pull latest model)

The Wire API (read-only) is co-hosted on the same Flask app:
  GET /chain         full blockchain (forwarded from Rust node)
  GET /trust         current trust scores
  GET /latest_model  current global model hash + accuracy
  GET /nodes         validator info + wallet address
  GET /health        liveness

Usage (internal — called by run_network.py):
    server = ValidatorHTTPServer(vid=0, port=7000, ...)
    server.start()   # background thread, returns immediately
"""

import sys, os, json, time, threading, logging
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from blockchain.crypto import sha256_bytes, Wallet

# Silence Flask's noisy startup log
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


class ValidatorHTTPServer:
    """
    One Flask HTTP server representing one FLoBC validator node.

    Ports
    -----
    vid=0 → port 7000
    vid=1 → port 7001
    vid=2 → port 7002

    Endpoints
    ---------
    POST /validate       receive trainer update, return accept/reject
    GET  /model          return current global model weights as JSON array
    GET  /chain          blockchain summary (from shared Python chain object)
    GET  /trust          trust scores (from shared trust service)
    GET  /latest_model   model hash + validation accuracy
    GET  /nodes          this validator's identity
    GET  /health         liveness check
    """

    def __init__(self, vid: int, port: int,
                 X_val: np.ndarray, y_val: np.ndarray,
                 global_model,        # PneumoniaModel reference (shared)
                 trust_service,       # TrustService reference (shared)
                 chain,               # RealBlockchain reference (shared)
                 rust_bc_url: str = "http://127.0.0.1:8100"):
        self.vid          = vid
        self.port         = port
        self.X_val        = X_val
        self.y_val        = y_val
        self.global_model = global_model
        self.trust        = trust_service
        self.chain        = chain
        self.rust_url     = rust_bc_url
        self.wallet       = Wallet()
        self._round_num   = 0
        self._baseline    = None   # cached baseline accuracy per round

        # Try to import Flask
        try:
            from flask import Flask
            self.app = Flask(f"validator_{vid}")
            self._flask_ok = True
        except ImportError:
            self._flask_ok = False
            print(f"  [V{vid}] Flask not installed — HTTP server disabled. "
                  "Run: pip install flask")
            return

        self._setup_routes()

    # ──────────────────────────────────────────────────────────────────────
    # Route registration
    # ──────────────────────────────────────────────────────────────────────

    def _setup_routes(self):
        from flask import request, jsonify

        app = self.app
        srv = self   # closure reference

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status":    "ok",
                "vid":       srv.vid,
                "port":      srv.port,
                "address":   srv.wallet.address,
                "curve":     srv.wallet.curve,
            })

        @app.route("/validate", methods=["POST"])
        def validate():
            """
            Receive a trainer's model update and decide accept/reject.

            Request JSON:
              {
                "trainer_id":   int,
                "node_id":      "A" | "B" | "C" | "D",
                "round_num":    int,
                "weights":      [float, ...],   // flattened model weights
                "weights_hash": str,            // SHA-256 of weights bytes
                "signature":    str,            // ECDSA sig of weights_hash
                "pub_key_hex":  str,            // trainer's compressed pubkey
              }

            Response JSON:
              {
                "accepted":     bool,
                "score":        float,   // candidate accuracy
                "baseline":     float,   // current global accuracy
                "delta":        float,   // score - baseline
                "validator_id": int,
                "address":      str,
              }
            """
            data        = request.get_json(force=True)
            trainer_id  = int(data.get("trainer_id", 0))
            round_num   = int(data.get("round_num", 0))
            weights_arr = np.array(data["weights"], dtype=np.float32)
            weights_hash= data.get("weights_hash", "")
            signature   = data.get("signature", "")
            pub_key_hex = data.get("pub_key_hex", "")

            # 1. Verify ECDSA signature on the weights hash
            sig_ok = True
            if pub_key_hex and signature:
                from blockchain.crypto import Wallet as W
                sig_ok = W.verify_with_pubkey(pub_key_hex, weights_hash, signature)

            # 2. Evaluate candidate model on validator's private data
            candidate = srv.global_model.clone()
            candidate.unflatten(weights_arr)
            candidate_score = float(candidate.accuracy(srv.X_val, srv.y_val))

            # 3. Baseline: current global model accuracy on this round
            if srv._round_num != round_num or srv._baseline is None:
                srv._round_num = round_num
                srv._baseline  = float(srv.global_model.accuracy(
                                           srv.X_val, srv.y_val))
            baseline = srv._baseline

            # 4. Accept if candidate ≥ baseline - 0.08 (forgiving threshold)
            delta    = candidate_score - baseline
            accepted = sig_ok and (candidate_score >= max(0.45, baseline - 0.08))

            return jsonify({
                "accepted":     accepted,
                "score":        round(candidate_score, 4),
                "baseline":     round(baseline, 4),
                "delta":        round(delta, 4),
                "sig_ok":       sig_ok,
                "validator_id": srv.vid,
                "address":      srv.wallet.address[:16] + "...",
            })

        @app.route("/model", methods=["GET"])
        def get_model():
            """Return current global model weights as a float list."""
            flat = srv.global_model.flatten().tolist()
            h    = sha256_bytes(srv.global_model.flatten()
                                    .astype(np.float32).tobytes())
            return jsonify({
                "weights":      flat,
                "weights_hash": h,
                "accuracy":     round(float(srv.global_model.accuracy(
                                    srv.X_val, srv.y_val)), 4),
                "dim":          len(flat),
            })

        @app.route("/chain", methods=["GET"])
        def get_chain():
            c = srv.chain
            blocks = []
            for b in c._chain:
                blocks.append({
                    "index":      b.index,
                    "hash":       b.block_hash[:14] + "...",
                    "prev":       b.previous_hash[:14] + "...",
                    "tx_count":   len(b.transactions),
                    "merkle":     b.merkle_root[:14] + "...",
                    "timestamp":  b.timestamp,
                })
            return jsonify({
                "chain_length": c.length(),
                "chain_valid":  c.is_chain_valid(),
                "blocks":       blocks,
            })

        @app.route("/trust", methods=["GET"])
        def get_trust():
            return jsonify({
                str(k): round(float(v), 6)
                for k, v in srv.trust.scores.items()
            })

        @app.route("/latest_model", methods=["GET"])
        def latest_model():
            h   = sha256_bytes(srv.global_model.flatten()
                                   .astype(np.float32).tobytes())
            acc = float(srv.global_model.accuracy(srv.X_val, srv.y_val))
            return jsonify({
                "weights_hash": h,
                "accuracy":     round(acc, 4),
                "validator_id": srv.vid,
                "round":        srv._round_num,
            })

        @app.route("/nodes", methods=["GET"])
        def get_nodes():
            return jsonify({
                "type":    "validator",
                "vid":     srv.vid,
                "port":    srv.port,
                "address": srv.wallet.address,
                "pub_key": srv.wallet.public_pem,
                "curve":   srv.wallet.curve,
            })

    # ──────────────────────────────────────────────────────────────────────
    # Start / stop
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        """Start Flask in a daemon background thread."""
        if not self._flask_ok:
            return None

        def _run():
            self.app.run(
                host="127.0.0.1", port=self.port,
                debug=False, use_reloader=False, threaded=True,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(0.4)   # brief wait for socket to bind
        print(f"  [HTTP] Validator {self.vid} → http://127.0.0.1:{self.port}")
        return t

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
