"""
http_engine.py  —  FLoBC HTTP-based Federated Learning Engine
==============================================================
This module runs the same FL training loop as FloBCPneumonia but uses
REAL HTTP communication between trainers and validators (replacing
in-process function calls).

Architecture
------------
  TrainerHTTPClient  →  POST /validate  →  ValidatorHTTPServer  (Flask)
  Python FL engine   →  POST /block/propose  →  Rust pBFT node  (port 8100)
  Wire API           →  GET  /chain,/trust   ←  JavaScript client

Each round:
  1. Every hospital trainer trains locally (NumPy SGD — unchanged)
  2. Trainer signs weights with ECDSA secp256k1 wallet
  3. Trainer POSTs weights + signature to each validator via HTTP
  4. Validators accept/reject via HTTP response
  5. Accepted updates are FedAvg-aggregated into global model
  6. Block is committed to Rust pBFT node via HTTP POST
  7. Trust scores + accuracy pushed to Rust Wire API
"""

import sys, os, time, json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.flobc_pneumonia_engine  import PneumoniaModel, TrustService, HOSPITAL_NODES
from core.pneumonia_loader        import HOSPITAL_NODES as _HN
from network.validator_node       import ValidatorHTTPServer
from network.trainer_client       import TrainerHTTPClient
from network.wire_api             import WireAPIServer
from blockchain.crypto            import sha256_bytes, Wallet
from blockchain.chain             import RealBlockchain, ProofOfStake


HOSP = {nid: cfg["name"] for nid, cfg in _HN.items()}
NID_TO_TID = {"A": 0, "B": 1, "C": 2, "D": 3}

VALIDATOR_BASE_PORT = 7000
WIRE_API_PORT       = 8080
RUST_BC_URL         = "http://127.0.0.1:8100"


class HTTPFloBCEngine:
    """
    Production-grade FLoBC engine with real HTTP communication.

    Differences from FloBCPneumonia (in-process baseline):
    - Trainer→validator communication goes over HTTP POST (real network protocol)
    - Weights + ECDSA signatures travel as JSON over TCP
    - Blocks are committed to the Rust pBFT node via HTTP
    - Wire API exposes live state to the JavaScript dashboard
    - secp256k1 wallets sign every transaction (upgraded from RSA-2048)

    Usage:
        engine = HTTPFloBCEngine(per_node_train, X_val, y_val, X_test, y_test)
        engine.start_servers()    # starts Flask validators + Wire API in threads
        results = engine.train(n_rounds=20)
        engine.stop_servers()
    """

    def __init__(
        self,
        per_node_train: Dict[str, Tuple[np.ndarray, np.ndarray]],
        X_val:  np.ndarray, y_val:  np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        n_validators:  int   = 3,
        lr:            float = 0.008,
        batch_size:    int   = 512,
        local_epochs:  int   = 5,
        noise_profile: Optional[Dict[str, float]] = None,
        verbose:       bool  = True,
    ):
        self.per_node_train = per_node_train
        self.X_val,  self.y_val  = X_val,  y_val
        self.X_test, self.y_test = X_test, y_test
        self.n_validators  = n_validators
        self.lr            = lr
        self.batch_size    = batch_size
        self.local_epochs  = local_epochs
        self.noise_profile = noise_profile or {}
        self.verbose       = verbose

        feat_dim = X_val.shape[1]

        # ── Global model (shared reference across all validators) ──────────
        self.global_model = PneumoniaModel(feat_dim, 256, output_dim=2)

        # ── Trust service ──────────────────────────────────────────────────
        self.trust = TrustService(list(range(4)))

        # ── Python blockchain (source of truth for Python layer) ───────────
        self.chain = RealBlockchain()
        self._fw_wallet = Wallet()
        n_val = n_validators
        val_stakes = {}

        # ── Validator HTTP servers (Flask) ─────────────────────────────────
        self.validator_servers: List[ValidatorHTTPServer] = []
        val_split = len(X_val) // n_val
        for i in range(n_val):
            lo = i * val_split
            hi = lo + val_split if i < n_val - 1 else len(X_val)
            srv = ValidatorHTTPServer(
                vid=i,
                port=VALIDATOR_BASE_PORT + i,
                X_val=X_val[lo:hi],
                y_val=y_val[lo:hi],
                global_model=self.global_model,
                trust_service=self.trust,
                chain=self.chain,
                rust_bc_url=RUST_BC_URL,
            )
            self.validator_servers.append(srv)
            val_stakes[srv.wallet.address] = 1.0 / n_val

        self.pos = ProofOfStake(val_stakes)

        # ── HTTP trainer clients ───────────────────────────────────────────
        validator_urls = [f"http://127.0.0.1:{VALIDATOR_BASE_PORT + i}"
                          for i in range(n_val)]
        self.trainer_clients: List[TrainerHTTPClient] = []
        for nid in ["A", "B", "C", "D"]:
            tc = TrainerHTTPClient(
                node_id=nid,
                validator_urls=validator_urls,
                rust_bc_url=RUST_BC_URL,
            )
            self.trainer_clients.append(tc)

        # ── Local model copies per trainer (hospital data private) ─────────
        self._trainer_models: Dict[str, PneumoniaModel] = {}
        for nid in ["A", "B", "C", "D"]:
            m = self.global_model.clone()
            self._trainer_models[nid] = m

        # ── Wire API server ────────────────────────────────────────────────
        self.wire_api = WireAPIServer(rust_bc_url=RUST_BC_URL, port=WIRE_API_PORT)

        # ── Logging ────────────────────────────────────────────────────────
        self.accuracy_log: List[float] = []
        self.trust_log: Dict[int, List[float]] = defaultdict(list)
        self.round_times: List[float] = []

    # ──────────────────────────────────────────────────────────────────────
    # Server lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start_servers(self):
        """Start Flask validators + Wire API in background threads."""
        print("\n  [HTTP] Starting validator HTTP servers ...")
        for srv in self.validator_servers:
            srv.start()

        print("  [HTTP] Starting Wire API server ...")
        self.wire_api.start()

        # Register all nodes with Rust Wire API
        for tc in self.trainer_clients:
            tc.register_node_with_rust(role="trainer")
            self.wire_api.register_node(tc.identity)
        for srv in self.validator_servers:
            self.wire_api.register_node({
                "role":    "validator",
                "vid":     srv.vid,
                "address": srv.wallet.address,
                "pub_key": srv.wallet.public_pem,
                "curve":   srv.wallet.curve,
            })

        print(f"  [HTTP] All servers ready.")
        print(f"  [HTTP] JavaScript dashboard → "
              f"open dashboard/client.html in a browser")
        print(f"  [HTTP] Wire API             → http://127.0.0.1:{WIRE_API_PORT}")
        print()

    def stop_servers(self):
        """Background threads are daemonized — they stop with the process."""
        pass

    # ──────────────────────────────────────────────────────────────────────
    # Local training (unchanged — pure NumPy SGD)
    # ──────────────────────────────────────────────────────────────────────

    def _local_train(self, nid: str) -> np.ndarray:
        """Run local SGD on hospital's private data. Returns flat weights."""
        X_tr, y_tr = self.per_node_train[nid]
        model = self._trainer_models[nid]

        noise_std = self.noise_profile.get(nid, 0.0)

        n = len(X_tr)
        for _ in range(self.local_epochs):
            perm    = np.random.permutation(n)
            X_sh, y_sh = X_tr[perm], y_tr[perm]
            for s in range(0, n, self.batch_size):
                Xb, yb = X_sh[s:s + self.batch_size], y_sh[s:s + self.batch_size]
                if len(Xb):
                    model.sgd_step(Xb, yb, lr=self.lr)

        flat = model.flatten()
        if noise_std > 0:
            flat = flat + np.random.normal(0, noise_std, flat.shape).astype(np.float32)
        return flat

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self, n_rounds: int = 20) -> Dict:
        """
        Run n_rounds of HTTP-based federated learning.
        Returns the same result dict as FloBCPneumonia.train().
        """
        init_acc = self.global_model.accuracy(self.X_test, self.y_test)
        self.accuracy_log.append(init_acc)
        print(f"  Start accuracy: {init_acc:.4f} (random init)")
        print(f"  Trainer→Validator communication: HTTP POST (real network)")
        print(f"  Blockchain backend: Rust pBFT node at {RUST_BC_URL}")
        print()

        DIV = "=" * 66

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            print(f"  +-- Round {rnd:02d}/{n_rounds:02d}  [HTTP-BSP]"
                  + "-" * 40 + "+")

            updates_accepted = {}   # nid -> flat weights
            block_txs        = []

            # ── A: Local training (all hospitals in parallel conceptually) ─
            for tc, nid in zip(self.trainer_clients, ["A", "B", "C", "D"]):
                tid  = NID_TO_TID[nid]
                flat = self._local_train(nid)
                w_hash = sha256_bytes(flat.astype(np.float32).tobytes())

                # ── B: Submit to validators via HTTP ──────────────────────
                responses = tc.submit_update(flat, rnd, tid)
                n_accepted = sum(1 for r in responses if r.get("accepted", False))
                n_total    = len(responses)
                accepted   = n_accepted > (n_total * 2 / 3)

                # ── C: Trust update ───────────────────────────────────────
                if responses:
                    avg_score = np.mean([r.get("score", 0.0) for r in responses])
                    avg_base  = np.mean([r.get("baseline", 0.0) for r in responses])
                    delta_acc = avg_score - avg_base
                else:
                    delta_acc = -0.01

                self.trust.update(tid, delta_acc)

                if accepted:
                    updates_accepted[nid] = flat

                # Record transaction for block
                block_txs.append({
                    "tx_type":      "MODEL_UPDATE",
                    "sender":       tc.wallet.address,
                    "payload":      {"node_id": nid, "weights_hash": w_hash,
                                     "round": rnd, "accepted": accepted,
                                     "n_validators_accepted": n_accepted},
                    "signature":    tc.wallet.sign(w_hash),
                })

                if self.verbose:
                    print(f"  |  [{nid}] HTTP validators: "
                          f"{n_accepted}/{n_total} accepted  "
                          f"hash={w_hash[:10]}...")

            # ── D: FedAvg aggregation ─────────────────────────────────────
            if updates_accepted:
                weights_list, weight_factors = [], []
                for nid, flat in updates_accepted.items():
                    tid = NID_TO_TID[nid]
                    weights_list.append(flat)
                    weight_factors.append(self.trust.weight(tid))

                total_w = sum(weight_factors) or 1e-9
                stacked = np.stack(weights_list, axis=0)
                factors = np.array(weight_factors) / total_w
                agg_flat = (stacked * factors[:, None]).sum(axis=0).astype(np.float32)
                self.global_model.unflatten(agg_flat)

                # Sync trainer models to new global
                for nid in ["A", "B", "C", "D"]:
                    self._trainer_models[nid].unflatten(
                        self.global_model.flatten())

            # ── E: Evaluate global model ──────────────────────────────────
            g_acc = self.global_model.accuracy(self.X_test, self.y_test)
            self.accuracy_log.append(g_acc)
            for tid, s in self.trust.scores.items():
                self.trust_log[tid].append(s)

            # ── F: Commit block to Rust pBFT node via HTTP ────────────────
            votes = [
                {"validator": srv.wallet.address,
                 "vote":  True,
                 "stake": 1.0 / self.n_validators}
                for srv in self.validator_servers
            ]
            rust_result = self.trainer_clients[0].propose_block_to_rust(
                transactions=block_txs,
                validator_address=self.validator_servers[0].wallet.address,
                votes=votes,
            )
            rust_ok = rust_result is not None and rust_result.get("committed", False)

            # ── G: Push state to Wire API ──────────────────────────────────
            self.wire_api.update_accuracy(self.accuracy_log)
            self.wire_api.update_trust(self.trust.scores)
            self.wire_api.update_round(rnd)
            self.trainer_clients[0].push_trust_to_rust(self.trust.scores)
            self.trainer_clients[0].push_accuracy_to_rust(self.accuracy_log)

            elapsed = time.time() - t0
            self.round_times.append(elapsed)
            trust_str = "  ".join(
                f"N{chr(65+i)}={self.trust.scores.get(i, 0):.3f}"
                for i in range(4))

            print(f"  |  Global accuracy = {g_acc:.4f}  "
                  f"Rust block = {'OK' if rust_ok else 'MISS'}")
            print(f"  |  Trust: {trust_str}  ({elapsed:.1f}s)")
            print(f"  +" + "-" * 65 + "+")
            print()

        return {
            "accuracy_log":  self.accuracy_log,
            "trust_log":     dict(self.trust_log),
            "round_times":   self.round_times,
            "final_accuracy": self.accuracy_log[-1],
            "chain_valid":   self.chain.is_chain_valid(),
            "chain_length":  self.chain.length(),
            "final_trust":   dict(self.trust.scores),
        }
