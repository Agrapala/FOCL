"""
trainer_client.py  —  HTTP Trainer Client
==========================================
Wraps the existing HospitalTrainer with real HTTP communication.
Instead of calling validator.validate() directly (in-process),
it POSTs the trained model update to each validator's HTTP endpoint.

This is the real decentralized communication path described in the paper:
  Trainer  →  HTTP POST /validate  →  ValidatorServer (Flask)

Also pushes each round's results to the Rust blockchain node via
  Python  →  HTTP POST /block/propose  →  Rust pBFT node (port 8100)
"""

import sys, os, json, time
import numpy as np
from typing import List, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from blockchain.crypto import sha256_bytes, Wallet


class TrainerHTTPClient:
    """
    HTTP client used by a hospital trainer to communicate with validators.

    Each FL round:
    1.  Trainer trains locally (NumPy SGD — unchanged)
    2.  Trainer signs the weights hash with ECDSA secp256k1
    3.  For each validator, POST /validate with weights + signature
    4.  Collect accept/reject decisions
    5.  Report back to the orchestrator

    The weights themselves travel over HTTP (real network protocol).
    Only the hash is stored on-chain (privacy: raw weights never on-chain).
    """

    def __init__(self, node_id: str, validator_urls: List[str],
                 rust_bc_url: str = "http://127.0.0.1:8100"):
        self.node_id        = node_id
        self.validator_urls = validator_urls  # e.g. ["http://127.0.0.1:7000", ...]
        self.rust_bc_url    = rust_bc_url
        self.wallet         = Wallet()
        self._requests_ok   = False

        try:
            import requests
            self._requests_ok = True
        except ImportError:
            print(f"  [Trainer {node_id}] 'requests' not installed. "
                  "Run: pip install requests")

    # ──────────────────────────────────────────────────────────────────────
    # Validator communication
    # ──────────────────────────────────────────────────────────────────────

    def submit_update(
        self,
        weights:     np.ndarray,
        round_num:   int,
        trainer_id:  int,
        timeout:     float = 10.0,
    ) -> List[Dict]:
        """
        Send model weights to ALL validators and collect their verdicts.

        Returns a list of response dicts, one per validator:
          {"accepted": bool, "score": float, "delta": float,
           "validator_id": int, "sig_ok": bool}
        """
        if not self._requests_ok:
            return []

        import requests

        weights_f32  = weights.astype(np.float32)
        weights_hash = sha256_bytes(weights_f32.tobytes())
        signature    = self.wallet.sign(weights_hash)

        payload = {
            "trainer_id":   trainer_id,
            "node_id":      self.node_id,
            "round_num":    round_num,
            "weights":      weights_f32.tolist(),
            "weights_hash": weights_hash,
            "signature":    signature,
            "pub_key_hex":  self.wallet.public_pem,
        }

        results = []
        for url in self.validator_urls:
            try:
                r = requests.post(
                    f"{url}/validate",
                    json=payload,
                    timeout=timeout,
                )
                if r.status_code == 200:
                    results.append(r.json())
                else:
                    results.append({
                        "accepted": False,
                        "error":    f"HTTP {r.status_code}",
                        "score":    0.0,
                    })
            except requests.exceptions.RequestException as e:
                results.append({
                    "accepted": False,
                    "error":    str(e),
                    "score":    0.0,
                })

        return results

    def pull_model(self, validator_url: str) -> Optional[np.ndarray]:
        """
        Pull the latest global model weights from a validator via GET /model.
        Used when a trainer needs to sync to the current global model.
        """
        if not self._requests_ok:
            return None

        import requests
        try:
            r = requests.get(f"{validator_url}/model", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                return np.array(data["weights"], dtype=np.float32)
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Rust blockchain interaction
    # ──────────────────────────────────────────────────────────────────────

    def propose_block_to_rust(
        self,
        transactions: List[Dict],
        validator_address: str,
        votes: List[Dict],
        threshold: float = 2.0 / 3.0,
    ) -> Optional[Dict]:
        """
        Submit a block proposal to the Rust pBFT blockchain node.

        votes = [{"validator": address, "vote": True, "stake": 0.333}, ...]
        """
        if not self._requests_ok:
            return None

        import requests
        payload = {
            "transactions": transactions,
            "validator":    validator_address,
            "votes":        votes,
            "threshold":    threshold,
        }
        try:
            r = requests.post(
                f"{self.rust_bc_url}/block/propose",
                json=payload,
                timeout=10.0,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            print(f"  [HTTP] Rust node unreachable: {e}")
            return None

    def push_trust_to_rust(self, trust_scores: Dict) -> bool:
        """Push current trust scores to Rust Wire API."""
        if not self._requests_ok:
            return False
        import requests
        try:
            r = requests.post(
                f"{self.rust_bc_url}/wire/trust",
                json={str(k): float(v) for k, v in trust_scores.items()},
                timeout=3.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def push_accuracy_to_rust(self, accuracy_log: List[float]) -> bool:
        """Push accuracy log to Rust Wire API."""
        if not self._requests_ok:
            return False
        import requests
        try:
            r = requests.post(
                f"{self.rust_bc_url}/wire/accuracy",
                json=[float(a) for a in accuracy_log],
                timeout=3.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def register_node_with_rust(self, role: str = "trainer") -> bool:
        """Register this node's identity with the Rust Wire API."""
        if not self._requests_ok:
            return False
        import requests
        try:
            r = requests.post(
                f"{self.rust_bc_url}/wire/nodes",
                json={
                    "role":    role,
                    "node_id": self.node_id,
                    "address": self.wallet.address,
                    "pub_key": self.wallet.public_pem,
                    "curve":   self.wallet.curve,
                },
                timeout=3.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    @property
    def identity(self) -> Dict:
        return {
            "node_id": self.node_id,
            "address": self.wallet.address,
            "pub_key": self.wallet.public_pem,
            "curve":   self.wallet.curve,
        }
