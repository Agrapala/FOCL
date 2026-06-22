"""
IPFS Off-Chain Storage Simulation
===================================
In the paper, full model files (.h5 / .npy) are stored in IPFS while only
the Content Identifier (CID) and accuracy metadata are stored on-chain.

This module simulates IPFS:
  - store(data_bytes)  → CID (SHA-256 hash prefixed with "Qm" like real IPFS)
  - retrieve(cid)      → data_bytes
  - exists(cid)        → bool

In production: replace with ipfshttpclient.Client().add_bytes() / .cat()

Benefits simulated:
  - Gas cost reduction: only 64-char CID on-chain vs full model weights
  - Immutability: CID = hash of content (content-addressed)
  - Retrieval: any node with the CID can fetch the model
"""

import hashlib
import numpy as np
from typing import Dict, Optional


class IPFSNode:
    """
    In-memory IPFS simulation.
    CID format: "Qm" + SHA-256(content)[:46]  (mimics IPFS CIDv0 length)
    """

    def __init__(self):
        self._store: Dict[str, bytes] = {}
        self._pin_counts: Dict[str, int] = {}
        self._total_stored_bytes: int = 0

    def _make_cid(self, data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        return "Qm" + h[:46]

    def store(self, data: bytes) -> str:
        """Store bytes, return CID. Content-addressed: same data → same CID."""
        cid = self._make_cid(data)
        if cid not in self._store:
            self._store[cid] = data
            self._total_stored_bytes += len(data)
        self._pin_counts[cid] = self._pin_counts.get(cid, 0) + 1
        return cid

    def store_model(self, weights: np.ndarray) -> str:
        """Serialize numpy array and store."""
        data = weights.astype(np.float32).tobytes()
        return self.store(data)

    def retrieve(self, cid: str) -> Optional[bytes]:
        return self._store.get(cid)

    def retrieve_model(self, cid: str, shape_hint=None) -> Optional[np.ndarray]:
        data = self.retrieve(cid)
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.float32).copy()
        return arr

    def exists(self, cid: str) -> bool:
        return cid in self._store

    def pin_count(self, cid: str) -> int:
        return self._pin_counts.get(cid, 0)

    def stats(self) -> dict:
        return {
            "total_objects":      len(self._store),
            "total_bytes_stored": self._total_stored_bytes,
            "total_bytes_MB":     round(self._total_stored_bytes / 1e6, 3),
        }

    def gas_savings_vs_onchain(self, n_blocks: int,
                                weights_size_bytes: int) -> dict:
        """
        Estimate gas/cost savings from off-chain IPFS vs storing full weights on-chain.
        On Ethereum/Exonum: 1 byte on-chain ≈ 68 gas ≈ 0.0000002 ETH (rough estimate).
        CID on-chain = 48 bytes. Full model = weights_size_bytes.
        """
        cid_bytes         = 48
        gas_per_byte      = 68
        onchain_gas       = weights_size_bytes * gas_per_byte * n_blocks
        ipfs_gas          = cid_bytes * gas_per_byte * n_blocks
        savings_gas       = onchain_gas - ipfs_gas
        savings_pct       = 100.0 * savings_gas / (onchain_gas + 1)
        return {
            "onchain_gas_total":   onchain_gas,
            "ipfs_gas_total":      ipfs_gas,
            "savings_gas":         savings_gas,
            "savings_pct":         round(savings_pct, 2),
            "model_size_bytes":    weights_size_bytes,
            "cid_size_bytes":      cid_bytes,
        }


# Global shared IPFS node (all network participants share one simulated node)
IPFS = IPFSNode()
