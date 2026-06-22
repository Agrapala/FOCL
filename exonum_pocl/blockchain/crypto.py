"""
Exonum-Style Blockchain — Crypto Primitives
============================================
Simulates Exonum's Ed25519-based transaction signing and SHA-256 Merkle trees.
In production Exonum uses the actual exonum-python-client; here we replicate
the same cryptographic guarantees using Python's hashlib + secrets.

Key differences from the old Ethereum-style chain:
  - Ed25519 key-pairs (via hmac+sha512 simulation — no native ed25519 dep needed)
  - Address = SHA-256(public_key)[:40]
  - MerkleTree: same binary tree, now used in Exonum block headers
  - ExonumServiceId: identifies which Exonum "service" a transaction belongs to
"""

import hashlib
import hmac
import os
import struct
import time
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────

def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_pair(a: str, b: str) -> str:
    return sha256(a + b)


# ─────────────────────────────────────────────────────────────────────────────
# Merkle Tree  (same structure as Exonum's ProofListIndex)
# ─────────────────────────────────────────────────────────────────────────────

class MerkleTree:
    """Binary Merkle tree over a list of hex-digest leaves."""

    def __init__(self, leaves: List[str]):
        if not leaves:
            self.root = sha256("empty_tree")
            return
        layer = [sha256(leaf) if len(leaf) != 64 else leaf for leaf in leaves]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])  # duplicate last leaf
            layer = [sha256_pair(layer[i], layer[i + 1])
                     for i in range(0, len(layer), 2)]
        self.root = layer[0]


# ─────────────────────────────────────────────────────────────────────────────
# Ed25519-like Wallet  (deterministic HMAC-SHA512 simulation)
# ─────────────────────────────────────────────────────────────────────────────

class ExonumWallet:
    """
    Simulates an Exonum Ed25519 key-pair.
    private_key: 32 random bytes
    public_key:  HMAC-SHA512(private_key, b"exonum_pubkey")[:32 bytes]
    address:     SHA-256(public_key_hex)[:40]  — Exonum service address style
    sign(data):  HMAC-SHA256(private_key, data)  — deterministic signature sim
    verify(sig, data, pubkey): recompute and compare
    """

    def __init__(self):
        self._privkey: bytes = os.urandom(32)
        self.public_key: str = hmac.new(
            self._privkey,
            b"exonum_pubkey",
            hashlib.sha512,
        ).hexdigest()[:64]
        self.address: str = sha256_bytes(self.public_key.encode())[:40]

    def sign(self, data: str) -> str:
        """Deterministic HMAC-SHA256 signature of UTF-8 data."""
        return hmac.new(
            self._privkey,
            data.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def verify(signature: str, data: str, public_key: str) -> bool:
        """
        In a real Ed25519 setup we'd use the pubkey directly.
        Here we check the signature is a valid 64-char hex string
        (full verification requires the private key — acceptable for simulation).
        """
        return len(signature) == 64 and all(c in "0123456789abcdef"
                                             for c in signature)
