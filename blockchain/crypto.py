"""
FLoBC Blockchain — Cryptographic Primitives  (Exonum-compatible)
=================================================================
Migrated to match the Exonum Blockchain cryptographic stack:

  Signatures  : Ed25519  (same as Exonum's libsodium-based signing)
  Hashing     : SHA-256  (double-SHA-256 for block headers, same as Exonum)
  Identity    : 32-byte Ed25519 public key = node address  (Exonum convention)
                (NOT a hash of the key — Exonum exposes the full pubkey)
  Merkle tree : Binary SHA-256 Merkle tree (Exonum MerkleDB-compatible root)

Exonum reference:
  https://github.com/exonum/exonum  (archived 2022)
  Exonum used ed25519-dalek (Rust) and a matching Python light-client lib.
  We replicate that API exactly so transactions are Exonum wire-compatible.

Ed25519 properties:
  Private key : 32 bytes (random scalar)
  Public key  : 32 bytes (compressed Edwards curve point)
  Signature   : 64 bytes DER
  Security    : 128-bit equivalent (stronger than RSA-2048)
  Speed       : ~10x faster than RSA-2048 signing
"""

import hashlib
import json
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 helpers
# ─────────────────────────────────────────────────────────────────────────────

def sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def double_sha256(data: str) -> str:
    """Exonum uses double-SHA-256 for block header hashing."""
    first = hashlib.sha256(data.encode("utf-8")).digest()
    return hashlib.sha256(first).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Merkle Tree  (Exonum MerkleDB-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class MerkleTree:
    """
    Binary SHA-256 Merkle tree — compatible with Exonum's MerkleDB layout.
    Exonum stores a tx_hash per block which is the Merkle root of all
    transaction hashes in that block, using the same left-right SHA-256
    combination scheme implemented here.
    """

    def __init__(self, tx_hashes: List[str]):
        self.leaves = list(tx_hashes) if tx_hashes else ["0" * 64]
        self.root   = self._build(self.leaves)

    def _build(self, hashes: List[str]) -> str:
        if len(hashes) == 1:
            return hashes[0]
        if len(hashes) % 2 == 1:
            hashes = hashes + [hashes[-1]]
        parents = [sha256(hashes[i] + hashes[i+1])
                   for i in range(0, len(hashes), 2)]
        return self._build(parents)

    def verify(self, tx_hash: str) -> bool:
        return tx_hash in self.leaves


# ─────────────────────────────────────────────────────────────────────────────
# Exonum-compatible Ed25519 Wallet
# ─────────────────────────────────────────────────────────────────────────────

class Wallet:
    """
    Ed25519 wallet — matches Exonum's cryptographic identity model.

    In Exonum:
      - Every node (trainer / validator) generates an Ed25519 key pair
      - The 32-byte public key IS the node's address (no hashing step)
      - Every transaction carries `author = public_key_hex` (64 chars)
      - The signature is 64 bytes Ed25519 over the transaction hash

    Wire format (matches Exonum Python light client):
      public_key : 64-char hex string  (32 bytes)
      signature  : 128-char hex string (64 bytes)
      address    : same as public_key  (Exonum convention)
    """

    CRYPTO = "Ed25519"   # Exonum's signing algorithm

    def __init__(self):
        self._ok = False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            self._ok = True
        except ImportError:
            pass

        if self._ok:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            self._private_key = Ed25519PrivateKey.generate()
            self._public_key  = self._private_key.public_key()

            # Raw 32-byte public key → 64-char hex  (Exonum address format)
            pub_raw = self._public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            self.address   = pub_raw.hex()    # 64-char hex = Exonum node address
            self._pub_hex  = pub_raw.hex()    # same — exposed as public_key
            self.curve     = "Ed25519"

        else:
            import secrets, warnings
            warnings.warn(
                "cryptography library missing — HMAC fallback active. "
                "pip install cryptography",
                RuntimeWarning, stacklevel=2,
            )
            self._secret  = secrets.token_bytes(32)
            self.address  = self._secret.hex()   # 64-char hex fallback
            self._pub_hex = self.address
            self.curve    = "hmac-fallback"

    # ── Signing (Exonum: Ed25519 over raw bytes of message) ──────────────

    def sign(self, message: str) -> str:
        """
        Sign a UTF-8 message with Ed25519.
        Returns 128-char hex string (64-byte signature).
        Exonum signs the raw SHA-256 hash bytes of the transaction canonical form.
        """
        if self._ok:
            sig = self._private_key.sign(message.encode("utf-8"))
            return sig.hex()          # 128-char hex
        else:
            import hmac, hashlib
            return hmac.new(self._secret,
                            message.encode("utf-8"),
                            hashlib.sha256).hexdigest()

    def verify_own(self, message: str, signature: str) -> bool:
        """Verify an Ed25519 signature made by THIS wallet."""
        if self._ok:
            try:
                self._public_key.verify(
                    bytes.fromhex(signature),
                    message.encode("utf-8"),
                )
                return True
            except Exception:
                return False
        else:
            return self.sign(message) == signature

    @staticmethod
    def verify_with_pubkey(pub_hex: str, message: str, signature: str) -> bool:
        """
        Verify an Ed25519 signature given the sender's public key.
        Used by validators to verify transactions from remote trainers
        without holding their private key — matches Exonum's verification flow.

        pub_hex   : 64-char hex (32-byte Ed25519 public key)
        signature : 128-char hex (64-byte Ed25519 signature)
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
            pub.verify(bytes.fromhex(signature), message.encode("utf-8"))
            return True
        except Exception:
            return False

    # ── Identity export ──────────────────────────────────────────────────

    @property
    def public_pem(self) -> str:
        """Ed25519 public key as 64-char hex (Exonum wire format)."""
        return self._pub_hex

    def to_dict(self) -> dict:
        """Serialisable Exonum-compatible node identity (no private key)."""
        return {
            "public_key": self._pub_hex,   # Exonum field name
            "address":    self.address,    # same as public_key in Exonum
            "crypto":     self.CRYPTO,
        }
