"""
FLoBC Blockchain — Transaction Layer  (Exonum-compatible format)
================================================================
Exonum transaction wire format:
  {
    "service_id" : int,   # 1 = FLoBC service
    "message_id" : int,   # see MESSAGE_ID_* constants below
    "author"     : str,   # Ed25519 public key hex (64 chars = 32 bytes)
    "payload"    : dict,  # event-specific data
    "signature"  : str,   # Ed25519 signature of tx_hash (128 chars = 64 bytes)
    "tx_hash"    : str,   # SHA-256 of canonical JSON
    "tx_type"    : str,   # human-readable label (Exonum extension for FLoBC)
    "timestamp"  : float,
  }

Exonum service / message ID registry for FLoBC:
  Service 1 = "flobc"
    Message 0 = MODEL_UPDATE
    Message 1 = VALIDATION
    Message 2 = TRUST_UPDATE
    Message 3 = GLOBAL_MODEL
    Message 4 = PREDICTION_PROPOSAL
    Message 5 = VOTE
    Message 6 = WINNER_SELECTION
    Message 7 = REWARD
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any

from blockchain.crypto import sha256, Wallet


# ── Exonum service registry ──────────────────────────────────────────────────
EXONUM_SERVICE_ID   = 1       # FLoBC service running on the Exonum node
EXONUM_SERVICE_NAME = "flobc"

# Exonum message IDs (transaction type discriminants within service 1)
MESSAGE_ID = {
    "MODEL_UPDATE":         0,
    "VALIDATION":           1,
    "TRUST_UPDATE":         2,
    "GLOBAL_MODEL":         3,
    "PREDICTION_PROPOSAL":  4,
    "VOTE":                 5,
    "WINNER_SELECTION":     6,
    "REWARD":               7,
}
MESSAGE_NAME = {v: k for k, v in MESSAGE_ID.items()}

# Human-readable labels (kept for logging)
TX_MODEL_UPDATE  = "MODEL_UPDATE"
TX_VALIDATION    = "VALIDATION"
TX_TRUST_UPDATE  = "TRUST_UPDATE"
TX_GLOBAL_MODEL  = "GLOBAL_MODEL"
TX_PREDICTION    = "PREDICTION_PROPOSAL"
TX_VOTE          = "VOTE"
TX_WINNER        = "WINNER_SELECTION"
TX_REWARD        = "REWARD"


@dataclass
class Transaction:
    """
    Exonum-compatible signed transaction.

    Fields match the Exonum wire format exactly:
      service_id  — always 1 (FLoBC Exonum service)
      message_id  — discriminant within the service (0-7)
      author      — Ed25519 public key hex (the sender's Exonum identity)
      payload     — event-specific dict
      signature   — Ed25519 sig of tx_hash  (128-char hex)
      tx_hash     — SHA-256 of canonical JSON
      tx_type     — human-readable label (derived from message_id)
      timestamp   — Unix epoch float

    NOTE: `sender` is kept as an alias for `author` so existing engine code
    that references `tx.sender` continues to work unchanged.
    """
    tx_type:    str
    author:     str                        # Ed25519 pubkey hex (Exonum `author`)
    payload:    Dict[str, Any]
    timestamp:  float = field(default_factory=time.time)
    tx_hash:    str   = field(default="")
    signature:  str   = field(default="")
    service_id: int   = EXONUM_SERVICE_ID
    message_id: int   = field(default=0)

    def __post_init__(self):
        self.message_id = MESSAGE_ID.get(self.tx_type, 0)

    @property
    def sender(self) -> str:
        """Alias — existing engine code uses tx.sender; Exonum calls it author."""
        return self.author

    def _canonical(self) -> str:
        """
        Exonum canonical form for hashing:
        deterministic JSON with sorted keys and no whitespace.
        Includes service_id and message_id so two different tx types with
        identical payloads produce different hashes.
        """
        return json.dumps({
            "service_id": self.service_id,
            "message_id": self.message_id,
            "author":     self.author,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
        }, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        return sha256(self._canonical())

    def sign(self, wallet: Wallet) -> "Transaction":
        """Hash the canonical form then sign with Ed25519; returns self."""
        self.tx_hash  = self.compute_hash()
        self.signature = wallet.sign(self.tx_hash)
        return self

    def is_valid(self) -> bool:
        return self.tx_hash == self.compute_hash()

    def to_dict(self) -> dict:
        """Exonum wire format — what the Exonum node stores and returns."""
        return {
            "service_id": self.service_id,
            "message_id": self.message_id,
            "tx_type":    self.tx_type,
            "author":     self.author,
            "sender":     self.author,        # alias for backward compat
            "payload":    self.payload,
            "timestamp":  self.timestamp,
            "tx_hash":    self.tx_hash,
            "signature":  self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transaction":
        tx = cls(
            tx_type   = d.get("tx_type", MESSAGE_NAME.get(d.get("message_id", 0), "?")),
            author    = d.get("author", d.get("sender", "")),
            payload   = d["payload"],
            timestamp = d["timestamp"],
            tx_hash   = d["tx_hash"],
            signature = d["signature"],
        )
        tx.service_id = d.get("service_id", EXONUM_SERVICE_ID)
        tx.message_id = d.get("message_id", MESSAGE_ID.get(tx.tx_type, 0))
        return tx


# ── Exonum transaction factory helpers ────────────────────────────────────────

def make_model_update_tx(wallet: Wallet, trainer_id: int, round_num: int,
                          weights_hash: str, noise_level: float = 0.0) -> Transaction:
    return Transaction(
        tx_type  = TX_MODEL_UPDATE,
        author   = wallet.address,
        payload  = {"trainer_id": trainer_id, "round_num": round_num,
                    "weights_hash": weights_hash, "noise_level": noise_level},
    ).sign(wallet)


def make_validation_tx(wallet: Wallet, validator_id: int, trainer_id: int,
                        round_num: int, accepted: bool, score: float) -> Transaction:
    return Transaction(
        tx_type  = TX_VALIDATION,
        author   = wallet.address,
        payload  = {"validator_id": validator_id, "trainer_id": trainer_id,
                    "round_num": round_num, "accepted": accepted,
                    "score": round(score, 6)},
    ).sign(wallet)


def make_trust_update_tx(wallet: Wallet, trainer_id: int, old_score: float,
                          new_score: float, round_num: int) -> Transaction:
    return Transaction(
        tx_type  = TX_TRUST_UPDATE,
        author   = wallet.address,
        payload  = {"trainer_id": trainer_id, "old_score": round(old_score, 6),
                    "new_score": round(new_score, 6), "round_num": round_num},
    ).sign(wallet)


def make_global_model_tx(wallet: Wallet, round_num: int, weights_hash: str,
                          accuracy: float, accepted_trainers: list,
                          trust_scores: dict) -> Transaction:
    return Transaction(
        tx_type  = TX_GLOBAL_MODEL,
        author   = wallet.address,
        payload  = {"round_num": round_num, "weights_hash": weights_hash,
                    "accuracy": round(accuracy, 6),
                    "accepted_trainers": accepted_trainers,
                    "trust_scores": {str(k): round(v, 6)
                                     for k, v in trust_scores.items()}},
    ).sign(wallet)


def make_prediction_tx(wallet: Wallet, trainer_id: int, round_num: int,
                        predictions_hash: str, submit_elapsed: float) -> Transaction:
    return Transaction(
        tx_type  = TX_PREDICTION,
        author   = wallet.address,
        payload  = {"trainer_id": trainer_id, "round_num": round_num,
                    "predictions_hash": predictions_hash,
                    "submit_elapsed_s": round(submit_elapsed, 4)},
    ).sign(wallet)


def make_vote_tx(wallet: Wallet, voter_id: int, trainer_id: int, round_num: int,
                  accuracy_score: float, timeliness_score: float,
                  vote_score: float) -> Transaction:
    return Transaction(
        tx_type  = TX_VOTE,
        author   = wallet.address,
        payload  = {"voter_id": voter_id, "trainer_id": trainer_id,
                    "round_num": round_num,
                    "accuracy_score": round(accuracy_score, 6),
                    "timeliness_score": round(timeliness_score, 6),
                    "vote_score": round(vote_score, 6)},
    ).sign(wallet)


def make_winner_tx(wallet: Wallet, round_num: int, winners: list,
                    vote_scores: dict) -> Transaction:
    return Transaction(
        tx_type  = TX_WINNER,
        author   = wallet.address,
        payload  = {"round_num": round_num, "winners": winners,
                    "vote_scores": {str(k): round(v, 6)
                                    for k, v in vote_scores.items()}},
    ).sign(wallet)


def make_reward_tx(wallet: Wallet, trainer_id: int, round_num: int,
                    contribution: float) -> Transaction:
    return Transaction(
        tx_type  = TX_REWARD,
        author   = wallet.address,
        payload  = {"trainer_id": trainer_id, "round_num": round_num,
                    "contribution": round(contribution, 6)},
    ).sign(wallet)
