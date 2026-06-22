"""
Exonum-Style Transactions
==========================
Each transaction type maps to an Exonum "message" with:
  - service_id   : which Exonum service handles this message
  - message_id   : message type within the service
  - payload      : dict of business data
  - sender       : wallet address
  - signature    : HMAC-SHA256(private_key, canonical_payload)
  - tx_hash      : SHA-256(service_id + message_id + sender + signature)

Transaction types used in this FL system (mapped onto the paper's
PoCL 5-phase protocol — Section 3.4):

  MODEL_UPDATE         (service=1, msg=1) — Phase A: Model Proposal
                        Trainer/Miner submits IPFS CID + accuracy of its
                        locally trained (and optionally DP-noised) model.

  VALIDATION           (service=1, msg=2) — Validator accept/reject vote
                        on a miner's model proposal (secure validation set).

  TRUST_UPDATE         (service=1, msg=3) — Reputation/trust change for a
                        miner, driven by the PoCL reward mechanism.

  GLOBAL_MODEL         (service=1, msg=4) — Phase F: Block Creation summary.
                        Aggregated global model hash + round metadata.

  PBFT_VOTE            (service=2, msg=1) — pBFT/PoCL-pBFT prepare/commit vote.

  POCL_REWARD          (service=1, msg=5) — Phase E: Reward Mechanism.
                        Reward/penalty record for a miner (R_i formula).

  PREDICTION_PROPOSAL  (service=1, msg=6) — Phase B: Prediction Proposal.
                        Miner submits a hash of its predictions on the
                        shared evaluation records.

  VOTE_PROPOSAL        (service=1, msg=7) — Phase C: Vote Proposal.
                        Miner casts votes on peers' prediction quality
                        (accuracy + timeliness criteria).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .crypto import sha256, ExonumWallet


# Exonum service identifiers
FL_SERVICE_ID    = 1   # Federated learning service
PBFT_SERVICE_ID  = 2   # pBFT consensus service


@dataclass
class ExonumTransaction:
    service_id:  int
    message_id:  int
    sender:      str                      # wallet address
    payload:     Dict[str, Any]
    signature:   str = ""
    tx_hash:     str = field(default="")
    timestamp:   float = field(default_factory=time.time)
    tx_type:     str = "UNKNOWN"

    def _canonical(self) -> str:
        return json.dumps({
            "service_id": self.service_id,
            "message_id": self.message_id,
            "sender":     self.sender,
            "payload":    self.payload,
            "timestamp":  round(self.timestamp, 3),
        }, sort_keys=True, separators=(",", ":"))

    def sign(self, wallet: ExonumWallet) -> "ExonumTransaction":
        canon = self._canonical()
        self.signature = wallet.sign(canon)
        self.tx_hash   = sha256(
            f"{self.service_id}{self.message_id}{self.sender}{self.signature}"
        )
        return self

    def is_valid(self) -> bool:
        return (
            len(self.tx_hash)  == 64
            and len(self.signature) == 64
            and ExonumWallet.verify(self.signature, self._canonical(), "")
        )

    def to_dict(self) -> dict:
        return {
            "tx_type":    self.tx_type,
            "service_id": self.service_id,
            "message_id": self.message_id,
            "sender":     self.sender,
            "payload":    self.payload,
            "signature":  self.signature,
            "tx_hash":    self.tx_hash,
            "timestamp":  self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_model_update_tx(wallet: ExonumWallet, trainer_id: int,
                         round_num: int, ipfs_cid: str,
                         accuracy: float, noise_level: float) -> ExonumTransaction:
    """Phase A: Model Proposal — miner submits IPFS CID + metadata."""
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=1,
        sender=wallet.address,
        payload={"trainer_id": trainer_id, "round": round_num,
                 "ipfs_cid": ipfs_cid, "accuracy": round(accuracy, 6),
                 "dp_noise": round(noise_level, 6)},
        tx_type="MODEL_UPDATE",
    )
    return tx.sign(wallet)


def make_validation_tx(wallet: ExonumWallet, validator_id: int,
                       trainer_id: int, round_num: int,
                       accepted: bool, score: float) -> ExonumTransaction:
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=2,
        sender=wallet.address,
        payload={"validator_id": validator_id, "trainer_id": trainer_id,
                 "round": round_num, "accepted": accepted,
                 "score": round(score, 6)},
        tx_type="VALIDATION",
    )
    return tx.sign(wallet)


def make_trust_update_tx(wallet: ExonumWallet, trainer_id: int,
                         old_score: float, new_score: float,
                         round_num: int) -> ExonumTransaction:
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=3,
        sender=wallet.address,
        payload={"trainer_id": trainer_id, "round": round_num,
                 "old_score": round(old_score, 6),
                 "new_score": round(new_score, 6)},
        tx_type="TRUST_UPDATE",
    )
    return tx.sign(wallet)


def make_global_model_tx(wallet: ExonumWallet, round_num: int,
                         ipfs_cid: str, accuracy: float,
                         accepted_trainers: list,
                         trust_scores: dict,
                         winners: Optional[list] = None,
                         sync_mode: Optional[str] = None) -> ExonumTransaction:
    """Phase F: Block Creation — aggregated global model record."""
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=4,
        sender=wallet.address,
        payload={"round": round_num, "ipfs_cid": ipfs_cid,
                 "accuracy": round(accuracy, 6),
                 "accepted_trainers": accepted_trainers,
                 "winners": winners or [],
                 "sync_mode": sync_mode or "",
                 "trust_scores": {str(k): round(v, 6)
                                  for k, v in trust_scores.items()}},
        tx_type="GLOBAL_MODEL",
    )
    return tx.sign(wallet)


def make_pbft_vote_tx(wallet: ExonumWallet, voter_id: int,
                      round_num: int, block_hash: str,
                      phase: str, vote: bool) -> ExonumTransaction:
    """phase: 'PREPARE' | 'COMMIT'"""
    tx = ExonumTransaction(
        service_id=PBFT_SERVICE_ID, message_id=1,
        sender=wallet.address,
        payload={"voter_id": voter_id, "round": round_num,
                 "block_hash": block_hash, "phase": phase, "vote": vote},
        tx_type="PBFT_VOTE",
    )
    return tx.sign(wallet)


def make_pocl_reward_tx(wallet: ExonumWallet, trainer_id: int,
                        reward: float, round_num: int,
                        reason: str, contribution: float = 0.0) -> ExonumTransaction:
    """Phase E: Reward Mechanism — R_i contribution-based reward/penalty."""
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=5,
        sender=wallet.address,
        payload={"trainer_id": trainer_id, "reward": round(reward, 6),
                 "round": round_num, "reason": reason,
                 "contribution_R_i": round(contribution, 8)},
        tx_type="POCL_REWARD",
    )
    return tx.sign(wallet)


def make_prediction_proposal_tx(wallet: ExonumWallet, trainer_id: int,
                                round_num: int, predictions_hash: str,
                                n_predictions: int) -> ExonumTransaction:
    """Phase B: Prediction Proposal — miner submits hash of its predictions
    on the shared evaluation record set."""
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=6,
        sender=wallet.address,
        payload={"trainer_id": trainer_id, "round": round_num,
                 "predictions_hash": predictions_hash,
                 "n_predictions": n_predictions},
        tx_type="PREDICTION_PROPOSAL",
    )
    return tx.sign(wallet)


def make_vote_proposal_tx(wallet: ExonumWallet, voter_id: int,
                          round_num: int, votes: Dict[int, float]) -> ExonumTransaction:
    """Phase C: Vote Proposal — miner votes on peers' prediction quality
    (accuracy + timeliness combined score)."""
    tx = ExonumTransaction(
        service_id=FL_SERVICE_ID, message_id=7,
        sender=wallet.address,
        payload={"voter_id": voter_id, "round": round_num,
                 "votes": {str(k): round(v, 6) for k, v in votes.items()}},
        tx_type="VOTE_PROPOSAL",
    )
    return tx.sign(wallet)
