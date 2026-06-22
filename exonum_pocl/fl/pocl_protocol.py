"""
PoCL Protocol — 5-Phase Proof-of-Collaborative-Learning (Paper Section 3.4)
=============================================================================
"The PoCL mechanism serves as the decentralized consensus layer of the BC-FL
system, ensuring secure, fair, and performance-based validation of model
updates. Each FL round proceeds through the following stages:"

  A. Model Proposal Phase
     A global model is distributed to miners. Each miner trains it on local
     data and forms a model proposal block (IPFS CID + metadata). A proposal
     deadline (Administrator-defined) ensures timely participation; late
     submissions are handled by the active SyncManager (SP/SSP/BAP).
     -> implemented in fl/roles.py: Miner.train_and_propose()

  B. Prediction Proposal Phase
     Submitted test records are distributed among miners. Each miner runs
     inference using its locally trained model and submits predictions
     (hashed) to the PoCL smart contract within a prediction deadline.
     -> implemented in fl/roles.py: Miner.predict_on_shared_records()
     -> aggregated here in PoCLProtocol.collect_predictions()

  C. Vote Proposal Phase
     Miners evaluate peers' prediction results using accuracy and
     time-of-submission criteria and cast votes. Early, accurate
     submissions are prioritized.
     -> implemented in fl/roles.py: Miner.vote_on_peers()
     -> aggregated here in PoCLProtocol.aggregate_votes()

  D. Winner Selection Phase
     The PoCL chaincode aggregates validator scores + peer votes + raw
     accuracy and selects the top-K miners as winners. Winning models are
     verified against their original IPFS CIDs before aggregation.
     -> implemented here in PoCLProtocol.select_winners()

  E. Reward Mechanism
     Reward function measures each miner's contribution to the new global
     model based on weight differences (R_i, see fl/roles.py Aggregator).
     Rewards/penalties are distributed via the (simulated) PoCL smart
     contract; malicious/low-quality updates receive penalties.
     -> implemented here in PoCLProtocol.compute_rewards()

  F. Block Creation
     The (simulated) smart contract compiles validated transactions and the
     winning/aggregated model into a new block, appended to the Exonum
     blockchain via the active consensus engine (pBFT / PoCL-pBFT / PoS).
     -> implemented in fl/engine.py + blockchain/chain.py
"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PoCLPhaseResult:
    round_num:        int
    val_accuracy:     Dict[int, float]            # Phase A: per-miner val accuracy
    validator_scores: Dict[int, float]            # validator-quorum mean score
    peer_votes:       Dict[int, Dict[int, float]] # Phase C: voter -> {target: score}
    aggregated_votes: Dict[int, float]            # Phase C: mean vote per target
    combined_scores:  Dict[int, float]            # Phase D: ranking scores
    winners:          List[int]                   # Phase D: top-K winners
    rewards:          Dict[int, float]            # Phase E: reward/penalty


class PoCLProtocol:
    """
    Orchestrates Phases B-E of the PoCL mechanism for a single FL round.
    Phase A is performed by Miners directly (fl/roles.py); Phase F is
    performed by fl/engine.py + the blockchain/consensus layer.
    """

    def __init__(self, top_k: int = 2,
                 w_validator: float = 0.5,
                 w_peer_vote: float = 0.3,
                 w_accuracy:  float = 0.2,
                 base_reward: float = 0.08,
                 penalty:     float = -0.08):
        self.top_k       = top_k
        self.w_validator = w_validator
        self.w_peer_vote = w_peer_vote
        self.w_accuracy  = w_accuracy
        self.base_reward = base_reward
        self.penalty     = penalty

    # ── Phase B: Prediction Proposal ─────────────────────────────────────────

    @staticmethod
    def predictions_hash(preds: np.ndarray) -> str:
        """Hash a miner's prediction array for on-chain submission
        (Phase B prevents reverse-engineering test labels by only
        revealing a hash; full predictions are used locally for voting)."""
        return hashlib.sha256(preds.astype(np.int8).tobytes()).hexdigest()

    def collect_predictions(self, miners: dict, eligible_ids: List[int],
                            X_eval: np.ndarray) -> Dict[int, np.ndarray]:
        """Each eligible miner runs inference on the shared evaluation
        record set."""
        return {tid: miners[tid].predict_on_shared_records(X_eval)
                for tid in eligible_ids}

    # ── Phase C: Vote Proposal ───────────────────────────────────────────────

    def collect_votes(self, miners: dict, eligible_ids: List[int],
                      predictions: Dict[int, np.ndarray],
                      y_eval: np.ndarray,
                      latencies: Dict[int, float]) -> Dict[int, Dict[int, float]]:
        """Every eligible miner votes on every other eligible miner's
        predictions (accuracy + timeliness)."""
        peer_votes: Dict[int, Dict[int, float]] = {}
        for voter_id in eligible_ids:
            peer_votes[voter_id] = miners[voter_id].vote_on_peers(
                predictions, y_eval, latencies)
        return peer_votes

    def aggregate_votes(self, peer_votes: Dict[int, Dict[int, float]]) -> Dict[int, float]:
        """Average all peer votes received by each target miner."""
        agg: Dict[int, List[float]] = {}
        for _voter, votes in peer_votes.items():
            for target, score in votes.items():
                agg.setdefault(target, []).append(score)
        return {tid: float(np.mean(scores)) for tid, scores in agg.items()}

    # ── Phase D: Winner Selection ─────────────────────────────────────────────

    def select_winners(self, eligible_ids: List[int],
                       val_accuracy: Dict[int, float],
                       validator_scores: Dict[int, float],
                       peer_vote_scores: Dict[int, float]
                       ) -> Tuple[List[int], Dict[int, float]]:
        """
        combined_score(i) = w_validator * validator_score(i)
                           + w_peer_vote * peer_vote_score(i)
                           + w_accuracy  * val_accuracy(i)

        Top-K (Administrator-configured) miners by combined_score become
        the PoCL "winners" — their models are verified against their
        original IPFS CIDs (handled in fl/engine.py) and integrated by
        the Aggregator via FedAvg.
        """
        combined: Dict[int, float] = {}
        for tid in eligible_ids:
            combined[tid] = (
                self.w_validator * validator_scores.get(tid, 0.0)
                + self.w_peer_vote * peer_vote_scores.get(tid, 0.0)
                + self.w_accuracy  * val_accuracy.get(tid, 0.0)
            )
        ranked = sorted(eligible_ids, key=lambda t: combined[t], reverse=True)
        k = max(1, min(self.top_k, len(ranked)))
        return ranked[:k], combined

    # ── Phase E: Reward Mechanism ─────────────────────────────────────────────

    def compute_rewards(self, eligible_ids: List[int], winners: List[int],
                        contribution_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Winners receive base_reward scaled by their contribution R_i
        (higher contribution -> larger reward). Non-winning (but still
        eligible/accepted) miners and rejected miners receive `penalty`.
        """
        rewards: Dict[int, float] = {}
        for tid in eligible_ids:
            if tid in winners:
                contrib    = contribution_scores.get(tid, 0.0)
                rewards[tid] = self.base_reward * (1.0 + contrib)
            else:
                rewards[tid] = self.penalty
        return rewards

    def info(self) -> dict:
        return {
            "top_k":       self.top_k,
            "w_validator": self.w_validator,
            "w_peer_vote": self.w_peer_vote,
            "w_accuracy":  self.w_accuracy,
            "base_reward": self.base_reward,
            "penalty":     self.penalty,
        }
