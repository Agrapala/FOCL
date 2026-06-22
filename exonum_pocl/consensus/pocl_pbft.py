"""
PoCL-pBFT Consensus — Proof-of-Collaborative-Learning variant of pBFT
=======================================================================
Extension of pBFT where:
  - PREPARE votes are weighted by the validator's trust/reputation score
  - A validator with low trust score has reduced vote weight
  - Quorum is still 2f+1 but by WEIGHTED votes (sum of trust weights > 2/3)
  - Winners in PoCL (top-K accuracy miners) are written into the block header
    and their trust boost is part of the consensus decision

PoCL Phases (extends pBFT):
  MODEL PROPOSAL  : miners submit IPFS CIDs of locally trained models
  PREDICTION      : validators run inference with each submitted model
  VOTE            : validators vote on best models (accuracy + timing)
  WINNER SELECT   : top-K models selected, FedAvg aggregation triggered
  REWARD          : smart contract (simulated) distributes rewards/penalties

This module only implements the consensus voting logic; the FL training loop
in fl/engine.py calls this after training is done each round.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .pbft import ConsensusResult


@dataclass
class PoCLRound:
    round_num:     int
    miner_scores:  Dict[int, float]    # {trainer_id: accuracy on val set}
    miner_timing:  Dict[int, float]    # {trainer_id: submission latency}
    top_k:         int = 2


class PoCLPBFTConsensus:
    """
    PoCL-pBFT: combines pBFT voting with PoCL quality-driven winner selection.

    Trust-weighted voting:
      Each validator v has weight w_v = trust_score_v / sum(trust_scores)
      Prepare quorum: sum(w_v for voting validators) > 2/3
      Commit quorum:  same threshold
    """

    def __init__(self, validator_wallets: list,
                 validator_trust: Dict[str, float],
                 top_k: int = 2):
        self.validators       = validator_wallets
        self.validator_trust  = validator_trust  # {address: trust_weight}
        self.n                = len(validator_wallets)
        self.f                = (self.n - 1) // 3
        self.top_k            = top_k
        self.view             = 0

    @property
    def leader(self):
        # Leader = validator with highest trust score
        best_addr = max(self.validator_trust,
                        key=lambda a: self.validator_trust[a],
                        default=self.validators[0].address)
        for v in self.validators:
            if v.address == best_addr:
                return v
        return self.validators[0]

    def _weighted_quorum_met(self, voters: List[str]) -> bool:
        total = sum(self.validator_trust.values()) or 1e-9
        yes_w = sum(self.validator_trust.get(a, 0.0) for a in voters)
        return yes_w > total * (2 / 3)

    def _sign(self, wallet, block_hash: str, phase: str) -> str:
        import hmac as _hmac
        return _hmac.new(
            wallet._privkey,
            f"POCL:{phase}:{block_hash}:{self.view}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def select_winners(self, pocl_round: PoCLRound) -> List[int]:
        """
        Select top-K miners by combined score:
          score = 0.7 * accuracy + 0.3 * (1 / (1 + latency))
        """
        combined: Dict[int, float] = {}
        max_lat = max(pocl_round.miner_timing.values(), default=1.0) or 1.0
        for tid, acc in pocl_round.miner_scores.items():
            lat = pocl_round.miner_timing.get(tid, max_lat)
            combined[tid] = 0.7 * acc + 0.3 * (1.0 / (1.0 + lat / max_lat))
        sorted_miners = sorted(combined, key=lambda t: combined[t], reverse=True)
        return sorted_miners[:pocl_round.top_k]

    def run(self, block_hash: str, block_txs_count: int,
            pocl_round: PoCLRound) -> Tuple[ConsensusResult, List[int]]:
        """
        Returns (ConsensusResult, winning_trainer_ids).
        """
        t_start = time.perf_counter()
        phase_times: Dict[str, float] = {}

        # ── MODEL PROPOSAL / PRE-PREPARE ──────────────────────────────────
        t0 = time.perf_counter()
        winners = self.select_winners(pocl_round)
        phase_times["pocl_winner_select"] = time.perf_counter() - t0

        # ── PREPARE (trust-weighted) ───────────────────────────────────────
        t0 = time.perf_counter()
        prepare_votes: Dict[str, str] = {}
        for val in self.validators:
            sig = self._sign(val, block_hash, "PREPARE")
            prepare_votes[val.address] = sig
        phase_times["prepare"] = time.perf_counter() - t0
        prepare_ok = self._weighted_quorum_met(list(prepare_votes.keys()))

        # ── COMMIT ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        commit_votes: Dict[str, str] = {}
        if prepare_ok:
            for val in self.validators:
                sig = self._sign(val, block_hash, "COMMIT")
                commit_votes[val.address] = sig
        phase_times["commit"] = time.perf_counter() - t0

        commit_ok  = self._weighted_quorum_met(list(commit_votes.keys()))
        total_time = time.perf_counter() - t_start

        # Gas: pBFT msgs + PoCL votes + block txs
        gas_eq = (1 + len(prepare_votes) + len(commit_votes)
                  + block_txs_count + len(pocl_round.miner_scores))

        result = ConsensusResult(
            committed       = commit_ok,
            consensus_name  = "PoCL-pBFT",
            pbft_round      = self.view,
            precommit_sigs  = commit_votes,
            phase_times     = phase_times,
            yes_votes       = len(commit_votes),
            total_nodes     = self.n,
            fault_tolerance = self.f,
            delay_sec       = total_time,
            gas_equivalent  = gas_eq,
        )
        return result, winners


class PoSConsensus:
    """
    Proof-of-Stake consensus for comparison.
    Validator with highest stake proposes; others vote YES proportional to stake.
    Quorum: cumulative stake > 2/3.
    """

    def __init__(self, validator_wallets: list,
                 stakes: Dict[str, float]):
        self.validators = validator_wallets
        self.stakes     = stakes   # {address: stake_weight}
        self.n          = len(validator_wallets)
        self.f          = (self.n - 1) // 3

    def run(self, block_hash: str, block_txs_count: int) -> ConsensusResult:
        import hmac as _hmac
        t_start = time.perf_counter()

        total_stake = sum(self.stakes.values()) or 1.0
        commit_votes: Dict[str, str] = {}
        for val in self.validators:
            sig = _hmac.new(
                val._privkey,
                f"POS:{block_hash}".encode(),
                hashlib.sha256,
            ).hexdigest()
            commit_votes[val.address] = sig

        yes_stake = sum(self.stakes.get(a, 0) for a in commit_votes)
        committed = yes_stake > total_stake * (2 / 3)
        delay     = time.perf_counter() - t_start
        gas_eq    = len(commit_votes) + block_txs_count

        return ConsensusResult(
            committed       = committed,
            consensus_name  = "PoS",
            pbft_round      = 0,
            precommit_sigs  = commit_votes,
            phase_times     = {"vote": delay},
            yes_votes       = len(commit_votes),
            total_nodes     = self.n,
            fault_tolerance = self.f,
            delay_sec       = delay,
            gas_equivalent  = gas_eq,
        )
