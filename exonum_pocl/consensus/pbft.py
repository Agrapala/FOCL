"""
pBFT Consensus Engine
======================
Practical Byzantine Fault Tolerance (Castro & Liskov, 1999) — the consensus
mechanism used inside Exonum's blockchain.

Phases (3-phase protocol):
  PRE-PREPARE : Leader broadcasts proposed block to all replicas
  PREPARE     : Each replica broadcasts PREPARE vote to all others
                Block moves to COMMIT phase when >= 2f+1 PREPARE votes
                (f = floor((n-1)/3) = max faulty nodes)
  COMMIT      : Each replica broadcasts COMMIT vote
                Block is committed when >= 2f+1 COMMIT votes received

View change: if leader is faulty and timeout expires, replicas elect a new
leader (view_number increments). Simulated here with a single-round fast path
since all nodes are honest.

Fault tolerance: tolerates up to f Byzantine faults in n = 3f+1 nodes.
With 4 validator nodes: n=4, f=1 → tolerates 1 faulty node.

Returns:
  ConsensusResult with: committed bool, phase_times dict, precommit_sigs dict,
                        pbft_round int, fault_count int.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ConsensusResult:
    committed:      bool
    consensus_name: str
    pbft_round:     int
    precommit_sigs: Dict[str, str]     # {validator_addr: commit_sig}
    phase_times:    Dict[str, float]   # timing per phase
    yes_votes:      int
    total_nodes:    int
    fault_tolerance: int               # f value
    delay_sec:      float              # total consensus time
    gas_equivalent: int                # tx count proxy for cost


class PBFTConsensus:
    """
    pBFT for n validator nodes.
    All validators are represented by their address strings + wallet objects.
    """

    def __init__(self, validator_wallets: list, timeout: float = 2.0):
        """
        validator_wallets: list of ExonumWallet objects (one per validator)
        """
        self.validators  = validator_wallets
        self.n           = len(validator_wallets)
        self.f           = (self.n - 1) // 3      # max Byzantine faults tolerated
        self.quorum      = 2 * self.f + 1          # 2f+1 needed
        self.timeout     = timeout
        self.view        = 0                        # current view number
        self._leader_idx = 0

    @property
    def leader(self):
        return self.validators[self._leader_idx % self.n]

    def _sign_phase(self, wallet, block_hash: str, phase: str) -> str:
        import hmac as _hmac
        return _hmac.new(
            wallet._privkey,
            f"{phase}:{block_hash}:{self.view}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def run(self, block_hash: str, block_txs_count: int) -> ConsensusResult:
        """
        Execute full pBFT 3-phase protocol for a proposed block.
        Returns ConsensusResult.
        """
        t_start = time.perf_counter()
        phase_times: Dict[str, float] = {}

        # ── PRE-PREPARE (leader proposes) ──────────────────────────────────
        t0 = time.perf_counter()
        leader_sig = self._sign_phase(self.leader, block_hash, "PRE-PREPARE")
        phase_times["pre_prepare"] = time.perf_counter() - t0

        # ── PREPARE (all replicas respond to leader's proposal) ────────────
        t0 = time.perf_counter()
        prepare_votes: Dict[str, str] = {}
        for val in self.validators:
            sig = self._sign_phase(val, block_hash, "PREPARE")
            prepare_votes[val.address] = sig
        phase_times["prepare"] = time.perf_counter() - t0

        # Check if prepare quorum met (>= 2f+1)
        prepare_ok = len(prepare_votes) >= self.quorum

        # ── COMMIT ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        commit_votes: Dict[str, str] = {}
        if prepare_ok:
            for val in self.validators:
                sig = self._sign_phase(val, block_hash, "COMMIT")
                commit_votes[val.address] = sig
        phase_times["commit"] = time.perf_counter() - t0

        commit_ok  = len(commit_votes) >= self.quorum
        total_time = time.perf_counter() - t_start

        # Gas equivalent: each vote = 1 tx, PRE-PREPARE = 1 tx
        gas_eq = 1 + len(prepare_votes) + len(commit_votes) + block_txs_count

        return ConsensusResult(
            committed       = commit_ok,
            consensus_name  = "pBFT",
            pbft_round      = self.view,
            precommit_sigs  = commit_votes,
            phase_times     = phase_times,
            yes_votes       = len(commit_votes),
            total_nodes     = self.n,
            fault_tolerance = self.f,
            delay_sec       = total_time,
            gas_equivalent  = gas_eq,
        )

    def trigger_view_change(self):
        """Simulate leader failure → increment view, new leader selected."""
        self.view        += 1
        self._leader_idx  = self.view % self.n
        print(f"  [pBFT] View change → view={self.view}, "
              f"new leader={self.leader.address[:12]}...")
