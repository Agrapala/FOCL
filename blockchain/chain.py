"""
FLoBC Real Blockchain — Block & Chain
======================================
Each Block contains:
  - index          : position in the chain (0 = genesis)
  - previous_hash  : links to parent block (tamper-evident)
  - timestamp      : Unix epoch
  - transactions   : list of signed Transaction objects
  - merkle_root    : Merkle root of all transaction hashes
  - validator      : address of the validator that proposed this block
  - stake_votes    : dict {validator_address: stake_weight} — PoS votes
  - nonce          : incremented until block_hash meets difficulty (light PoW fallback)
  - block_hash     : SHA-256 of all the above fields

Chain invariants:
  - block[i].previous_hash == block[i-1].block_hash
  - all transactions pass is_valid()
  - block_hash is correct
  - stake_votes represent > 2/3 of total stake (pBFT / PoS)
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from blockchain.crypto  import sha256, MerkleTree
from blockchain.transaction import Transaction


# ─────────────────────────────────────────────────────────────────────────────
# Block
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    index:         int
    previous_hash: str
    transactions:  List[Transaction]
    validator:     str                         # proposing validator address
    stake_votes:   Dict[str, float]            # {addr: stake} votes
    timestamp:     float = field(default_factory=time.time)
    nonce:         int   = 0
    block_hash:    str   = field(default="")
    merkle_root:   str   = field(default="")

    def _compute_merkle(self) -> str:
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        return MerkleTree(tx_hashes).root

    def _header_string(self) -> str:
        """Canonical string of all fields except block_hash (for hashing)."""
        return json.dumps({
            "index":         self.index,
            "previous_hash": self.previous_hash,
            "merkle_root":   self.merkle_root,
            "validator":     self.validator,
            "stake_votes":   self.stake_votes,
            "timestamp":     self.timestamp,
            "nonce":         self.nonce,
        }, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        return sha256(self._header_string())

    def finalise(self) -> "Block":
        """Set merkle_root then block_hash. Call before committing."""
        self.merkle_root = self._compute_merkle()
        self.block_hash  = self.compute_hash()
        return self

    def is_valid(self, previous_hash: str) -> bool:
        """Verify structural integrity."""
        if self.previous_hash != previous_hash:
            return False
        if self.block_hash != self.compute_hash():
            return False
        if self.merkle_root != self._compute_merkle():
            return False
        for tx in self.transactions:
            if not tx.is_valid():
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "index":         self.index,
            "previous_hash": self.previous_hash,
            "merkle_root":   self.merkle_root,
            "validator":     self.validator,
            "stake_votes":   self.stake_votes,
            "timestamp":     self.timestamp,
            "nonce":         self.nonce,
            "block_hash":    self.block_hash,
            "transactions":  [tx.to_dict() for tx in self.transactions],
        }

    def summary(self) -> str:
        return (f"Block #{self.index} | "
                f"hash={self.block_hash[:12]}... | "
                f"txs={len(self.transactions)} | "
                f"validator={self.validator[:12]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Proof-of-Stake Consensus
# ─────────────────────────────────────────────────────────────────────────────

class ProofOfStake:
    """
    Lightweight PoS matching the paper's pBFT-style voting.

    Each validator node holds a 'stake' (derived from its trust score).
    A block is valid when validators holding > 2/3 of total stake have
    voted YES — identical to the paper's Byzantine fault-tolerance threshold.

    No energy-intensive mining. Validator selection is deterministic:
      proposer = validator with highest stake in current round.
    """

    def __init__(self, validators: Dict[str, float]):
        """validators: {address: stake_weight}"""
        self.validators = dict(validators)

    def total_stake(self) -> float:
        return sum(self.validators.values()) or 1.0

    def select_proposer(self) -> str:
        """Validator with highest stake proposes the next block."""
        return max(self.validators, key=lambda a: self.validators[a])

    def vote(self, voter_address: str, agree: bool,
             votes: Dict[str, float]) -> Dict[str, float]:
        """Record a YES vote by adding the voter's stake."""
        if agree and voter_address in self.validators:
            votes[voter_address] = self.validators[voter_address]
        return votes

    def has_consensus(self, votes: Dict[str, float]) -> bool:
        """True when YES votes exceed 2/3 of total stake (pBFT threshold)."""
        yes_stake = sum(votes.values())
        return yes_stake > self.total_stake() * (2 / 3)

    def update_stake(self, address: str, new_stake: float):
        """Adjust a validator's stake (called after trust-score updates)."""
        if address in self.validators:
            self.validators[address] = max(0.001, new_stake)


# ─────────────────────────────────────────────────────────────────────────────
# Blockchain
# ─────────────────────────────────────────────────────────────────────────────

class RealBlockchain:
    """
    Cryptographically linked chain of Blocks with PoS consensus.

    Public API
    ----------
    propose_block(transactions, validator_address, pos)
        → Build candidate block, run PoS vote, commit if consensus reached.

    get_chain()         → full list of Block objects
    get_block(index)    → Block at given index
    is_chain_valid()    → verify entire chain integrity
    export_json(path)   → write chain to JSON file
    print_chain()       → pretty-print all blocks
    """

    GENESIS_HASH = "0" * 64   # sentinel previous_hash for genesis block

    def __init__(self):
        self._chain: List[Block] = []
        self._pending: List[Transaction] = []
        self._create_genesis()

    # ── Genesis ────────────────────────────────────────────────────────────

    def _create_genesis(self):
        genesis = Block(
            index=0,
            previous_hash=self.GENESIS_HASH,
            transactions=[],
            validator="GENESIS",
            stake_votes={"GENESIS": 1.0},
            timestamp=0.0,
        ).finalise()
        self._chain.append(genesis)
        print(f"  [Blockchain] Genesis block created | "
              f"hash={genesis.block_hash[:16]}...")

    # ── Transaction pool ───────────────────────────────────────────────────

    def add_transaction(self, tx: Transaction):
        """Add a verified transaction to the pending pool."""
        if not tx.is_valid():
            raise ValueError(f"Invalid transaction: {tx.tx_hash}")
        self._pending.append(tx)

    def pending_count(self) -> int:
        return len(self._pending)

    # ── Block proposal & PoS consensus ─────────────────────────────────────

    def propose_block(self,
                      transactions: List[Transaction],
                      validator_address: str,
                      pos: ProofOfStake,
                      all_validator_addresses: List[str]) -> Optional[Block]:
        """
        1. Build candidate block from transactions.
        2. Each validator votes YES (simulated — all honest in this network).
        3. If >2/3 stake says YES → finalise & commit.
        Returns the committed Block, or None if consensus failed.
        """
        if not transactions:
            return None

        candidate = Block(
            index=len(self._chain),
            previous_hash=self._chain[-1].block_hash,
            transactions=transactions,
            validator=validator_address,
            stake_votes={},
        )
        candidate.finalise()

        # PoS voting round
        votes: Dict[str, float] = {}
        for addr in all_validator_addresses:
            # Each validator independently checks the block
            block_ok = candidate.is_valid(self._chain[-1].block_hash)
            votes = pos.vote(addr, block_ok, votes)

        if pos.has_consensus(votes):
            candidate.stake_votes = votes
            # Re-finalise with votes included
            candidate.block_hash = candidate.compute_hash()
            self._chain.append(candidate)
            self._pending.clear()
            return candidate
        else:
            print(f"  [Blockchain] ! Consensus FAILED for block #{candidate.index}")
            return None

    # ── Query ──────────────────────────────────────────────────────────────

    def latest_block(self) -> Block:
        return self._chain[-1]

    def get_block(self, index: int) -> Optional[Block]:
        if 0 <= index < len(self._chain):
            return self._chain[index]
        return None

    def get_chain(self) -> List[Block]:
        return list(self._chain)

    def length(self) -> int:
        return len(self._chain)

    # ── Validation ─────────────────────────────────────────────────────────

    def is_chain_valid(self) -> bool:
        """Walk the entire chain verifying every block and link."""
        for i in range(1, len(self._chain)):
            curr = self._chain[i]
            prev = self._chain[i - 1]
            if not curr.is_valid(prev.block_hash):
                print(f"  [Blockchain] X Block #{i} INVALID")
                return False
        return True

    # ── Export ─────────────────────────────────────────────────────────────

    def export_json(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "chain_length": len(self._chain),
            "is_valid":     self.is_chain_valid(),
            "blocks":       [b.to_dict() for b in self._chain],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Pretty print ───────────────────────────────────────────────────────

    def print_chain(self, max_blocks: int = 10):
        print(f"\n  {'='*62}")
        print(f"  BLOCKCHAIN  |  length={len(self._chain)}  "
              f"|  valid={self.is_chain_valid()}")
        print(f"  {'='*62}")
        show = self._chain[:max_blocks]
        for blk in show:
            print(f"  Block #{blk.index:3d} | "
                  f"hash={blk.block_hash[:14]}... | "
                  f"prev={blk.previous_hash[:14]}... | "
                  f"txs={len(blk.transactions):2d} | "
                  f"merkle={blk.merkle_root[:10]}...")
            for tx in blk.transactions:
                print(f"          -> [{tx.tx_type:<14}] "
                      f"sender={tx.sender[:10]}... "
                      f"hash={tx.tx_hash[:10]}...")
        if len(self._chain) > max_blocks:
            print(f"  ... ({len(self._chain) - max_blocks} more blocks)")
        print(f"  {'='*62}\n")
