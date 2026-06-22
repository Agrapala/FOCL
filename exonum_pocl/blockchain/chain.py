"""
Exonum-Style Blockchain — Block & Chain
=========================================
Exonum uses a Byzantine Fault Tolerant consensus (pBFT under the hood).
Each block contains:
  - height          : Exonum term for block index
  - prev_hash       : links to previous block
  - tx_hash         : Merkle root of all transactions in this block
  - state_hash      : SHA-256 of the FL global model state
  - proposer_id     : node that proposed this block
  - round           : pBFT round number in which consensus was reached
  - precommit_sigs  : list of validator signatures on the COMMIT phase
  - time            : block creation time
  - block_hash      : SHA-256(all header fields)

This module also contains the ExonumBlockchain class which wraps the chain
and exposes propose_block() called by whichever consensus module is active.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .crypto import sha256, MerkleTree
from .transaction import ExonumTransaction


@dataclass
class ExonumBlock:
    height:         int
    prev_hash:      str
    transactions:   List[ExonumTransaction]
    proposer_id:    str                        # validator address
    round_num:      int                        # pBFT round
    precommit_sigs: Dict[str, str]             # {validator_addr: signature}
    state_hash:     str = ""                   # FL model state fingerprint
    timestamp:      float = field(default_factory=time.time)
    tx_hash:        str = field(default="")    # Merkle root
    block_hash:     str = field(default="")

    def _merkle_root(self) -> str:
        return MerkleTree([tx.tx_hash for tx in self.transactions]).root

    def _header_str(self) -> str:
        return json.dumps({
            "height":         self.height,
            "prev_hash":      self.prev_hash,
            "tx_hash":        self.tx_hash,
            "state_hash":     self.state_hash,
            "proposer_id":    self.proposer_id,
            "round_num":      self.round_num,
            "precommit_sigs": self.precommit_sigs,
            "timestamp":      round(self.timestamp, 3),
        }, sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        return sha256(self._header_str())

    def finalise(self, state_hash: str = "") -> "ExonumBlock":
        self.state_hash = state_hash
        self.tx_hash    = self._merkle_root()
        self.block_hash = self.compute_hash()
        return self

    def is_valid(self, prev_block_hash: str) -> bool:
        if self.prev_hash != prev_block_hash:
            return False
        if self.tx_hash != self._merkle_root():
            return False
        if self.block_hash != self.compute_hash():
            return False
        for tx in self.transactions:
            if not tx.is_valid():
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "height":         self.height,
            "prev_hash":      self.prev_hash,
            "tx_hash":        self.tx_hash,
            "state_hash":     self.state_hash,
            "proposer_id":    self.proposer_id,
            "round_num":      self.round_num,
            "precommit_sigs": self.precommit_sigs,
            "timestamp":      self.timestamp,
            "block_hash":     self.block_hash,
            "transactions":   [tx.to_dict() for tx in self.transactions],
        }

    def summary(self) -> str:
        return (f"Block h={self.height} | "
                f"hash={self.block_hash[:12]}... | "
                f"txs={len(self.transactions)} | "
                f"proposer={self.proposer_id[:10]}...")


class ExonumBlockchain:
    """
    Simulated Exonum blockchain.
    Blocks are committed by whichever consensus engine calls propose_block().
    The chain only records finalised, consensus-approved blocks.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._chain: List[ExonumBlock] = []
        self._create_genesis()

    def _create_genesis(self):
        genesis = ExonumBlock(
            height=0,
            prev_hash=self.GENESIS_HASH,
            transactions=[],
            proposer_id="GENESIS",
            round_num=0,
            precommit_sigs={"GENESIS": "0" * 64},
            timestamp=0.0,
        ).finalise(state_hash="genesis_state")
        self._chain.append(genesis)
        print(f"  [Exonum] Genesis block | hash={genesis.block_hash[:16]}...")

    def propose_block(self,
                      transactions: List[ExonumTransaction],
                      proposer_id: str,
                      round_num: int,
                      precommit_sigs: Dict[str, str],
                      state_hash: str = "") -> Optional[ExonumBlock]:
        """Called by consensus engine once COMMIT phase succeeds."""
        if not transactions:
            return None
        block = ExonumBlock(
            height=len(self._chain),
            prev_hash=self._chain[-1].block_hash,
            transactions=transactions,
            proposer_id=proposer_id,
            round_num=round_num,
            precommit_sigs=precommit_sigs,
        ).finalise(state_hash=state_hash)
        self._chain.append(block)
        return block

    def latest_block(self) -> ExonumBlock:
        return self._chain[-1]

    def get_block(self, height: int) -> Optional[ExonumBlock]:
        if 0 <= height < len(self._chain):
            return self._chain[height]
        return None

    def get_chain(self) -> List[ExonumBlock]:
        return list(self._chain)

    def length(self) -> int:
        return len(self._chain)

    def is_chain_valid(self) -> bool:
        for i in range(1, len(self._chain)):
            if not self._chain[i].is_valid(self._chain[i - 1].block_hash):
                return False
        return True

    def export_json(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "blockchain":   "Exonum (simulated)",
            "chain_length": len(self._chain),
            "is_valid":     self.is_chain_valid(),
            "blocks":       [b.to_dict() for b in self._chain],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def print_chain(self, max_blocks: int = 10):
        print(f"\n  {'═'*66}")
        print(f"  EXONUM BLOCKCHAIN | height={len(self._chain)} | "
              f"valid={self.is_chain_valid()}")
        print(f"  {'═'*66}")
        for blk in self._chain[:max_blocks]:
            print(f"  h={blk.height:3d} | {blk.block_hash[:14]}... | "
                  f"prev={blk.prev_hash[:14]}... | "
                  f"txs={len(blk.transactions):2d} | "
                  f"merkle={blk.tx_hash[:10]}...")
        if len(self._chain) > max_blocks:
            print(f"  ... ({len(self._chain)-max_blocks} more blocks)")
        print(f"  {'═'*66}\n")
