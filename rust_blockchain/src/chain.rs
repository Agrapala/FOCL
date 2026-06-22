/// chain.rs — Cryptographically linked blockchain for FLoBC.
///
/// The chain is a vector of Blocks where each block's previous_hash
/// equals the block_hash of its predecessor, forming a tamper-evident
/// linked list.  Any modification to a historical block breaks all
/// subsequent hash links — detected instantly by is_valid().

use crate::block::{Block, Transaction, TxData};
use std::collections::HashMap;

pub struct Blockchain {
    pub blocks: Vec<Block>,
}

impl Blockchain {
    /// Create a new chain with only the genesis block.
    pub fn new() -> Self {
        Blockchain {
            blocks: vec![Block::genesis()],
        }
    }

    /// Number of blocks in the chain (including genesis).
    pub fn length(&self) -> usize {
        self.blocks.len()
    }

    /// Append a committed block to the chain.
    /// Transactions come in as `TxData` DTOs and are converted to `Transaction`.
    pub fn add_block(
        &mut self,
        tx_data:    Vec<TxData>,
        validator:  String,
        stake_votes: HashMap<String, f64>,
    ) -> &Block {
        let prev_hash = self.blocks.last().unwrap().block_hash.clone();
        let index     = self.blocks.len() as u64;

        let transactions: Vec<Transaction> = tx_data
            .into_iter()
            .map(|d| Transaction::new(d.tx_type, d.sender, d.payload, d.signature))
            .collect();

        self.blocks.push(Block::new(
            index, prev_hash, transactions, validator, stake_votes,
        ));
        self.blocks.last().unwrap()
    }

    /// Walk the entire chain and verify every block and hash link.
    /// Returns true only if ALL blocks are structurally valid.
    pub fn is_valid(&self) -> bool {
        for i in 1..self.blocks.len() {
            let curr = &self.blocks[i];
            let prev = &self.blocks[i - 1];
            if !curr.is_valid(&prev.block_hash) {
                return false;
            }
        }
        true
    }

    /// Get a block by index; returns None if out of range.
    pub fn get_block(&self, index: usize) -> Option<&Block> {
        self.blocks.get(index)
    }

    /// Serialise the entire chain to a JSON-compatible structure.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "chain_length": self.length(),
            "is_valid":     self.is_valid(),
            "blocks": self.blocks.iter().map(|b| {
                serde_json::json!({
                    "index":         b.index,
                    "previous_hash": b.previous_hash,
                    "block_hash":    b.block_hash,
                    "merkle_root":   b.merkle_root,
                    "validator":     b.validator,
                    "timestamp":     b.timestamp,
                    "tx_count":      b.transactions.len(),
                    "transactions":  b.transactions,
                    "stake_votes":   b.stake_votes,
                })
            }).collect::<Vec<_>>(),
        })
    }
}
