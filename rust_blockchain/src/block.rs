/// block.rs — Block and Transaction data types for the FLoBC Rust blockchain.
///
/// Every FLoBC round produces one Block containing:
///   - MODEL_UPDATE transactions  (one per trainer, weight hash + ECDSA sig)
///   - VALIDATION    transactions  (one per validator per trainer)
///   - TRUST_UPDATE  transactions  (one per trainer after validation)
///   - GLOBAL_MODEL  transaction   (one per round, new global model hash)
///
/// The block is hashed as SHA-256(index || prev_hash || merkle_root ||
///   validator || timestamp) to create the tamper-evident chain link.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Transaction
// ─────────────────────────────────────────────────────────────────────────────

/// A signed, immutable record of one event in the FLoBC network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Category: MODEL_UPDATE | VALIDATION | TRUST_UPDATE | GLOBAL_MODEL
    pub tx_type:   String,
    /// ECDSA secp256k1 address (SHA-256 of compressed public key)
    pub sender:    String,
    /// Event-specific data (weights hash, score, etc.)
    pub payload:   serde_json::Value,
    /// Unix epoch timestamp (seconds, fractional)
    pub timestamp: f64,
    /// SHA-256 of (tx_type || sender || payload_json || timestamp)
    pub tx_hash:   String,
    /// ECDSA secp256k1 DER signature of tx_hash, hex-encoded
    pub signature: String,
}

impl Transaction {
    pub fn new(
        tx_type: String,
        sender: String,
        payload: serde_json::Value,
        signature: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let canonical = format!(
            "{}{}{}{}",
            tx_type,
            sender,
            serde_json::to_string(&payload).unwrap_or_default(),
            timestamp,
        );
        let tx_hash = hex::encode(Sha256::digest(canonical.as_bytes()));

        Transaction { tx_type, sender, payload, timestamp, tx_hash, signature }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block
// ─────────────────────────────────────────────────────────────────────────────

/// One block in the FLoBC chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index:         u64,
    pub previous_hash: String,
    pub transactions:  Vec<Transaction>,
    /// Merkle root of all transaction hashes — commits to tx set integrity.
    pub merkle_root:   String,
    /// Address of the validator that proposed this block.
    pub validator:     String,
    /// PoS vote record: { validator_address -> stake_weight }
    pub stake_votes:   std::collections::HashMap<String, f64>,
    pub timestamp:     f64,
    pub block_hash:    String,
}

impl Block {
    /// Build a genesis block (index=0, no transactions, prev_hash=0*64).
    pub fn genesis() -> Self {
        let mut b = Block {
            index:         0,
            previous_hash: "0".repeat(64),
            transactions:  vec![],
            merkle_root:   "0".repeat(64),
            validator:     "genesis".into(),
            stake_votes:   Default::default(),
            timestamp:     SystemTime::now()
                               .duration_since(UNIX_EPOCH)
                               .unwrap()
                               .as_secs_f64(),
            block_hash:    String::new(),
        };
        b.block_hash = b.compute_hash();
        b
    }

    /// Build a new block from a list of transactions.
    pub fn new(
        index:         u64,
        previous_hash: String,
        transactions:  Vec<Transaction>,
        validator:     String,
        stake_votes:   std::collections::HashMap<String, f64>,
    ) -> Self {
        let merkle_root = compute_merkle(&transactions);
        let timestamp   = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut b = Block {
            index, previous_hash, transactions,
            merkle_root, validator, stake_votes, timestamp,
            block_hash: String::new(),
        };
        b.block_hash = b.compute_hash();
        b
    }

    /// SHA-256 of the canonical header string.
    pub fn compute_hash(&self) -> String {
        let header = format!(
            "{}{}{}{}{}",
            self.index, self.previous_hash, self.merkle_root,
            self.validator, self.timestamp,
        );
        hex::encode(Sha256::digest(header.as_bytes()))
    }

    /// Structural integrity check:
    ///   1. block_hash matches recomputed hash
    ///   2. previous_hash matches supplied expected value
    ///   3. merkle_root matches recomputed Merkle root of transactions
    pub fn is_valid(&self, expected_prev: &str) -> bool {
        if self.previous_hash != expected_prev {
            return false;
        }
        if self.block_hash != self.compute_hash() {
            return false;
        }
        if self.merkle_root != compute_merkle(&self.transactions) {
            return false;
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Merkle root helper
// ─────────────────────────────────────────────────────────────────────────────

pub fn compute_merkle(txs: &[Transaction]) -> String {
    if txs.is_empty() {
        return "0".repeat(64);
    }
    let mut hashes: Vec<String> = txs.iter().map(|t| t.tx_hash.clone()).collect();
    while hashes.len() > 1 {
        if hashes.len() % 2 == 1 {
            hashes.push(hashes.last().unwrap().clone());
        }
        hashes = hashes
            .chunks(2)
            .map(|pair| {
                hex::encode(Sha256::digest(
                    format!("{}{}", pair[0], pair[1]).as_bytes(),
                ))
            })
            .collect();
    }
    hashes.into_iter().next().unwrap_or_else(|| "0".repeat(64))
}

// ─────────────────────────────────────────────────────────────────────────────
// Request/response DTOs (from Python via HTTP)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TxData {
    pub tx_type:   String,
    pub sender:    String,
    pub payload:   serde_json::Value,
    pub signature: String,
}

#[derive(Debug, Deserialize)]
pub struct VoteData {
    /// Validator wallet address
    pub validator: String,
    /// true = YES, false = NO
    pub vote:      bool,
    /// PoS stake weight
    pub stake:     f64,
}

#[derive(Debug, Deserialize)]
pub struct ProposeRequest {
    pub transactions: Vec<TxData>,
    pub validator:    String,
    pub votes:        Vec<VoteData>,
    /// pBFT threshold (default 2/3 = 0.6667)
    pub threshold:    f64,
}
