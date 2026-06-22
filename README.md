# FLoBC — Real Blockchain Federated Learning
**Abuzied et al., Cluster Computing 2024**
DOI: [10.1007/s10586-024-04273-1](https://doi.org/10.1007/s10586-024-04273-1)

---

## What's Implemented

### Real Cryptographic Blockchain
| Component | Technology |
|-----------|-----------|
| Block hashing | SHA-256 (double-hash) |
| Transaction integrity | SHA-256 signed payload |
| Digital signatures | RSA-2048 + PKCS1v15 + SHA-256 |
| Transaction ordering | Binary Merkle Tree |
| Consensus | Proof-of-Stake (pBFT, >2/3 threshold) |
| Node identity | RSA public-key wallet address |

### Every Training Round Produces a Block Containing:
```
Block #N
├── MODEL_UPDATE  tx  (trainer 0, signed with trainer-0 wallet)
├── MODEL_UPDATE  tx  (trainer 1, signed with trainer-1 wallet)
│   ...
├── VALIDATION    tx  (validator 0 scored trainer 0, signed)
│   ...
├── TRUST_UPDATE  tx  (trainer 0 reputation updated)
│   ...
└── GLOBAL_MODEL  tx  (aggregated model hash, signed by framework)

Block hash  = SHA-256(index + prev_hash + merkle_root + PoS_votes + timestamp)
Merkle root = binary tree over all tx hashes above
PoS votes   = {validator_address: stake_weight} — accepted when >2/3 stake says YES
```

---

## Project Structure

```
flobc/
├── blockchain/
│   ├── crypto.py          ← SHA-256, Merkle Tree, RSA Wallet
│   ├── transaction.py     ← Signed transactions (4 types)
│   ├── chain.py           ← Block, ProofOfStake, RealBlockchain
│   └── __init__.py
├── core/
│   ├── flobc_engine.py    ← Full FLoBC (trainers + validators + real chain)
│   ├── data_utils.py      ← Synthetic data generators
│   └── __init__.py
├── experiments/
│   └── run_experiments.py ← All 8 paper experiments
├── dashboard/
│   └── plot_results.py    ← Matplotlib charts
├── results/               ← Auto-generated outputs
├── demo.py                ← 4-demo starter including tamper detection
├── run_all.py             ← ONE COMMAND runs everything
├── requirements.txt
└── README.md
```

---

## Step-by-Step Instructions

### Step 1 — Open terminal in project folder
```bash
cd C:\Users\SASINI\Desktop\research\flobc
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run everything with ONE command
```bash
python run_all.py
```

This automatically:
1. Checks & installs dependencies
2. Runs the demo (sanity check)
3. Runs all 8 experiments
4. Generates all charts
5. Prints a full summary

---

## Or run steps individually

```bash
# Quick demo only (~30 seconds)
python demo.py

# All 8 experiments (~90 seconds)
python experiments/run_experiments.py

# Generate charts (needs matplotlib)
python dashboard/plot_results.py
```

---

## Output Files After Running

```
results/
├── all_results.json              ← all experiment metrics
├── demo1_blockchain.json         ← full chain from demo 1
├── demo2_blockchain.json         ← full chain from demo 2
├── blockchain_exp0.json          ← exp 0 chain (30 blocks)
├── blockchain_exp2_scoring.json  ← exp 2 chain
├── blockchain_exp3_BSP.json      ← exp 3 BSP chain
├── blockchain_exp3_SSP.json
├── blockchain_exp3_BAP_1.0.json
├── blockchain_exp3_BAP_0.6.json
├── blockchain_exp4.json
├── blockchain_exp6.json
└── blockchain_exp7.json

dashboard/
├── exp0_cent_vs_dec.png
├── exp1_tv_ratio.png
├── exp2_reward_penalty.png
├── exp3_sync_schemes.png
├── exp7_vs_dispfl.png
└── exp8_vs_pvdfl.png
```

Each `blockchain_expN.json` contains the full cryptographic chain:
```json
{
  "chain_length": 31,
  "is_valid": true,
  "blocks": [
    {
      "index": 0,
      "block_hash": "000...genesis...",
      "previous_hash": "0000...64 zeros...",
      "merkle_root": "abc123...",
      "transactions": [],
      "stake_votes": {"GENESIS": 1.0}
    },
    {
      "index": 1,
      "block_hash": "f3a9...",
      "previous_hash": "000...genesis...",
      "merkle_root": "7b2c...",
      "transactions": [
        {"tx_type": "MODEL_UPDATE", "sender": "a1b2...", "tx_hash": "..."},
        {"tx_type": "VALIDATION",   "sender": "c3d4...", "tx_hash": "..."},
        {"tx_type": "TRUST_UPDATE", "sender": "e5f6...", "tx_hash": "..."},
        {"tx_type": "GLOBAL_MODEL", "sender": "g7h8...", "tx_hash": "..."}
      ],
      "stake_votes": {"a1b2...": 0.333, "c3d4...": 0.333, "e5f6...": 0.334}
    }
  ]
}
```

---

## Blockchain vs Paper

| Paper (Exonum) | This Implementation |
|----------------|---------------------|
| Rust + JavaScript | Pure Python |
| Exonum pBFT | Custom pBFT (ProofOfStake class) |
| Network HTTP nodes | In-process node objects |
| Cryptographic PoW | Proof-of-Stake (>2/3 vote) |
| Immutable ledger | Append-only list + hash links |
| RSA signatures | RSA-2048 via `cryptography` lib |
| Merkle proofs | Binary Merkle tree |

Same logical behaviour, portable Python implementation.

---

## Citation
```bibtex
@article{abuzied2024flobc,
  title   = {A privacy-preserving federated learning framework for blockchain networks},
  author  = {Abuzied, Youssif and Ghanem, Mohamed and Dawoud, Fadi and
             Gamal, Habiba and Soliman, Eslam and Sharara, Hossam and ElBatt, Tamer},
  journal = {Cluster Computing},
  year    = {2024},
  doi     = {10.1007/s10586-024-04273-1}
}
```
