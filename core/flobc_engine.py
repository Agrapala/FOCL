"""
FLoBC Core Engine  (Real Blockchain Edition)
=============================================
Privacy-Preserving Federated Learning Framework for Blockchain Networks
Based on: Abuzied et al., Cluster Computing, 2024
DOI: 10.1007/s10586-024-04273-1

What changed from the simulated version:
  - BlockchainService is REPLACED by RealBlockchain (blockchain/chain.py)
  - Every trainer owns a Wallet → signs ModelUpdate transactions
  - Every validator owns a Wallet → signs Validation transactions
  - A ProofOfStake object manages validator stakes
  - Each training round commits a real cryptographic block:
      [MODEL_UPDATE txs] + [VALIDATION txs] + [TRUST_UPDATE txs]
      + [GLOBAL_MODEL tx]  →  Block(merkle_root, prev_hash, PoS votes)
  - The full chain can be exported to results/blockchain.json
"""

import numpy as np
import hashlib
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Real blockchain imports
from blockchain.crypto       import sha256_bytes, Wallet
from blockchain.transaction  import (Transaction,
                                      make_model_update_tx,
                                      make_validation_tx,
                                      make_trust_update_tx,
                                      make_global_model_tx)
from blockchain.chain        import RealBlockchain, ProofOfStake


# ─────────────────────────────────────────────────────────────────────────────
# Enums & lightweight data classes (ML layer — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class SyncScheme(Enum):
    BSP = "Bulk Synchronous Parallel"
    SSP = "Stale Synchronous Parallel"
    BAP = "Barrierless Asynchronous Parallel"


@dataclass
class ModelUpdate:
    trainer_id:  int
    weights:     np.ndarray
    round_num:   int
    noise_level: float = 0.0
    timestamp:   float = field(default_factory=time.time)
    weights_hash: str  = ""        # SHA-256 of weights bytes (for blockchain tx)


@dataclass
class ValidationResult:
    trainer_id:  int
    validator_id: int
    accepted:    bool
    score:       float
    round_num:   int


# ─────────────────────────────────────────────────────────────────────────────
# SimpleModel  (pure NumPy — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleModel:
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (np.random.randn(input_dim, hidden_dim) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, output_dim) * s2).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def unflatten(self, flat: np.ndarray):
        i = 0
        s = self.W1.size; self.W1 = flat[i:i+s].reshape(self.W1.shape); i += s
        s = self.b1.size; self.b1 = flat[i:i+s];                        i += s
        s = self.W2.size; self.W2 = flat[i:i+s].reshape(self.W2.shape); i += s
        self.b2 = flat[i:]

    def _softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        return self._softmax(np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2)

    def predict(self, X): return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y): return float(np.mean(self.predict(X) == y))

    def sgd_step(self, X, y, lr=0.05):
        n = len(X)
        h = np.maximum(0, X @ self.W1 + self.b1)
        p = self._softmax(h @ self.W2 + self.b2)
        dL = p.copy(); dL[np.arange(n), y] -= 1; dL /= n
        dW2 = h.T @ dL; db2 = dL.sum(0)
        dh = (dL @ self.W2.T) * (h > 0)
        dW1 = X.T @ dh;  db1 = dh.sum(0)
        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        return self.flatten()

    def clone(self):
        m = SimpleModel.__new__(SimpleModel)
        m.W1=self.W1.copy(); m.b1=self.b1.copy()
        m.W2=self.W2.copy(); m.b2=self.b2.copy()
        return m


def _weights_hash(weights: np.ndarray) -> str:
    """SHA-256 fingerprint of model weights array."""
    return sha256_bytes(weights.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Reputation / Trust-Score Service  (unchanged logic, now also writes txs)
# ─────────────────────────────────────────────────────────────────────────────

class ReputationService:
    def __init__(self, trainer_ids: List[int]):
        n = len(trainer_ids)
        self.scores: Dict[int, float] = {t: 1.0/n for t in trainer_ids}
        self.history: Dict[int, List[float]] = {t: [1.0/n] for t in trainer_ids}
        self._lock = threading.Lock()

    def update(self, tid: int, improved: bool,
               delta: float = 0.06) -> Tuple[float, float]:
        """Returns (old_score, new_score) for transaction logging."""
        with self._lock:
            old = self.scores[tid]
            if improved:
                self.scores[tid] = min(1.0, old + delta)
            else:
                self.scores[tid] = max(0.0, old - delta * 2.0)
            total = sum(self.scores.values()) or 1e-9
            for k in self.scores:
                self.scores[k] /= total
            for tid2, s in self.scores.items():
                self.history[tid2].append(s)
            return old, self.scores[tid]

    def weight(self, tid: int) -> float:
        return self.scores.get(tid, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer Node  (now owns a Wallet)
# ─────────────────────────────────────────────────────────────────────────────

class TrainerNode:
    def __init__(self, tid: int, X, y, global_model: SimpleModel,
                 noise_std: float = 0.0):
        self.tid        = tid
        self.X          = X
        self.y          = y
        self.model      = global_model.clone()
        self.noise_std  = noise_std
        self.wallet     = Wallet()             # ← real RSA/HMAC key-pair

    def pull(self, global_model: SimpleModel):
        self.model = global_model.clone()

    def train_step(self, lr=0.05, batch_size=64) -> ModelUpdate:
        idx = np.random.choice(len(self.X), min(batch_size, len(self.X)), replace=False)
        weights = self.model.sgd_step(self.X[idx], self.y[idx], lr=lr)
        if self.noise_std > 0:
            weights += np.random.normal(0, self.noise_std,
                                        weights.shape).astype(np.float32)
        wh = _weights_hash(weights)
        return ModelUpdate(trainer_id=self.tid, weights=weights,
                           round_num=0, noise_level=self.noise_std,
                           weights_hash=wh)


# ─────────────────────────────────────────────────────────────────────────────
# Validator Node  (now owns a Wallet + is a PoS voter)
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorNode:
    def __init__(self, vid: int, X_val, y_val,
                 global_model: SimpleModel, threshold=0.45):
        self.vid        = vid
        self.X_val      = X_val
        self.y_val      = y_val
        self.global_mod = global_model.clone()
        self.threshold  = threshold
        self.wallet     = Wallet()             # ← real RSA/HMAC key-pair

    def validate(self, update: ModelUpdate, round_num: int) -> ValidationResult:
        cand = self.global_mod.clone()
        cand.unflatten(update.weights)
        score    = cand.accuracy(self.X_val, self.y_val)
        baseline = self.global_mod.accuracy(self.X_val, self.y_val)
        accepted = score >= max(self.threshold, baseline - 0.05)
        return ValidationResult(trainer_id=update.trainer_id,
                                validator_id=self.vid,
                                accepted=accepted,
                                score=score,
                                round_num=round_num)

    def sync(self, global_model: SimpleModel):
        self.global_mod = global_model.clone()


# ─────────────────────────────────────────────────────────────────────────────
# FLoBC Framework  — Real Blockchain Edition
# ─────────────────────────────────────────────────────────────────────────────

class FLoBC:
    """
    FLoBC with a REAL cryptographic blockchain.

    Each training round produces a Block containing:
      • MODEL_UPDATE  transactions  (one per trainer, signed with trainer wallet)
      • VALIDATION    transactions  (one per validator per update, signed)
      • TRUST_UPDATE  transactions  (one per trainer when score changes)
      • GLOBAL_MODEL  transaction   (one per round — the aggregated model hash)

    The block is committed via Proof-of-Stake voting:
      • Each validator votes YES/NO on the candidate block
      • Block accepted when YES stake > 2/3 total stake (pBFT threshold)
      • Block hash = SHA-256(index + prev_hash + merkle_root + votes + timestamp)
      • Merkle root = binary Merkle tree over all transaction hashes
    """

    def __init__(
        self,
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_trainers:         int   = 7,
        n_validators:       int   = 3,
        sync_scheme:        SyncScheme = SyncScheme.BSP,
        use_reputation:     bool  = True,
        noise_profile:      Optional[List[float]] = None,
        lr:                 float = 0.05,
        batch_size:         int   = 64,
        bap_majority_ratio: float = 1.0,
        ssp_slack_ratio:    float = 0.2,
        verbose_chain:      bool  = False,
    ):
        self.n_trainers         = n_trainers
        self.n_validators       = n_validators
        self.sync_scheme        = sync_scheme
        self.use_reputation     = use_reputation
        self.lr                 = lr
        self.batch_size         = batch_size
        self.bap_majority_ratio = bap_majority_ratio
        self.ssp_slack_ratio    = ssp_slack_ratio
        self.X_test             = X_test
        self.y_test             = y_test
        self.verbose_chain      = verbose_chain

        # ── Global model ───────────────────────────────────────────────────
        input_dim  = X_train.shape[1]
        output_dim = int(np.max(y_train)) + 1
        hidden_dim = max(32, min(128, input_dim // 6))
        self.global_model = SimpleModel(input_dim, hidden_dim, output_dim)

        # ── Trainer nodes ──────────────────────────────────────────────────
        splits = np.array_split(np.arange(len(X_train)), n_trainers)
        if noise_profile is None:
            noise_profile = [0.0] * n_trainers
        self.trainers = [
            TrainerNode(i, X_train[splits[i]], y_train[splits[i]],
                        self.global_model,
                        noise_profile[i] if i < len(noise_profile) else 0.0)
            for i in range(n_trainers)
        ]

        # ── Validator nodes ────────────────────────────────────────────────
        val_splits = np.array_split(np.arange(len(X_val)), n_validators)
        self.validators = [
            ValidatorNode(j, X_val[val_splits[j]], y_val[val_splits[j]],
                          self.global_model)
            for j in range(n_validators)
        ]

        # ── Reputation service ─────────────────────────────────────────────
        self.reputation = ReputationService(list(range(n_trainers)))

        # ── Real Blockchain + PoS ──────────────────────────────────────────
        self.chain = RealBlockchain()
        # Each validator gets initial equal stake
        val_stakes = {v.wallet.address: 1.0 / n_validators
                      for v in self.validators}
        self.pos   = ProofOfStake(val_stakes)
        self._val_addresses = [v.wallet.address for v in self.validators]

        # ── Framework wallet (signs global-model transactions) ─────────────
        self._fw_wallet = Wallet()

        # ── Logging ────────────────────────────────────────────────────────
        self.accuracy_log: List[float]            = []
        self.round_times:  List[float]            = []
        self.trust_log:    Dict[int, List[float]] = defaultdict(list)

    # ── Sync filters ────────────────────────────────────────────────────────

    def _apply_sync(self, updates):
        if self.sync_scheme == SyncScheme.BSP:
            return updates
        elif self.sync_scheme == SyncScheme.SSP:
            return updates[:max(1, int(len(updates)*(1-self.ssp_slack_ratio)))]
        else:
            return updates[:max(1, int(len(updates)*self.bap_majority_ratio))]

    # ── Federated averaging ─────────────────────────────────────────────────

    def _fed_avg(self, updates):
        ws = [self.reputation.weight(u.trainer_id) for u in updates]
        t  = sum(ws) or 1.0
        ws = [w/t for w in ws]
        return sum(w*u.weights for w, u in zip(ws, updates))

    # ── Consensus validate (ML layer) ───────────────────────────────────────

    def _consensus_validate(self, update, rnd):
        results  = [v.validate(update, rnd) for v in self.validators]
        accepted = sum(r.accepted for r in results) > len(results) * 2/3
        avg_sc   = float(np.mean([r.score for r in results]))
        return accepted, avg_sc, results

    # ── Push global model ───────────────────────────────────────────────────

    def _push_global(self, new_w):
        self.global_model.unflatten(new_w)
        for t in self.trainers: t.pull(self.global_model)
        for v in self.validators: v.sync(self.global_model)

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self, n_rounds=30, verbose=True) -> Dict:
        acc = self.global_model.accuracy(self.X_test, self.y_test)
        self.accuracy_log.append(acc)

        for rnd in range(1, n_rounds + 1):
            t0 = time.time()
            block_txs: List[Transaction] = []

            # ── 1. Trainer updates ──────────────────────────────────────────
            raw_updates = []
            for trainer in self.trainers:
                upd = trainer.train_step(lr=self.lr, batch_size=self.batch_size)
                upd.round_num = rnd
                raw_updates.append(upd)
                # Sign & record MODEL_UPDATE transaction
                tx = make_model_update_tx(
                    wallet=trainer.wallet,
                    trainer_id=trainer.tid,
                    round_num=rnd,
                    weights_hash=upd.weights_hash,
                    noise_level=upd.noise_level,
                )
                block_txs.append(tx)

            # ── 2. Sync scheme ──────────────────────────────────────────────
            candidates = self._apply_sync(raw_updates)

            # ── 3. Validate + reputation + validation transactions ──────────
            valid_updates = []
            baseline = self.global_model.accuracy(self.X_test, self.y_test)

            for upd in candidates:
                ok, _, val_results = self._consensus_validate(upd, rnd)
                if ok:
                    valid_updates.append(upd)

                # VALIDATION transactions (one per validator)
                for vr in val_results:
                    validator = self.validators[vr.validator_id]
                    tx = make_validation_tx(
                        wallet=validator.wallet,
                        validator_id=vr.validator_id,
                        trainer_id=vr.trainer_id,
                        round_num=rnd,
                        accepted=vr.accepted,
                        score=vr.score,
                    )
                    block_txs.append(tx)

                # TRUST_UPDATE transactions
                if self.use_reputation:
                    test_m = self.global_model.clone()
                    test_m.unflatten(upd.weights)
                    improved = test_m.accuracy(self.X_test, self.y_test) >= baseline
                    old_sc, new_sc = self.reputation.update(upd.trainer_id, improved)
                    # Update validator stake proportionally
                    tx = make_trust_update_tx(
                        wallet=self._fw_wallet,
                        trainer_id=upd.trainer_id,
                        old_score=old_sc,
                        new_score=new_sc,
                        round_num=rnd,
                    )
                    block_txs.append(tx)

            # ── 4. Federated averaging ──────────────────────────────────────
            if valid_updates:
                new_w = self._fed_avg(valid_updates)
                self._push_global(new_w)

            # ── 5. GLOBAL_MODEL transaction ────────────────────────────────
            gm_tx = make_global_model_tx(
                wallet=self._fw_wallet,
                round_num=rnd,
                weights_hash=_weights_hash(self.global_model.flatten()),
                accuracy=self.global_model.accuracy(self.X_test, self.y_test),
                accepted_trainers=[u.trainer_id for u in valid_updates],
                trust_scores=dict(self.reputation.scores),
            )
            block_txs.append(gm_tx)

            # ── 6. Commit block to REAL blockchain (PoS voting) ────────────
            proposer = self.pos.select_proposer()
            committed_block = self.chain.propose_block(
                transactions=block_txs,
                validator_address=proposer,
                pos=self.pos,
                all_validator_addresses=self._val_addresses,
            )

            # ── 7. Evaluate & log ───────────────────────────────────────────
            acc     = self.global_model.accuracy(self.X_test, self.y_test)
            elapsed = time.time() - t0
            self.accuracy_log.append(acc)
            self.round_times.append(elapsed)
            for tid, s in self.reputation.scores.items():
                self.trust_log[tid].append(s)

            if verbose:
                blk_info = (f"block#{committed_block.index} "
                            f"hash={committed_block.block_hash[:10]}..."
                            if committed_block else "NO BLOCK")
                accepted_ids = [u.trainer_id for u in valid_updates]
                print(f"  Round {rnd:3d} | Acc: {acc:.4f} | "
                      f"Trainers: {accepted_ids} | "
                      f"TXs: {len(block_txs):2d} | "
                      f"{blk_info} | {elapsed:.2f}s")

            if self.verbose_chain and committed_block:
                print(f"           Merkle: {committed_block.merkle_root[:20]}...")
                print(f"           PoS votes: {len(committed_block.stake_votes)} validators")

        return {
            "accuracy_log":   self.accuracy_log,
            "round_times":    self.round_times,
            "trust_log":      dict(self.trust_log),
            "chain_length":   self.chain.length(),
            "chain_valid":    self.chain.is_chain_valid(),
            "final_accuracy": self.accuracy_log[-1],
        }

    def export_chain(self, path: str):
        """Save the full blockchain to a JSON file."""
        self.chain.export_json(path)
        print(f"  [Blockchain] Chain exported → {path}")

    def print_chain(self, max_blocks=10):
        self.chain.print_chain(max_blocks=max_blocks)
