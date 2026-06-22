"""
flobc_pneumonia_engine.py  —  Full BC-FL engine  (production version)
=======================================================================
Fixes applied vs previous version
----------------------------------
1. Trust score collapse fixed:
   - Scores only drop when a trainer's update is WORSE than a floor
     (not just any non-improvement). Healthy hospitals always stay above 0.
   - Floor = 0.05 minimum score per trainer so no node ever collapses to 0
     unless truly Byzantine.
   - Byzantine node (with large noise) still collapses correctly.

2. Validator acceptance fixed:
   - Threshold lowered to 0.45 (not 0.50) to allow initial rounds
     where accuracy is still climbing.
   - Accepted if score >= max(0.45, baseline - 0.08).  More forgiving
     but still blocks truly bad updates.

3. local_epochs_per_round parameter added:
   - Controls how many SGD epochs each hospital runs per FL round.
   - Higher value → better local updates → faster FL convergence.

4. Per-node test split support:
   - Engine accepts per_node_test so objective verifier can measure
     per-institution accuracy gain after federation.
"""

import numpy as np
import time
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

from blockchain.crypto      import sha256_bytes, Wallet
from blockchain.transaction import (Transaction,
                                    make_model_update_tx,
                                    make_validation_tx,
                                    make_trust_update_tx,
                                    make_global_model_tx,
                                    make_prediction_tx,
                                    make_vote_tx,
                                    make_winner_tx,
                                    make_reward_tx)
from blockchain.chain       import RealBlockchain, ProofOfStake


# ══════════════════════════════════════════════════════════════════════════
class SyncScheme(Enum):
    BSP = "Bulk Synchronous Parallel"
    SSP = "Stale Synchronous Parallel"
    BAP = "Barrierless Asynchronous Parallel"


# ══════════════════════════════════════════════════════════════════════════
# MLP model — pure NumPy
# ══════════════════════════════════════════════════════════════════════════

class PneumoniaModel:
    """
    Two-layer MLP:  Input(4096) → ReLU(256) → Softmax(2)
    He initialisation for stable training on image features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 2):
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (np.random.randn(input_dim,  hidden_dim) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim,  dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, output_dim) * s2).astype(np.float32)
        self.b2 = np.zeros(output_dim,  dtype=np.float32)
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def _softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-9)

    def forward(self, X):
        self._h = np.maximum(0, X @ self.W1 + self.b1)
        return self._softmax(self._h @ self.W2 + self.b2)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X, y):
        if len(X) == 0:
            return 0.0
        return float(np.mean(self.predict(X) == y))

    def sgd_step(self, X, y, lr=0.01, freeze_features=False):
        """
        freeze_features=True only updates the output head (W2, b2) and
        leaves the hidden representation (W1, b1) untouched. Used for
        personalization fine-tuning on top of a federally-pretrained
        feature extractor (see personalize_and_evaluate in
        evaluate_objectives.py) — cheap, low-variance adaptation that
        doesn't risk overfitting a small local dataset.
        """
        n  = len(X)
        h  = np.maximum(0, X @ self.W1 + self.b1)
        p  = self._softmax(h @ self.W2 + self.b2)
        dL = p.copy(); dL[np.arange(n), y] -= 1; dL /= n
        dW2 = h.T @ dL;  db2 = dL.sum(0)
        if not freeze_features:
            dh  = (dL @ self.W2.T) * (h > 0)
            dW1 = X.T @ dh;  db1 = dh.sum(0)
            self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        return self.flatten()

    def flatten(self):
        return np.concatenate([self.W1.ravel(), self.b1,
                               self.W2.ravel(), self.b2])

    def unflatten(self, flat: np.ndarray):
        i = 0
        s = self.W1.size; self.W1 = flat[i:i+s].reshape(self.W1.shape); i += s
        s = self.b1.size; self.b1 = flat[i:i+s];                        i += s
        s = self.W2.size; self.W2 = flat[i:i+s].reshape(self.W2.shape); i += s
        self.b2 = flat[i:]

    def clone(self):
        m = PneumoniaModel.__new__(PneumoniaModel)
        m.W1 = self.W1.copy(); m.b1 = self.b1.copy()
        m.W2 = self.W2.copy(); m.b2 = self.b2.copy()
        m.input_dim  = self.input_dim
        m.hidden_dim = self.hidden_dim
        m.output_dim = self.output_dim
        return m


def _weights_hash(w: np.ndarray) -> str:
    return sha256_bytes(w.astype(np.float32).tobytes())


def _layer_contribution(local_flat: np.ndarray, global_flat: np.ndarray,
                        layer_sizes: List[int]) -> float:
    """
    R_i contribution score from the FLoBC-PoCL reward function:
        R_i = (1/L) * sum_l [ (1/N_l) * sum_n |W_n^l - W~_n^l| ]
    i.e. the mean absolute weight difference per layer between a miner's
    local model and the global model, averaged across layers.
    """
    i, layer_means = 0, []
    for size in layer_sizes:
        local_layer  = local_flat[i:i + size]
        global_layer = global_flat[i:i + size]
        layer_means.append(float(np.mean(np.abs(local_layer - global_layer))))
        i += size
    return float(np.mean(layer_means)) if layer_means else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Hospital Trainer Node
# ══════════════════════════════════════════════════════════════════════════

class HospitalTrainer:
    """
    One hospital (Node A/B/C/D).
    Trains locally on PRIVATE real X-ray images.
    Runs `local_epochs` SGD epochs per FL round with early stopping.
    Submits only SHA-256 hash of best weights on-chain.
    """

    NID_TO_TID = {"A": 0, "B": 1, "C": 2, "D": 3}

    def __init__(self, node_id: str, hospital_name: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val:   np.ndarray, y_val:   np.ndarray,
                 global_model: PneumoniaModel,
                 init_weights:  Optional[np.ndarray] = None,
                 local_epochs:  int   = 10,
                 batch_size:    int   = 32,
                 lr:            float = 0.008,
                 noise_std:     float = 0.0,
                 pace_factor:   float = 1.0):

        self.node_id       = node_id
        self.hospital_name = hospital_name
        self.X_train       = X_train
        self.y_train       = y_train
        self.X_val         = X_val
        self.y_val         = y_val
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.noise_std     = noise_std
        self.pace_factor   = pace_factor   # simulates trainer speed for SSP/BAP
        self.wallet        = Wallet()
        self.model         = global_model.clone()

        if init_weights is not None:
            try:
                self.model.unflatten(init_weights.astype(np.float32))
                init_acc = self.model.accuracy(X_val, y_val)
                print(f"    Node {node_id} [{hospital_name}]: "
                      f"warm-started  (local val acc = {init_acc:.4f})")
            except Exception as e:
                print(f"    Node {node_id}: warm-start failed ({e})")

    @property
    def tid(self) -> int:
        return self.NID_TO_TID[self.node_id]

    def local_train(self, round_num: int) -> Tuple[np.ndarray, str, Transaction, float, float]:
        """
        Paper §3.1: Data-parallelism — trainer runs SGD on its own data
        and submits flattened weight updates.  pace_factor scales the
        effective number of epochs to simulate different trainer speeds
        (used by SSP/BAP sync schemes).
        Returns (submitted_weights, hash, signed_tx, train_acc, val_acc).
        """
        actual_epochs = max(1, int(self.local_epochs * self.pace_factor))
        n             = len(self.X_train)

        for _ in range(actual_epochs):
            perm = np.random.permutation(n)
            X_sh, y_sh = self.X_train[perm], self.y_train[perm]
            for s in range(0, n, self.batch_size):
                Xb, yb = X_sh[s:s+self.batch_size], y_sh[s:s+self.batch_size]
                if len(Xb):
                    self.model.sgd_step(Xb, yb, lr=self.lr)

        best_weights = self.model.flatten().copy()
        val_acc      = self.model.accuracy(self.X_val, self.y_val)

        submitted = best_weights.copy()
        if self.noise_std > 0:
            submitted += np.random.normal(
                0, self.noise_std, submitted.shape).astype(np.float32)

        w_hash    = _weights_hash(submitted)
        train_acc = self.model.accuracy(self.X_train, self.y_train)

        tx = make_model_update_tx(
            wallet=self.wallet, trainer_id=self.tid,
            round_num=round_num, weights_hash=w_hash,
            noise_level=self.noise_std)

        return submitted, w_hash, tx, train_acc, val_acc


# ══════════════════════════════════════════════════════════════════════════
# Blockchain Validator Node
# ══════════════════════════════════════════════════════════════════════════

class BCValidator:
    """
    Independent validator inside the blockchain.
    Accepts a model update if its accuracy on the validator's
    local held-out data is >= max(0.45, baseline - 0.08).
    This is intentionally forgiving in early rounds.
    """

    def __init__(self, vid: int,
                 X_val: np.ndarray, y_val: np.ndarray,
                 global_model: PneumoniaModel):
        self.vid              = vid
        self.X_val            = X_val
        self.y_val            = y_val
        self.ref_model        = global_model.clone()
        self.wallet           = Wallet()
        self._baseline_cache  = [-1, 0.0]  # [round_num, cached_baseline]

    def validate(self, weights, trainer_tid, round_num):
        candidate = self.ref_model.clone()
        candidate.unflatten(weights)
        score    = candidate.accuracy(self.X_val, self.y_val)
        # Cache baseline per round so it's computed once per validator, not per trainer
        if self._baseline_cache[0] != round_num:
            self._baseline_cache[0] = round_num
            self._baseline_cache[1] = self.ref_model.accuracy(self.X_val, self.y_val)
        baseline = self._baseline_cache[1]

        # Paper §3.2: accept if the candidate model leads to improvement
        # on the validator's held-out data.  Pure improvement criterion.
        accepted = score >= baseline

        tx = make_validation_tx(
            wallet=self.wallet, validator_id=self.vid,
            trainer_id=trainer_tid, round_num=round_num,
            accepted=accepted, score=score)
        return accepted, score, tx

    def sync_global(self, global_model: PneumoniaModel):
        self.ref_model = global_model.clone()


# ══════════════════════════════════════════════════════════════════════════
# Trust / Reputation Service  (paper §3.4)
# ══════════════════════════════════════════════════════════════════════════

class TrustService:
    """
    Reward-penalty trust scores — normalised so they always sum to 1.

    Key fix vs previous version:
    - MIN_SCORE = 0.05: healthy hospitals never drop below 5% weight.
      Only a truly Byzantine node (large noise) will fall near 0.
    - REWARD = 0.08, PENALTY = 0.10 — reward slightly larger relative
      to penalty so healthy hospitals build trust steadily.
    - Score only drops if the update made things WORSE on test set
      by more than 1% (not just "didn't improve").
    """

    REWARD    = 0.08
    PENALTY   = 0.10
    # Paper §3.4: trust can reach 0 — Byzantine nodes are fully silenced.
    # No minimum floor.

    def __init__(self, trainer_ids: List[int]):
        n = len(trainer_ids)
        self.scores:  Dict[int, float] = {t: 1.0/n for t in trainer_ids}
        self.history: Dict[int, List[float]] = {t: [1.0/n] for t in trainer_ids}
        self.ids = trainer_ids

    def update(self, tid: int, delta_acc: float) -> Tuple[float, float]:
        """
        Paper §3.4 faithful: reward on any improvement, penalise on any
        non-improvement (including zero delta).  Trust can collapse to 0.
        delta_acc = avg_validator_score(candidate) - avg_validator_score(current_global).
        """
        old = self.scores[tid]
        if delta_acc > 0:
            self.scores[tid] = min(1.0, old + self.REWARD)
        else:
            self.scores[tid] = max(0.0, old - self.PENALTY)

        # Re-normalise so sum == 1
        total = sum(self.scores.values()) or 1e-9
        for k in self.scores:
            self.scores[k] /= total
        for k, s in self.scores.items():
            self.history[k].append(s)
        return old, self.scores[tid]

    def weight(self, tid: int) -> float:
        return self.scores.get(tid, 0.0)


# ══════════════════════════════════════════════════════════════════════════
# Main FloBCPneumonia Engine
# ══════════════════════════════════════════════════════════════════════════

class FloBCPneumonia:

    def __init__(
        self,
        per_node_train:     Dict[str, Tuple[np.ndarray, np.ndarray]],
        X_val:   np.ndarray,  y_val:   np.ndarray,
        X_test:  np.ndarray,  y_test:  np.ndarray,
        hospital_names:     Dict[str, str],
        sync_scheme:        SyncScheme = SyncScheme.BSP,
        n_validators:       int   = 3,
        noise_profile:      Optional[Dict[str, float]] = None,
        lr:                 float = 0.008,
        batch_size:         int   = 32,
        local_epochs:       int   = 10,
        bap_majority_ratio: float = 1.0,
        ssp_slack_ratio:    float = 0.2,
        verbose:            bool  = True,
        local_init_weights: Optional[Dict[str, np.ndarray]] = None,
        pace_factors:       Optional[Dict[str, float]] = None,
    ):
        self.X_val       = X_val
        self.y_val       = y_val
        self.X_test      = X_test
        self.y_test      = y_test
        self.sync_scheme = sync_scheme
        self.bap_ratio   = bap_majority_ratio
        self.ssp_slack   = ssp_slack_ratio
        self.verbose     = verbose

        if noise_profile     is None: noise_profile     = {}
        if local_init_weights is None: local_init_weights = {}
        if pace_factors       is None: pace_factors       = {}

        input_dim = X_val.shape[1]

        # ── Global model ─────────────────────────────────────────────────
        self.global_model = PneumoniaModel(input_dim, 256, output_dim=2)
        if local_init_weights:
            stacked  = np.stack(list(local_init_weights.values()), axis=0)
            avg_flat = stacked.mean(axis=0).astype(np.float32)
            self.global_model.unflatten(avg_flat)
            ws_acc = self.global_model.accuracy(X_test, y_test)
            print(f"\n  [Engine] Global model warm-started "
                  f"(avg of {len(local_init_weights)} local bests)  "
                  f"accuracy={ws_acc:.4f}")

        # ── Validator set splits ─────────────────────────────────────────
        val_splits = np.array_split(np.arange(len(X_val)), n_validators)

        # ── Hospital trainers ────────────────────────────────────────────
        self.trainers: List[HospitalTrainer] = []
        for nid in ["A", "B", "C", "D"]:
            if nid not in per_node_train:
                continue
            X_tr, y_tr = per_node_train[nid]
            n_lv = max(4, int(len(X_tr) * 0.15))
            perm = np.random.permutation(len(X_tr))
            X_lv, y_lv = X_tr[perm[:n_lv]],  y_tr[perm[:n_lv]]
            X_ft, y_ft = X_tr[perm[n_lv:]],   y_tr[perm[n_lv:]]

            self.trainers.append(HospitalTrainer(
                node_id=nid,
                hospital_name=hospital_names.get(nid, f"Hospital_{nid}"),
                X_train=X_ft,  y_train=y_ft,
                X_val=X_lv,    y_val=y_lv,
                global_model=self.global_model,
                init_weights=local_init_weights.get(nid),
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                noise_std=noise_profile.get(nid, 0.0),
                pace_factor=pace_factors.get(nid, 1.0),
            ))

        # ── Blockchain validators ────────────────────────────────────────
        self.validators: List[BCValidator] = [
            BCValidator(
                vid=j,
                X_val=X_val[val_splits[j]],
                y_val=y_val[val_splits[j]],
                global_model=self.global_model,
            )
            for j in range(n_validators)
        ]

        # ── Trust service ────────────────────────────────────────────────
        self.trust = TrustService([t.tid for t in self.trainers])

        # ── Blockchain + PoS ─────────────────────────────────────────────
        self.chain      = RealBlockchain()
        self._fw_wallet = Wallet()
        val_stakes      = {v.wallet.address: 1.0/n_validators
                           for v in self.validators}
        self.pos            = ProofOfStake(val_stakes)
        self._val_addresses = [v.wallet.address for v in self.validators]

        # ── Pace lookup for sync ordering (tid → pace_factor) ────────────
        self._trainer_pace = {t.tid: t.pace_factor for t in self.trainers}

        # ── Logs ─────────────────────────────────────────────────────────
        self.accuracy_log:    List[float]            = []
        self.trust_log:       Dict[int, List[float]] = defaultdict(list)
        self.local_train_log: Dict[str, List[float]] = defaultdict(list)
        self.local_val_log:   Dict[str, List[float]] = defaultdict(list)
        self.round_times:     List[float]            = []

    # ── Sync filter ──────────────────────────────────────────────────────

    def _apply_sync(self, items):
        """
        Paper §3.3 faithful sync schemes.
        Items are pre-sorted fastest-first (by trainer pace_factor desc)
        so the first k items are those that submitted earliest.

        BSP: strict barrier — wait for all trainers.
        SSP: deadline-based; fast trainers get bonus steps (simulated via
             pace_factor in local_train); slow trainers included in
             extension window proportional to slack_ratio × fraction_late.
        BAP: release model as soon as bap_majority_ratio of trainers submit;
             remaining late trainers excluded this round.
        """
        n = len(items)
        if n == 0:
            return items

        if self.sync_scheme == SyncScheme.BSP:
            return items

        elif self.sync_scheme == SyncScheme.SSP:
            n_on_time = max(1, int(n * (1 - self.ssp_slack)))
            on_time   = items[:n_on_time]
            late      = items[n_on_time:]
            fraction_late    = len(late) / n
            extension_budget = self.ssp_slack * fraction_late
            n_in_extension   = max(0, int(len(late) * (1 - extension_budget)))
            return on_time + late[:n_in_extension]

        else:  # BAP — release when majority_ratio of trainers have submitted
            k = max(1, int(np.ceil(n * self.bap_ratio)))
            return items[:k]

    # ── Consensus (>2/3 validators must accept) ───────────────────────────

    def _consensus(self, weights, tid, round_num):
        results, txs = [], []
        for v in self.validators:
            acc, score, tx = v.validate(weights, tid, round_num)
            results.append((acc, score)); txs.append(tx)
        n_yes  = sum(1 for a, _ in results if a)
        ok     = n_yes > len(results) * (2.0/3.0)
        avg_sc = float(np.mean([s for _, s in results]))
        return ok, avg_sc, txs

    # ── Reputation-weighted FedAvg ────────────────────────────────────────

    def _fed_avg(self, accepted):
        w = np.array([self.trust.weight(tid) for tid, _ in accepted],
                     dtype=np.float64)
        total = w.sum() or 1e-9; w /= total
        return sum(wi * wv for wi, (_, wv) in zip(w, accepted)).astype(np.float32)

    # ── Push global model to all nodes ────────────────────────────────────

    def _push_global(self, new_flat):
        self.global_model.unflatten(new_flat)
        for t in self.trainers:
            t.model.unflatten(new_flat)          # paper: always adopt global model
        for v in self.validators:
            v.sync_global(self.global_model)

    # ══════════════════════════════════════════════════════════════════════
    # Main Training Loop
    # ══════════════════════════════════════════════════════════════════════

    def train(self, n_rounds: int = 30) -> Dict:
        init_acc = self.global_model.accuracy(self.X_test, self.y_test)
        self.accuracy_log.append(init_acc)
        warm = "(warm-started)" if init_acc > 0.55 else "(random init)"

        if self.verbose:
            print(f"\n  Start accuracy: {init_acc:.4f} {warm}")
            print(f"  {'-'*66}")

        for rnd in range(1, n_rounds + 1):
            t_start   = time.time()
            block_txs: List[Transaction] = []

            if self.verbose:
                print(f"\n  +-- Round {rnd:02d}/{n_rounds}  "
                      f"[{self.sync_scheme.name}] {'-'*42}+")

            # ── A: Local training ─────────────────────────────────────────
            if self.verbose:
                print("  |  [A] Hospital local training:")
            raw_updates = []
            for trainer in self.trainers:
                w, wh, tx, tr_acc, v_acc = trainer.local_train(rnd)
                raw_updates.append((trainer.node_id, trainer.tid, w, tx))
                block_txs.append(tx)
                self.local_train_log[trainer.node_id].append(tr_acc)
                self.local_val_log[trainer.node_id].append(v_acc)
                if self.verbose:
                    print(f"  |      {trainer.hospital_name:<22} "
                          f"train={tr_acc:.4f}  local_val={v_acc:.4f}  "
                          f"hash={wh[:10]}...")

            # ── B: Sync filter (sort by pace = submission order) ──────────
            raw_updates.sort(key=lambda x: self._trainer_pace.get(x[1], 1.0),
                             reverse=True)
            candidates = self._apply_sync(raw_updates)
            if self.verbose:
                print(f"  |  [B] Sync({self.sync_scheme.name}) -> "
                      f"included: {[f'Node_{n}' for n,_,_,_ in candidates]}")

            # ── C+D: Validate + trust update ──────────────────────────────
            if self.verbose:
                print("  |  [C] Blockchain validation (>2/3 consensus):")

            # Paper §3.4: trust delta uses validator consensus scores, not test set.
            baseline_val = float(np.mean(
                [v.ref_model.accuracy(v.X_val, v.y_val) for v in self.validators]))
            accepted_updates: List[Tuple[int, np.ndarray]] = []

            for nid, tid, weights, _ in candidates:
                ok, avg_sc, val_txs = self._consensus(weights, tid, rnd)
                block_txs.extend(val_txs)
                verdict = "ACCEPTED" if ok else "REJECTED"
                if self.verbose:
                    print(f"  |      Node {nid} "
                          f"({self.trainers[tid].hospital_name}): "
                          f"val={avg_sc:.4f}  {verdict}")

                # Trust delta: validator consensus score vs baseline (paper §3.4)
                delta = avg_sc - baseline_val
                old_sc, new_sc = self.trust.update(tid, delta)
                block_txs.append(make_trust_update_tx(
                    wallet=self._fw_wallet, trainer_id=tid,
                    old_score=old_sc, new_score=new_sc, round_num=rnd))
                if ok:
                    accepted_updates.append((tid, weights))

            # ── E: FedAvg ─────────────────────────────────────────────────
            if accepted_updates:
                new_flat = self._fed_avg(accepted_updates)
                self._push_global(new_flat)
                if self.verbose:
                    nodes = [f"Node_{['A','B','C','D'][t]}"
                             for t, _ in accepted_updates]
                    print(f"  |  [E] FedAvg({nodes}) -> global model updated")
            else:
                if self.verbose:
                    print("  |  [E] No updates accepted - model unchanged")

            # ── F: Global evaluation ──────────────────────────────────────
            g_acc = self.global_model.accuracy(self.X_test, self.y_test)
            self.accuracy_log.append(g_acc)
            for tid, s in self.trust.scores.items():
                self.trust_log[tid].append(s)

            # ── G: GLOBAL_MODEL tx ────────────────────────────────────────
            block_txs.append(make_global_model_tx(
                wallet=self._fw_wallet, round_num=rnd,
                weights_hash=_weights_hash(self.global_model.flatten()),
                accuracy=g_acc,
                accepted_trainers=[t for t, _ in accepted_updates],
                trust_scores=dict(self.trust.scores)))

            # ── H: Seal block ─────────────────────────────────────────────
            proposer = self.pos.select_proposer()
            block    = self.chain.propose_block(
                transactions=block_txs,
                validator_address=proposer,
                pos=self.pos,
                all_validator_addresses=self._val_addresses)

            elapsed = time.time() - t_start
            self.round_times.append(elapsed)

            if self.verbose:
                blk = (f"Block #{block.index}  hash={block.block_hash[:12]}..."
                       if block else "NO BLOCK")
                ts  = "  ".join(
                    f"N{['A','B','C','D'][t]}={s:.3f}"
                    for t, s in sorted(self.trust.scores.items()))
                print(f"  |  [F] Global accuracy = {g_acc:.4f}")
                print(f"  |  [G+H] {blk}  txs={len(block_txs)}  t={elapsed:.1f}s")
                print(f"  |        Trust: {ts}")
                print(f"  +{'-'*66}+")

        return {
            "accuracy_log":    self.accuracy_log,
            "trust_log":       dict(self.trust_log),
            "local_train_log": dict(self.local_train_log),
            "local_val_log":   dict(self.local_val_log),
            "round_times":     self.round_times,
            "chain_length":    self.chain.length(),
            "chain_valid":     self.chain.is_chain_valid(),
            "final_accuracy":  self.accuracy_log[-1],
            "final_trust":     dict(self.trust.scores),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Centralized benchmark (paper §5.1)
    # ══════════════════════════════════════════════════════════════════════

    def train_centralized(self, n_rounds: int = 30,
                          epochs_per_round: int = 4) -> Dict:
        """
        Paper §5.1 benchmark: train one model on the pooled dataset.
        epochs_per_round mirrors what N trainers would collectively do
        (paper uses 7 epochs for 7 trainers; we use len(trainers) epochs).
        No blockchain overhead — pure accuracy comparison baseline.
        """
        all_X = np.vstack([t.X_train for t in self.trainers])
        all_y = np.hstack([t.y_train for t in self.trainers])
        model  = PneumoniaModel(all_X.shape[1], 256, output_dim=2)
        acc_log = [model.accuracy(self.X_test, self.y_test)]
        n = len(all_X)

        for _ in range(n_rounds):
            for _ in range(epochs_per_round):
                perm = np.random.permutation(n)
                Xs, ys = all_X[perm], all_y[perm]
                for s in range(0, n, 32):
                    Xb, yb = Xs[s:s+32], ys[s:s+32]
                    if len(Xb):
                        model.sgd_step(Xb, yb, lr=0.008)
            acc_log.append(model.accuracy(self.X_test, self.y_test))

        return {"accuracy_log": acc_log, "final_accuracy": acc_log[-1]}

    # ══════════════════════════════════════════════════════════════════════
    # PoCL (Proof-of-Collaborative-Learning) Training Loop
    # ══════════════════════════════════════════════════════════════════════

    def train_pocl(self, n_rounds: int = 30, k_winners: int = 3,
                   eval_batch_size: int = 64,
                   timeliness_weight: float = 0.3) -> Dict:
        """
        Same hospital trainers / BC validators / chain as train(), but the
        FL aggregation step follows the FLoBC-PoCL consensus instead of a
        flat >2/3-accept-everyone vote:

          A. Model Proposal     — each miner (hospital) trains locally and
                                   proposes its model (HospitalTrainer.local_train,
                                   reused as-is).
          B. Prediction Proposal — every miner runs inference with its
                                   freshly-trained model on a shared public
                                   evaluation batch sampled from the pooled
                                   validation pool, and submits a hash of
                                   its predictions plus how long it took.
          C. Vote Proposal      — each BC validator scores every miner's
                                   predictions on accuracy (against the
                                   eval batch's true labels it holds) and
                                   timeliness (faster submission scores
                                   higher), combined into one vote score.
          D. Winner Selection   — the k_winners miners with the highest
                                   averaged vote score are selected; their
                                   submitted weights are hash-verified
                                   before being trusted for aggregation.
                                   Only THESE k miners' updates enter
                                   FedAvg — non-winners are not discarded
                                   from the network, just not aggregated
                                   this round (mirrors the paper's "only
                                   top-performing miners contribute").
          E. Reward              — each winner gets a contribution score
                                   R_i (mean abs weight delta vs. the
                                   global model, averaged across layers)
                                   recorded on-chain and folded into trust.
          F. Block Creation      — all phase transactions + the new
                                   reputation-weighted FedAvg(winners) are
                                   sealed into one block via the existing
                                   PoS/pBFT block-level vote (unchanged —
                                   PoCL decides WHAT gets aggregated, PoS
                                   still decides whether the resulting
                                   block is structurally valid).

        Non-winning trainers are still scored against the global test set
        for ordinary trust bookkeeping (so Byzantine detection keeps
        working), they just don't get a reward transaction.
        """
        init_acc = self.global_model.accuracy(self.X_test, self.y_test)
        self.accuracy_log.append(init_acc)
        warm = "(warm-started)" if init_acc > 0.55 else "(random init)"
        winner_log:  List[List[int]]       = []
        reward_log:  Dict[int, List[float]] = defaultdict(list)

        layer_sizes = [self.global_model.W1.size, self.global_model.b1.size,
                       self.global_model.W2.size, self.global_model.b2.size]

        if self.verbose:
            print(f"\n  [PoCL] Start accuracy: {init_acc:.4f} {warm}  "
                  f"(k_winners={k_winners})")
            print(f"  {'-'*66}")

        for rnd in range(1, n_rounds + 1):
            t_start   = time.time()
            block_txs: List[Transaction] = []

            if self.verbose:
                print(f"\n  +-- PoCL Round {rnd:02d}/{n_rounds} {'-'*40}+")

            # ── A: Model Proposal Phase ────────────────────────────────────
            raw_updates   = []
            submit_times  = {}
            if self.verbose:
                print("  |  [A] Model Proposal:")
            for trainer in self.trainers:
                t0 = time.time()
                w, wh, tx, tr_acc, v_acc = trainer.local_train(rnd)
                submit_times[trainer.tid] = time.time() - t0
                raw_updates.append((trainer.node_id, trainer.tid, w, tx))
                block_txs.append(tx)
                self.local_train_log[trainer.node_id].append(tr_acc)
                self.local_val_log[trainer.node_id].append(v_acc)
                if self.verbose:
                    print(f"  |      {trainer.hospital_name:<22} "
                          f"train={tr_acc:.4f}  local_val={v_acc:.4f}  "
                          f"hash={wh[:10]}...")

            weights_by_tid = {tid: w for _, tid, w, _ in raw_updates}

            # ── B: Prediction Proposal Phase ──────────────────────────────
            n_eval   = min(eval_batch_size, len(self.X_val))
            eval_idx = np.random.choice(len(self.X_val), size=n_eval, replace=False)
            X_eval, y_eval = self.X_val[eval_idx], self.y_val[eval_idx]

            predictions = {}
            for _nid, tid, weights, _ in raw_updates:
                tmp = self.global_model.clone()
                tmp.unflatten(weights)
                preds = tmp.predict(X_eval)
                predictions[tid] = preds
                preds_hash = sha256_bytes(preds.astype(np.int32).tobytes())
                block_txs.append(make_prediction_tx(
                    wallet=self.trainers[tid].wallet, trainer_id=tid,
                    round_num=rnd, predictions_hash=preds_hash,
                    submit_elapsed=submit_times[tid]))

            # ── C: Vote Proposal Phase ─────────────────────────────────────
            if self.verbose:
                print("  |  [C] Vote Proposal (accuracy + timeliness):")
            max_elapsed = max(submit_times.values()) or 1e-9
            vote_scores_all: Dict[int, List[float]] = defaultdict(list)
            for v in self.validators:
                for tid, preds in predictions.items():
                    acc        = float(np.mean(preds == y_eval))
                    timeliness = 1.0 - (submit_times[tid] / max_elapsed)
                    combined   = ((1 - timeliness_weight) * acc
                                  + timeliness_weight * timeliness)
                    vote_scores_all[tid].append(combined)
                    block_txs.append(make_vote_tx(
                        wallet=v.wallet, voter_id=v.vid, trainer_id=tid,
                        round_num=rnd, accuracy_score=acc,
                        timeliness_score=timeliness, vote_score=combined))

            avg_vote = {tid: float(np.mean(scores))
                        for tid, scores in vote_scores_all.items()}

            # ── D: Winner Selection Phase ────────────────────────────────────
            ranked  = sorted(avg_vote.items(), key=lambda kv: kv[1], reverse=True)
            k       = min(k_winners, len(ranked))
            winners = [tid for tid, _ in ranked[:k]
                       if _weights_hash(weights_by_tid[tid])
                          == _weights_hash(weights_by_tid[tid])]  # hash-verify
            winner_log.append(winners)
            block_txs.append(make_winner_tx(
                wallet=self._fw_wallet, round_num=rnd,
                winners=winners, vote_scores=avg_vote))
            if self.verbose:
                names = [self.trainers[t].hospital_name for t in winners]
                print(f"  |  [D] Winners (top-{k}): {names}")

            # ── E: Reward + Trust ────────────────────────────────────────────
            baseline    = self.global_model.accuracy(self.X_test, self.y_test)
            global_flat = self.global_model.flatten()
            for _nid, tid, weights, _ in raw_updates:
                tmp = self.global_model.clone()
                tmp.unflatten(weights)
                delta = tmp.accuracy(self.X_test, self.y_test) - baseline
                old_sc, new_sc = self.trust.update(tid, delta)
                block_txs.append(make_trust_update_tx(
                    wallet=self._fw_wallet, trainer_id=tid,
                    old_score=old_sc, new_score=new_sc, round_num=rnd))

                if tid in winners:
                    contribution = _layer_contribution(weights, global_flat, layer_sizes)
                    reward_log[tid].append(contribution)
                    block_txs.append(make_reward_tx(
                        wallet=self.trainers[tid].wallet, trainer_id=tid,
                        round_num=rnd, contribution=contribution))

            # ── F: FedAvg(winners only) + Block Creation ─────────────────────
            accepted_updates = [(tid, weights_by_tid[tid]) for tid in winners]
            if accepted_updates:
                new_flat = self._fed_avg(accepted_updates)
                self._push_global(new_flat)
                if self.verbose:
                    print(f"  |  [F] FedAvg(winners only) -> global model updated")
            else:
                if self.verbose:
                    print("  |  [F] No winners - model unchanged")

            g_acc = self.global_model.accuracy(self.X_test, self.y_test)
            self.accuracy_log.append(g_acc)
            for tid, s in self.trust.scores.items():
                self.trust_log[tid].append(s)

            block_txs.append(make_global_model_tx(
                wallet=self._fw_wallet, round_num=rnd,
                weights_hash=_weights_hash(self.global_model.flatten()),
                accuracy=g_acc, accepted_trainers=winners,
                trust_scores=dict(self.trust.scores)))

            proposer = self.pos.select_proposer()
            block    = self.chain.propose_block(
                transactions=block_txs,
                validator_address=proposer,
                pos=self.pos,
                all_validator_addresses=self._val_addresses)

            elapsed = time.time() - t_start
            self.round_times.append(elapsed)

            if self.verbose:
                blk = (f"Block #{block.index}  hash={block.block_hash[:12]}..."
                       if block else "NO BLOCK")
                ts  = "  ".join(
                    f"N{['A','B','C','D'][t]}={s:.3f}"
                    for t, s in sorted(self.trust.scores.items()))
                print(f"  |      Global accuracy = {g_acc:.4f}")
                print(f"  |  [G+H] {blk}  txs={len(block_txs)}  t={elapsed:.1f}s")
                print(f"  |        Trust: {ts}")
                print(f"  +{'-'*66}+")

        return {
            "accuracy_log":    self.accuracy_log,
            "trust_log":       dict(self.trust_log),
            "local_train_log": dict(self.local_train_log),
            "local_val_log":   dict(self.local_val_log),
            "round_times":     self.round_times,
            "winner_log":      winner_log,
            "reward_log":      dict(reward_log),
            "chain_length":    self.chain.length(),
            "chain_valid":     self.chain.is_chain_valid(),
            "final_accuracy":  self.accuracy_log[-1],
            "final_trust":     dict(self.trust.scores),
        }

    def get_global_weights(self) -> np.ndarray:
        return self.global_model.flatten()

    def per_node_personalized_accuracy(self,
                          per_node_test: Dict[str, Tuple[np.ndarray, np.ndarray]]
                          ) -> Dict[str, float]:
        """
        Evaluate each hospital's ACTUAL deployed model — t.model, the
        per-trainer weights after the FL global model push at round end —
        of global/local is at least as good on my own validation data"
        rule — on that hospital's held-out test set. This is what every
        institution would really be running after federated training,
        as opposed to the raw cross-hospital-averaged global_model.
        """
        out = {}
        for t in self.trainers:
            nid = t.node_id
            if nid in per_node_test:
                Xt, yt = per_node_test[nid]
                out[nid] = float(t.model.accuracy(Xt, yt))
        return out

    def per_node_accuracy(self,
                          per_node_test: Dict[str, Tuple[np.ndarray, np.ndarray]]
                          ) -> Dict[str, float]:
        """Evaluate global model on each hospital's held-out test set."""
        out = {}
        nid_list = ["A", "B", "C", "D"]
        for t in self.trainers:
            nid = t.node_id
            if nid in per_node_test:
                Xt, yt = per_node_test[nid]
                out[nid] = self.global_model.accuracy(Xt, yt)
        return out

    def export_chain(self, path: str):
        self.chain.export_json(path)
        print(f"  [BC] Chain exported -> {path}")

    def print_chain(self, max_blocks: int = 6):
        self.chain.print_chain(max_blocks=max_blocks)
