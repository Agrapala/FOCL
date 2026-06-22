"""
Network Actor Roles — FLoBC-PoCL Telemedicine System (Paper Section 3.2)
===========================================================================
Implements the 5 actor roles described in "Decentralization":

  Administrator : Oversees configuration of training rounds, deadlines,
                   and the number of winning miners. Implemented via a
                   (simulated) smart contract for transparent governance.

  Requester     : Submits telemedicine-related deep-learning tasks to the
                   global queue. Each task defines the model architecture
                   and the publicly shareable validation dataset used by
                   Validators.

  Miner          : (a.k.a. Trainer Node) Retrieves the global model, trains
  (Trainer)        it locally on private hospital data, submits only model
                   hashes/CIDs (Phase A), runs inference on shared records
                   (Phase B), and votes on peers' predictions (Phase C).

  Validator      : Independently assesses the quality of miner submissions
                   using a secure validation dataset and performs
                   decentralized voting on which updates are accepted.

  Aggregator     : Integrates only the top-performing (winning) models into
                   the global model using FedAvg, and computes each miner's
                   contribution weight (R_i) for the reward mechanism.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from fl.cnn_model import HealthcareCNN, DifferentialPrivacy
from blockchain.crypto import ExonumWallet
from ipfs.ipfs_node import IPFS


# ─────────────────────────────────────────────────────────────────────────────
# Requester — ModelTask definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelTask:
    """
    Requester-submitted task definition.
    "Each task defines the model architecture and any publicly shareable
    datasets required for collaborative training."
    """
    task_id:             str
    architecture:        str           # e.g. "HealthcareCNN-512-256-128-64"
    hidden_dims:         List[int]
    public_dataset_desc: str           # description of the shareable eval set
    n_winners:           int           # top-K winners requested per round
    round_deadline_sec:  float         # Phase A submission deadline


class Requester:
    """Submits a telemedicine deep-learning task to the global queue."""

    def __init__(self, task: ModelTask):
        self.task   = task
        self.wallet = ExonumWallet()

    def submit_task(self) -> ModelTask:
        return self.task


# ─────────────────────────────────────────────────────────────────────────────
# Administrator
# ─────────────────────────────────────────────────────────────────────────────

class Administrator:
    """
    Oversees the configuration of training rounds, deadlines, and the
    number of winning miners. Governance decisions (e.g. switching the
    synchronization strategy) are made transparently and recorded.
    """

    def __init__(self, n_winners: int = 2,
                 round_deadline_sec: float = 0.05,
                 slack_ratio: float = 0.5,
                 sync_mode: str = "SSP"):
        sync_mode = sync_mode.upper()
        if sync_mode not in ("SP", "SSP", "BAP"):
            raise ValueError(f"Unknown sync_mode: {sync_mode}")
        self.n_winners          = n_winners
        self.round_deadline_sec = round_deadline_sec
        self.slack_ratio        = slack_ratio
        self.sync_mode          = sync_mode
        self.wallet             = ExonumWallet()
        self._round_log: List[dict] = []

    def configure_round(self, round_num: int) -> dict:
        cfg = {
            "round":        round_num,
            "deadline_sec": self.round_deadline_sec,
            "n_winners":    self.n_winners,
            "sync_mode":    self.sync_mode,
            "slack_ratio":  self.slack_ratio,
        }
        self._round_log.append(cfg)
        return cfg

    def set_sync_mode(self, mode: str):
        """Governance action: dynamically switch synchronization strategy
        based on network performance / node-participation metrics
        (Paper Section 3.3)."""
        mode = mode.upper()
        if mode not in ("SP", "SSP", "BAP"):
            raise ValueError(f"Unknown sync_mode: {mode}")
        self.sync_mode = mode


# ─────────────────────────────────────────────────────────────────────────────
# Miner (Trainer Node)
# ─────────────────────────────────────────────────────────────────────────────

class Miner:
    """
    Trainer Node ("Miner" in the paper).

    Responsibilities
    -----------------
    Phase A (Model Proposal)     : retrieve global model, train locally on
                                    private hospital data, form a model
                                    proposal block (IPFS CID + metadata),
                                    optionally apply Differential Privacy.
    Phase B (Prediction Proposal): run inference on the shared evaluation
                                    record set, submit prediction hash.
    Phase C (Vote Proposal)      : score peers' predictions by accuracy and
                                    timeliness; cast votes.
    """

    def __init__(self, node: dict, global_model: HealthcareCNN, hp,
                 dp: Optional[DifferentialPrivacy] = None):
        self.nid     = node["node_id"]
        self.name    = node["name"]
        self.X_train = node["X_train"]; self.y_train = node["y_train"]
        self.X_val   = node["X_val"];   self.y_val   = node["y_val"]
        self.X_test  = node["X_test"];  self.y_test  = node["y_test"]
        self.hp      = hp
        self.dp      = dp
        self.model   = global_model.clone()
        self.wallet  = ExonumWallet()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def pull_global(self, global_model: HealthcareCNN):
        """Sync local model with the latest accepted global model."""
        self.model = global_model.clone()

    def local_accuracy(self) -> float:
        return self.model.accuracy(self.X_test, self.y_test)

    # ── Phase A: Model Proposal ──────────────────────────────────────────────

    def train_and_propose(self, round_num: int) -> dict:
        """
        Run local SGD training for hp.epochs_per_rnd epochs.
        If a DifferentialPrivacy mechanism is configured, the model UPDATE
        (post-train weights - pre-train weights) is clipped and noised
        before being stored on IPFS (privacy-preserving submission).

        Returns a "model proposal block" dict containing the IPFS CID,
        validation accuracy, training latency, and a submission timestamp
        used by the synchronization manager.
        """
        t0 = time.perf_counter()
        global_weights_before = self.model.flatten().copy()

        n = len(self.X_train)
        for _ in range(self.hp.epochs_per_rnd):
            idx = np.random.permutation(n)
            for start in range(0, n, self.hp.batch_size):
                batch = idx[start:start + self.hp.batch_size]
                if len(batch) == 0:
                    continue
                self.model.train_step(
                    self.X_train[batch], self.y_train[batch], lr=self.hp.lr)

        local_weights = self.model.flatten()
        dp_noise_norm = 0.0

        if self.dp is not None:
            update      = local_weights - global_weights_before
            priv_update = self.dp.privatise_update(update)
            dp_noise_norm = float(np.linalg.norm(priv_update - update))
            local_weights = (global_weights_before + priv_update).astype(np.float32)
            self.model.unflatten(local_weights)

        val_acc = self.model.accuracy(self.X_val, self.y_val)
        latency = time.perf_counter() - t0
        cid     = IPFS.store_model(local_weights)

        return {
            "trainer_id":   self.nid,
            "round":        round_num,
            "weights":      local_weights,
            "ipfs_cid":     cid,
            "val_accuracy": val_acc,
            "latency_sec":  latency,
            "dp_noise":     dp_noise_norm,
            "submit_time":  time.perf_counter(),
        }

    # ── Phase B: Prediction Proposal ─────────────────────────────────────────

    def predict_on_shared_records(self, X_eval: np.ndarray) -> np.ndarray:
        """Run inference with the freshly trained local model on the
        Requester's publicly shareable evaluation records."""
        return self.model.predict(X_eval)

    # ── Phase C: Vote Proposal ───────────────────────────────────────────────

    def vote_on_peers(self, peer_predictions: Dict[int, np.ndarray],
                      y_eval: np.ndarray,
                      peer_latencies: Dict[int, float]) -> Dict[int, float]:
        """
        Score each peer's submission using accuracy + timeliness:
            vote(peer) = 0.7 * accuracy + 0.3 * timeliness
        "Early and accurate submissions are prioritized to promote efficient
        participation and discourage adversarial behavior."
        """
        votes: Dict[int, float] = {}
        max_lat = max(peer_latencies.values(), default=1.0) or 1.0
        for pid, preds in peer_predictions.items():
            acc        = float((preds == y_eval).mean())
            timeliness = 1.0 / (1.0 + peer_latencies.get(pid, max_lat) / max_lat)
            votes[pid] = 0.7 * acc + 0.3 * timeliness
        return votes


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class Validator:
    """
    Independently assesses the quality of miner submissions using a secure
    validation dataset. Participates in decentralized voting to determine
    which updates are accepted for aggregation.
    """

    def __init__(self, vid: int, X_val: np.ndarray, y_val: np.ndarray,
                 global_model: HealthcareCNN, threshold: float):
        self.vid       = vid
        self.X_val     = X_val
        self.y_val     = y_val
        self.model     = global_model.clone()
        self.threshold = threshold
        self.wallet    = ExonumWallet()

    def validate(self, weights: np.ndarray) -> Tuple[bool, float]:
        """Accept if accuracy >= threshold OR within 5pp of current global
        model's accuracy on the secure validation set."""
        cand = self.model.clone()
        cand.unflatten(weights)
        score    = cand.accuracy(self.X_val, self.y_val)
        baseline = self.model.accuracy(self.X_val, self.y_val)
        accepted = score >= self.threshold or score >= (baseline - 0.05)
        return accepted, score

    def sync(self, global_model: HealthcareCNN):
        self.model = global_model.clone()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class Aggregator:
    """
    Integrates only the top-performing (winning) models into the global
    model using Federated Averaging (FedAvg). Also calculates the
    contribution weight (R_i) of each miner for the Reward Mechanism
    (Paper Section 3.4-E):

        R_i = (1/L) * sum_l [ (1/N_l) * sum_n |W_nl - W~_nl| ]

    where W_nl is miner i's local weight for parameter n in layer l, and
    W~_nl is the corresponding parameter in the NEW global model.
    """

    def __init__(self):
        self.wallet = ExonumWallet()

    def fed_avg(self, winner_ids: List[int],
               weight_vecs: Dict[int, np.ndarray],
               rep_weights: Dict[int, float]) -> np.ndarray:
        """Reputation- and staleness-weighted FedAvg over winning models."""
        total = sum(rep_weights.get(i, 0.0) for i in winner_ids)
        if total < 1e-12:
            total = float(len(winner_ids)) or 1.0
            rep_weights = {i: 1.0 for i in winner_ids}
        agg = None
        for wid in winner_ids:
            w = rep_weights.get(wid, 0.0) / total
            agg = (w * weight_vecs[wid]) if agg is None else agg + w * weight_vecs[wid]
        return agg

    def contribution_reward(self, local_weights: np.ndarray,
                            new_global_weights: np.ndarray,
                            layer_sizes: List[int]) -> float:
        """
        Implements the R_i formula: mean (over layers) of the mean absolute
        per-parameter difference between the miner's local weights and the
        NEW global weights, restricted to the [W_l, b_l] prefix of the
        flattened vector (the "layers" the paper refers to).
        """
        idx = 0
        layer_means = []
        for L in layer_sizes:
            local_slice  = local_weights[idx:idx + L]
            global_slice = new_global_weights[idx:idx + L]
            layer_means.append(float(np.mean(np.abs(local_slice - global_slice))))
            idx += L
        return float(np.mean(layer_means)) if layer_means else 0.0
