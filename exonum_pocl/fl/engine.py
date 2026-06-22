"""
FLoBC-PoCL FL Engine — Exonum + pBFT/PoCL-pBFT/PoS + Healthcare CNN + DP
==========================================================================
Topic: Exploring and Implementing Robust Privacy Mechanisms for Healthcare
       Data in Telemedicine Systems with Blockchain & Federated Learning

This engine implements the FULL methodology of Paper Section 3
("Blockchain-based Federated Learning Framework (FLoBC-PoCL)"):

  3.1 Parallel Training
      - Data-Parallelism: each hospital node (Miner) trains a full
        HealthcareCNN on its local chest X-ray partition.
      - FedAvg aggregation of winning models (fl/roles.py: Aggregator).

  3.2 Decentralization — 5 Actor Roles (fl/roles.py)
      - Administrator : configures rounds, deadlines, n_winners, sync mode
      - Requester     : submits the ModelTask (architecture + eval set)
      - Miners        : train locally, propose models, predict, vote
      - Validators    : secure-dataset validation + decentralized voting
      - Aggregator    : FedAvg + R_i contribution-reward computation

  3.3 Synchronization — SP / SSP / BAP (fl/sync_strategies.py)
      - Administrator selects / dynamically switches the sync strategy.

  3.4 PoCL Mechanism — 5 Phases (fl/pocl_protocol.py)
      A. Model Proposal      -> Miner.train_and_propose()
      B. Prediction Proposal -> PoCLProtocol.collect_predictions()
      C. Vote Proposal       -> PoCLProtocol.collect_votes() / aggregate_votes()
      D. Winner Selection    -> PoCLProtocol.select_winners()
      E. Reward Mechanism    -> PoCLProtocol.compute_rewards() + R_i
      F. Block Creation      -> ExonumBlockchain.propose_block() via consensus

  Privacy: Differential Privacy (Gaussian mechanism on model UPDATES,
           fl/cnn_model.py: DifferentialPrivacy) — Section 2.4 healthcare
           privacy requirements.

  Off-chain storage: IPFS (ipfs/ipfs_node.py) for full model weights;
           only CIDs + metadata go on-chain (Exonum, blockchain/).

  Consensus: selectable per run — "pbft" | "pocl_pbft" | "pos"
           (consensus/pbft.py, consensus/pocl_pbft.py)

Paper targets (telemedicine healthcare extension):
  - Node A: >=89% accuracy (baseline 84.73%)
  - All nodes: >=5pp FL improvement over local-only training
  - IPFS: meaningful gas cost reduction vs on-chain storage
  - pBFT vs PoCL-pBFT vs PoS: full comparison (accuracy/delay/gas/fault tol.)
  - SP vs SSP vs BAP: synchronization strategy comparison
"""

import time
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from fl.cnn_model       import HealthcareCNN, DifferentialPrivacy
from fl.data_loader     import load_all_nodes, global_test_set
from fl.roles           import (
    Administrator, Requester, ModelTask, Miner, Validator, Aggregator,
)
from fl.sync_strategies import SyncManager
from fl.pocl_protocol    import PoCLProtocol

from blockchain.crypto      import ExonumWallet
from blockchain.transaction import (
    make_model_update_tx, make_validation_tx, make_trust_update_tx,
    make_global_model_tx, make_pocl_reward_tx, make_pbft_vote_tx,
    make_prediction_proposal_tx, make_vote_proposal_tx,
)
from blockchain.chain import ExonumBlockchain

from ipfs.ipfs_node import IPFS

from consensus.pbft      import PBFTConsensus
from consensus.pocl_pbft import PoCLPBFTConsensus, PoSConsensus, PoCLRound


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperParams:
    # ── CNN / Optimisation ──────────────────────────────────────────────────
    lr:             float       = 3e-4
    batch_size:     int         = 32
    epochs_per_rnd: int         = 5
    hidden_dims:    List[int]   = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rate:   float       = 0.20
    weight_decay:   float       = 1e-4

    # ── PoCL Phase D (Winner Selection) weighting ──────────────────────────
    top_k_winners:  int         = 2
    val_threshold:  float       = 0.45
    w_validator:    float       = 0.5
    w_peer_vote:    float       = 0.3
    w_accuracy:     float       = 0.2

    # ── Synchronization (Paper 3.3) ─────────────────────────────────────────
    sync_mode:          str     = "SSP"   # "SP" | "SSP" | "BAP"
    round_deadline_sec: float    = 0.05
    slack_ratio:        float    = 0.5

    # ── Differential Privacy (Section 2.4) ──────────────────────────────────
    use_dp:        bool  = True
    dp_epsilon:    float = 8.0
    dp_delta:      float = 1e-5
    dp_clip_norm:  float = 1.0

    # ── Reward Mechanism (Phase E) ───────────────────────────────────────────
    base_reward:   float = 0.08
    penalty:       float = -0.08


# ─────────────────────────────────────────────────────────────────────────────
# Reputation / Trust service
# ─────────────────────────────────────────────────────────────────────────────

class ReputationService:
    def __init__(self, node_ids: List[int]):
        n = len(node_ids)
        self.scores:  Dict[int, float]       = {i: 1.0 / n for i in node_ids}
        self.history: Dict[int, List[float]] = {i: [1.0 / n] for i in node_ids}

    def update(self, nid: int, delta: float) -> Tuple[float, float]:
        """delta > 0: reward, < 0: penalty. Returns (old_score, new_score)."""
        old     = self.scores[nid]
        raw_new = float(np.clip(old + delta, 0.01, 2.0))
        self.scores[nid] = raw_new
        total = sum(self.scores.values()) or 1e-9
        for k in self.scores:
            self.scores[k] /= total
        for nid2 in self.scores:
            self.history[nid2].append(self.scores[nid2])
        return old, self.scores[nid]

    def weight(self, nid: int) -> float:
        return self.scores.get(nid, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main FL Engine
# ─────────────────────────────────────────────────────────────────────────────

class FLoCBPoCL:
    """
    FLoBC-PoCL Engine on Exonum Blockchain.

    consensus_mode: "pbft" | "pocl_pbft" | "pos"
    """

    def __init__(
        self,
        hp:              HyperParams = None,
        consensus_mode:  str         = "pocl_pbft",
        n_validators:    int         = 3,
        verbose:         bool        = True,
    ):
        self.hp             = hp or HyperParams()
        self.consensus_mode = consensus_mode
        self.verbose        = verbose

        # ── Data ─────────────────────────────────────────────────────────────
        self.nodes                   = load_all_nodes()
        self.X_global, self.y_global = global_test_set()

        # ── Global model ─────────────────────────────────────────────────────
        self.global_model = HealthcareCNN(
            input_dim    = 1024,
            hidden_dims  = list(self.hp.hidden_dims),
            dropout_rate = self.hp.dropout_rate,
            weight_decay = self.hp.weight_decay,
            seed         = 42,
        )
        self._layer_sizes = self.global_model.layer_param_sizes()

        # ── Section 3.2: Actor Roles ────────────────────────────────────────
        self.administrator = Administrator(
            n_winners          = self.hp.top_k_winners,
            round_deadline_sec = self.hp.round_deadline_sec,
            slack_ratio        = self.hp.slack_ratio,
            sync_mode          = self.hp.sync_mode,
        )
        self.requester = Requester(ModelTask(
            task_id             = "telemedicine-pneumonia-cnn",
            architecture        = f"HealthcareCNN-{'-'.join(map(str, self.hp.hidden_dims))}",
            hidden_dims         = list(self.hp.hidden_dims),
            public_dataset_desc = "Held-out balanced chest X-ray evaluation set (global_test_set)",
            n_winners           = self.hp.top_k_winners,
            round_deadline_sec  = self.hp.round_deadline_sec,
        ))

        self.dp = (DifferentialPrivacy(
                        epsilon=self.hp.dp_epsilon,
                        delta=self.hp.dp_delta,
                        clip_norm=self.hp.dp_clip_norm)
                   if self.hp.use_dp else None)

        self.miners: List[Miner] = [
            Miner(node, self.global_model, self.hp, dp=self.dp)
            for node in self.nodes
        ]
        self.miners_by_id: Dict[int, Miner] = {m.nid: m for m in self.miners}
        n_nodes = len(self.miners)

        # Validators — blended val sets from all hospital nodes
        all_X_val = np.vstack([nd["X_val"] for nd in self.nodes])
        all_y_val = np.concatenate([nd["y_val"] for nd in self.nodes])
        val_splits = np.array_split(np.arange(len(all_X_val)), n_validators)
        self.validators: List[Validator] = [
            Validator(j, all_X_val[val_splits[j]], all_y_val[val_splits[j]],
                      self.global_model, self.hp.val_threshold)
            for j in range(n_validators)
        ]

        self.aggregator = Aggregator()

        # ── Section 3.3: Synchronization ────────────────────────────────────
        self.sync_manager = SyncManager(
            mode=self.hp.sync_mode,
            base_deadline_sec=self.hp.round_deadline_sec,
            slack_ratio=self.hp.slack_ratio,
        )

        # ── Section 3.4: PoCL Protocol (Phases B-E) ─────────────────────────
        self.pocl = PoCLProtocol(
            top_k       = self.hp.top_k_winners,
            w_validator = self.hp.w_validator,
            w_peer_vote = self.hp.w_peer_vote,
            w_accuracy  = self.hp.w_accuracy,
            base_reward = self.hp.base_reward,
            penalty     = self.hp.penalty,
        )

        # Reputation
        self.reputation = ReputationService(list(range(n_nodes)))

        # Blockchain
        self.chain      = ExonumBlockchain()
        self._fw_wallet = ExonumWallet()   # "smart contract" / framework wallet

        # Consensus engine
        val_addrs = {v.wallet.address: 1.0 / n_validators for v in self.validators}
        if consensus_mode == "pbft":
            self.consensus_engine = PBFTConsensus(
                [v.wallet for v in self.validators])
        elif consensus_mode == "pocl_pbft":
            self.consensus_engine = PoCLPBFTConsensus(
                [v.wallet for v in self.validators],
                val_addrs,
                top_k=self.hp.top_k_winners,
            )
        elif consensus_mode == "pos":
            self.consensus_engine = PoSConsensus(
                [v.wallet for v in self.validators], val_addrs)
        else:
            raise ValueError(f"Unknown consensus_mode: {consensus_mode!r}")

        # ── Logging ──────────────────────────────────────────────────────────
        self.global_acc_log:   List[float]            = []
        self.per_node_acc_log: Dict[int, List[float]] = defaultdict(list)
        self.round_times:      List[float]            = []
        self.consensus_log:    List[dict]             = []
        self.trust_log:        Dict[int, List[float]] = defaultdict(list)
        self.sync_log:         List[dict]             = []
        self.winners_log:      List[List[int]]        = []
        self.dp_noise_log:     Dict[int, List[float]] = defaultdict(list)
        self._baseline_accs:   Dict[int, float]       = {}

    # ── Record initial baseline ─────────────────────────────────────────────

    def record_baselines(self):
        print("\n  [Baseline] Local-only accuracy (before any FL):")
        for m in self.miners:
            acc = m.local_accuracy()
            self._baseline_accs[m.nid] = acc
            print(f"    {m.name}: {acc:.4f} ({acc*100:.2f}%)")
        g_acc = self.global_model.accuracy(self.X_global, self.y_global)
        print(f"    Global (initial): {g_acc:.4f} ({g_acc*100:.2f}%)")
        if self.dp is not None:
            print(f"    [Privacy] Differential Privacy ENABLED: {self.dp.info()}")
        else:
            print(f"    [Privacy] Differential Privacy DISABLED")
        print(f"    [Sync] Strategy: {self.sync_manager.mode} "
              f"(deadline={self.sync_manager.base_deadline_sec}s, "
              f"slack={self.sync_manager.slack_ratio})")

    # ── Push new global model to all actors ─────────────────────────────────

    def _push_global(self, new_weights: np.ndarray):
        self.global_model.unflatten(new_weights)
        for m in self.miners:
            m.pull_global(self.global_model)
        for v in self.validators:
            v.sync(self.global_model)

    # ── Single FL round (PoCL Phases A-F) ───────────────────────────────────

    def _run_round(self, round_num: int) -> dict:
        t_round_start = time.perf_counter()
        block_txs = []
        n_miners  = len(self.miners)
        all_ids   = list(range(n_miners))

        # Administrator configures this round (deadlines, n_winners, sync mode)
        round_cfg = self.administrator.configure_round(round_num)

        # ── Phase A: Model Proposal ──────────────────────────────────────────
        proposals: Dict[int, dict] = {}
        for m in self.miners:
            prop = m.train_and_propose(round_num)
            proposals[m.nid] = prop
            self.dp_noise_log[m.nid].append(prop["dp_noise"])
            block_txs.append(make_model_update_tx(
                wallet=m.wallet, trainer_id=m.nid, round_num=round_num,
                ipfs_cid=prop["ipfs_cid"], accuracy=prop["val_accuracy"],
                noise_level=prop["dp_noise"],
            ))

        # ── Synchronization (SP/SSP/BAP) ─────────────────────────────────────
        included_ids, stale_w, late_flags = self.sync_manager.process_submissions(proposals)

        # ── Validator pipeline (secure dataset, decentralized voting) ───────
        accepted_ids: List[int] = []
        val_scores: Dict[int, List[float]] = defaultdict(list)
        for tid in included_ids:
            votes_ok = 0
            for v in self.validators:
                accepted, score = v.validate(proposals[tid]["weights"])
                val_scores[tid].append(score)
                if accepted:
                    votes_ok += 1
                block_txs.append(make_validation_tx(
                    wallet=v.wallet, validator_id=v.vid, trainer_id=tid,
                    round_num=round_num, accepted=accepted, score=score,
                ))
            quorum_ok = votes_ok >= max(1, len(self.validators) * 2 // 3)
            if quorum_ok:
                accepted_ids.append(tid)
        validator_scores = {tid: float(np.mean(val_scores[tid]))
                             for tid in accepted_ids}

        # ── Phase B: Prediction Proposal ─────────────────────────────────────
        predictions = self.pocl.collect_predictions(
            self.miners_by_id, accepted_ids, self.X_global)
        for tid in accepted_ids:
            phash = self.pocl.predictions_hash(predictions[tid])
            block_txs.append(make_prediction_proposal_tx(
                wallet=self.miners_by_id[tid].wallet, trainer_id=tid,
                round_num=round_num, predictions_hash=phash,
                n_predictions=len(predictions[tid]),
            ))

        # ── Phase C: Vote Proposal ────────────────────────────────────────────
        latencies = {tid: proposals[tid]["latency_sec"] for tid in accepted_ids}
        peer_votes = self.pocl.collect_votes(
            self.miners_by_id, accepted_ids, predictions, self.y_global, latencies)
        for voter_id, votes in peer_votes.items():
            block_txs.append(make_vote_proposal_tx(
                wallet=self.miners_by_id[voter_id].wallet, voter_id=voter_id,
                round_num=round_num, votes=votes,
            ))
        aggregated_votes = self.pocl.aggregate_votes(peer_votes)

        # ── Phase D: Winner Selection ────────────────────────────────────────
        val_acc = {tid: proposals[tid]["val_accuracy"] for tid in accepted_ids}
        if accepted_ids:
            winners, combined_scores = self.pocl.select_winners(
                accepted_ids, val_acc, validator_scores, aggregated_votes)
            # Verify winners against their original IPFS CIDs (Phase D check)
            winners = [w for w in winners if IPFS.exists(proposals[w]["ipfs_cid"])]
        else:
            winners, combined_scores = [], {}

        # ── Aggregation (FedAvg, reputation + staleness weighted) ───────────
        if winners:
            weight_vecs = {tid: proposals[tid]["weights"] for tid in winners}
            rep_weights = {tid: self.reputation.weight(tid) * stale_w.get(tid, 1.0)
                           for tid in winners}
            new_w = self.aggregator.fed_avg(winners, weight_vecs, rep_weights)
            self._push_global(new_w)
            cid_global = IPFS.store_model(new_w)
            state_hash = IPFS._make_cid(new_w.tobytes())
        else:
            new_w      = self.global_model.flatten()
            cid_global = "QmNO_UPDATE"
            state_hash = "no_update_" + str(round_num)

        g_acc = self.global_model.accuracy(self.X_global, self.y_global)

        # ── Phase E: Reward Mechanism (R_i contribution) ─────────────────────
        contribution_scores: Dict[int, float] = {}
        for tid in accepted_ids:
            contribution_scores[tid] = self.aggregator.contribution_reward(
                proposals[tid]["weights"], new_w, self._layer_sizes)

        rewards: Dict[int, float] = self.pocl.compute_rewards(
            accepted_ids, winners, contribution_scores) if accepted_ids else {}

        # Apply rewards/penalties + record trust + PoCL-reward transactions
        for tid in all_ids:
            if tid in accepted_ids:
                delta  = rewards.get(tid, self.hp.penalty)
                reason = "winner" if tid in winners else "accepted_not_winner"
                contrib = contribution_scores.get(tid, 0.0)
            elif tid in included_ids:
                delta, reason, contrib = self.hp.penalty, "rejected_by_validators", 0.0
            else:
                delta, reason, contrib = self.hp.penalty, "late_dropped_by_sync", 0.0

            old_sc, new_sc = self.reputation.update(tid, delta)
            block_txs.append(make_trust_update_tx(
                wallet=self._fw_wallet, trainer_id=tid,
                old_score=old_sc, new_score=new_sc, round_num=round_num,
            ))
            block_txs.append(make_pocl_reward_tx(
                wallet=self._fw_wallet, trainer_id=tid,
                reward=delta, round_num=round_num, reason=reason,
                contribution=contrib,
            ))
            self.trust_log[tid].append(new_sc)

        # ── Phase F: Block Creation (header summary) ─────────────────────────
        block_txs.append(make_global_model_tx(
            wallet=self._fw_wallet, round_num=round_num, ipfs_cid=cid_global,
            accuracy=g_acc, accepted_trainers=accepted_ids,
            trust_scores=dict(self.reputation.scores),
            winners=winners, sync_mode=self.sync_manager.mode,
        ))

        # ── Consensus (pBFT / PoCL-pBFT / PoS) ───────────────────────────────
        tx_hashes = [tx.tx_hash for tx in block_txs]
        cand_hash = hashlib.sha256("".join(tx_hashes).encode()).hexdigest()

        if self.consensus_mode == "pocl_pbft":
            pocl_round = PoCLRound(
                round_num    = round_num,
                miner_scores = {tid: val_acc.get(tid, 0.0) for tid in all_ids},
                miner_timing = {tid: proposals[tid]["latency_sec"] for tid in all_ids},
                top_k        = self.hp.top_k_winners,
            )
            consensus_result, _ = self.consensus_engine.run(
                cand_hash, len(block_txs), pocl_round)
        else:
            consensus_result = self.consensus_engine.run(cand_hash, len(block_txs))

        if consensus_result.committed:
            self.chain.propose_block(
                transactions   = block_txs,
                proposer_id    = self.validators[0].wallet.address,
                round_num      = consensus_result.pbft_round,
                precommit_sigs = consensus_result.precommit_sigs,
                state_hash     = state_hash,
            )

        # ── Logging ───────────────────────────────────────────────────────────
        self.global_acc_log.append(g_acc)
        for m in self.miners:
            self.per_node_acc_log[m.nid].append(
                m.model.accuracy(m.X_test, m.y_test))

        round_time = time.perf_counter() - t_round_start
        self.round_times.append(round_time)
        self.winners_log.append(winners)
        self.sync_log.append({
            "round":       round_num,
            "mode":        self.sync_manager.mode,
            "included":    included_ids,
            "late_flags":  late_flags,
            "stale_w":     {k: round(v, 3) for k, v in stale_w.items()},
        })
        self.consensus_log.append({
            "round":          round_num,
            "consensus":      consensus_result.consensus_name,
            "committed":      consensus_result.committed,
            "delay_sec":      round(consensus_result.delay_sec, 6),
            "gas_equivalent": consensus_result.gas_equivalent,
            "yes_votes":      consensus_result.yes_votes,
            "total_nodes":    consensus_result.total_nodes,
            "global_acc":     round(g_acc, 6),
            "accepted_nodes": accepted_ids,
            "winners":        winners,
        })

        return {
            "round_num":    round_num,
            "global_acc":   g_acc,
            "included_ids": included_ids,
            "accepted_ids": accepted_ids,
            "winners":      winners,
            "local_accs":   {k: round(v, 4) for k, v in val_acc.items()},
            "consensus":    consensus_result.consensus_name,
            "committed":    consensus_result.committed,
            "delay_sec":    consensus_result.delay_sec,
            "gas_eq":       consensus_result.gas_equivalent,
            "round_sec":    round_time,
        }

    # ── Full training run ───────────────────────────────────────────────────

    def train(self, n_rounds: int = 20) -> dict:
        self.record_baselines()
        g_acc0 = self.global_model.accuracy(self.X_global, self.y_global)
        self.global_acc_log.append(g_acc0)

        print(f"\n  [FL] Starting {n_rounds} rounds | "
              f"Consensus: {self.consensus_mode.upper()} | "
              f"Sync: {self.sync_manager.mode} | "
              f"DP: {'ON' if self.dp else 'OFF'} | "
              f"Blockchain: Exonum\n")

        for rnd in range(1, n_rounds + 1):
            result = self._run_round(rnd)
            if self.verbose:
                node_str = " | ".join(
                    f"N{nid}:{self.per_node_acc_log[nid][-1]:.3f}"
                    for nid in range(len(self.miners))
                )
                print(f"  Rnd {rnd:3d} | G:{result['global_acc']:.4f} | "
                      f"{node_str} | "
                      f"win={result['winners']} | "
                      f"acc_ids={result['accepted_ids']} | "
                      f"delay={result['delay_sec']*1000:.1f}ms | "
                      f"gas={result['gas_eq']}")

        return self._summary()

    def _summary(self) -> dict:
        n_nodes = len(self.miners)
        final_per_node = {
            nid: self.per_node_acc_log[nid][-1] if self.per_node_acc_log[nid] else 0.0
            for nid in range(n_nodes)
        }
        improvements = {
            nid: round((final_per_node[nid] - self._baseline_accs.get(nid, 0.0)) * 100, 2)
            for nid in range(n_nodes)
        }
        avg_delay = (float(np.mean([c["delay_sec"] for c in self.consensus_log]))
                     if self.consensus_log else 0.0)
        avg_gas   = (float(np.mean([c["gas_equivalent"] for c in self.consensus_log]))
                     if self.consensus_log else 0.0)
        ipfs_stats = IPFS.stats()

        # Sync strategy statistics
        n_rounds = len(self.sync_log)
        total_late = sum(sum(s["late_flags"].values()) for s in self.sync_log)
        total_slots = n_rounds * n_nodes
        late_pct = round(100.0 * total_late / total_slots, 2) if total_slots else 0.0

        return {
            "consensus_mode":         self.consensus_mode,
            "sync_mode":              self.sync_manager.mode,
            "dp_enabled":             self.dp is not None,
            "dp_info":                self.dp.info() if self.dp else None,
            "blockchain":             "Exonum (simulated)",
            "n_rounds":               len(self.global_acc_log) - 1,
            "global_acc_final":       round(self.global_acc_log[-1], 6),
            "global_acc_log":         [round(x, 6) for x in self.global_acc_log],
            "per_node_acc_final":     {str(k): round(v, 6) for k, v in final_per_node.items()},
            "per_node_acc_log":       {str(k): [round(x, 6) for x in v]
                                       for k, v in self.per_node_acc_log.items()},
            "baseline_accs":          {str(k): round(v, 6) for k, v in self._baseline_accs.items()},
            "fl_improvement_pct":     {str(k): v for k, v in improvements.items()},
            "chain_length":           self.chain.length(),
            "chain_valid":            self.chain.is_chain_valid(),
            "avg_consensus_delay_ms": round(avg_delay * 1000, 3),
            "avg_gas_equivalent":     round(avg_gas, 1),
            "consensus_log":          self.consensus_log,
            "ipfs_stats":             ipfs_stats,
            "trust_log":              {str(k): [round(x, 6) for x in v]
                                       for k, v in self.trust_log.items()},
            "winners_log":            self.winners_log,
            "sync_log":               self.sync_log,
            "sync_late_pct":          late_pct,
            "dp_noise_log":           {str(k): [round(x, 8) for x in v]
                                       for k, v in self.dp_noise_log.items()},
            "pocl_config":            self.pocl.info(),
        }

    def export_chain(self, path: str):
        self.chain.export_json(path)
        print(f"  [Exonum] Chain exported -> {path}")
