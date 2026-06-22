"""
run_paper_experiments.py  —  FLoBC Paper Experiments (Abuzied et al., 2024)
============================================================================
Replicates the paper's four experiments using real chest X-ray images from
4 Sri Lankan hospitals instead of MNIST / Alarm Network.

  Benchmark  : Centralized vs Decentralized (§5.1)
  Experiment 1: Trainer-to-Validator ratio (§5.2) — vary 1/2/3 validators
  Experiment 2: Reward-penalty policy (§5.3)       — graded noise k=0.0545
  Experiment 3: Synchronisation schemes (§5.4)     — BSP / SSP / BAP_1.0 / BAP_0.6

Run:
  cd C:\\Users\\SASINI\\Desktop\\research\\flobc
  python run_paper_experiments.py

Outputs saved to results/paper_experiments.json and dashboard/paper_exp_*.png
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
OUT  = os.path.join(ROOT, "dashboard")
os.makedirs(OUT, exist_ok=True)

from core.pneumonia_loader       import load_all_nodes, build_splits, HOSPITAL_NODES
from core.flobc_pneumonia_engine import FloBCPneumonia, SyncScheme

HOSP_NAMES = {nid: cfg["name"] for nid, cfg in HOSPITAL_NODES.items()}
N_ROUNDS   = 15      # enough to show convergence trend
LR         = 0.008
BATCH      = 512     # large batch → fewer BLAS calls per epoch (pure NumPy speed)
LOC_EPOCHS = 4
MAX_PER_HOSPITAL = 500   # enough for non-IID diversity
DOWNSAMPLE = 2       # pixel stride for 64x64→32x32 (1024 features, 4× fewer matmul ops)
DIV        = "=" * 68


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fresh engine per experiment (no state bleed between experiments)
# ─────────────────────────────────────────────────────────────────────────────

def make_engine(per_node_train, X_val, y_val, X_test, y_test,
                n_validators   = 3,
                sync_scheme    = SyncScheme.BSP,
                noise_profile  = None,
                uniform_trust  = False,
                pace_factors   = None,
                bap_ratio      = 1.0,
                ssp_slack      = 0.2,
                verbose        = False):
    fw = FloBCPneumonia(
        per_node_train   = per_node_train,
        X_val            = X_val,   y_val  = y_val,
        X_test           = X_test,  y_test = y_test,
        hospital_names   = HOSP_NAMES,
        sync_scheme      = sync_scheme,
        n_validators     = n_validators,
        noise_profile    = noise_profile or {},
        lr               = LR,
        batch_size       = BATCH,
        local_epochs     = LOC_EPOCHS,
        bap_majority_ratio = bap_ratio,
        ssp_slack_ratio    = ssp_slack,
        verbose          = verbose,
        pace_factors     = pace_factors or {},
    )
    if uniform_trust:
        n = len(fw.trainers)
        for tid in fw.trust.scores:
            fw.trust.scores[tid] = 1.0 / n
    return fw


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: Centralized vs Decentralized (paper §5.1)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(per_node_train, X_val, y_val, X_test, y_test):
    print(f"\n{DIV}")
    print("  BENCHMARK: Centralized vs Decentralized (paper §5.1)")
    print(f"  {N_ROUNDS} iterations | 4 trainers + 3 validators | BSP | reward-penalty")
    print(DIV)

    # Decentralized: 4T + 3V, BSP, reward-penalty active
    fw = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                     n_validators=3, sync_scheme=SyncScheme.BSP, verbose=True)
    res_dec = fw.train(n_rounds=N_ROUNDS)
    print(f"\n  Decentralized final accuracy : {res_dec['final_accuracy']:.4f}")

    # Centralized: pooled data, epochs_per_round = n_trainers (paper §5.1)
    print("\n  [Centralized] Training on pooled data ...")
    fw2 = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                      n_validators=3, verbose=False)
    res_cen = fw2.train_centralized(n_rounds=N_ROUNDS,
                                    epochs_per_round=len(fw2.trainers))
    print(f"  Centralized final accuracy   : {res_cen['final_accuracy']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(res_cen["accuracy_log"], label="Centralized",    color="#E05A2B", linewidth=2)
    ax.plot(res_dec["accuracy_log"], label="Decentralized (FLoBC)", color="#1F76B4", linewidth=2)
    ax.set_xlabel("Training Iteration"); ax.set_ylabel("Accuracy")
    ax.set_title("Benchmark: Centralized vs Decentralized (§5.1)\n"
                 "Pneumonia X-ray | 4 Hospital Nodes | BSP | Reward-Penalty")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim([max(0, min(res_cen["accuracy_log"] + res_dec["accuracy_log"]) - 0.05), 1.0])
    fig.tight_layout()
    p = os.path.join(OUT, "paper_benchmark.png")
    fig.savefig(p, dpi=120); plt.close()
    print(f"  Chart saved -> {p}")

    return {"centralized": res_cen["accuracy_log"],
            "decentralized": res_dec["accuracy_log"],
            "dec_trust": res_dec["trust_log"]}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Trainer-to-Validator ratio (paper §5.2)
# Paper uses N=10, tries 9T+1V through 5T+5V.
# With 4 fixed hospitals we vary validator count: 1V, 2V, 3V.
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1_ratio(per_node_train, X_val, y_val, X_test, y_test):
    print(f"\n{DIV}")
    print("  EXP 1: Trainer-to-Validator Ratio (paper §5.2)")
    print(f"  4 fixed trainers | vary validators 1-3 | BSP | {N_ROUNDS} rounds")
    print(DIV)

    configs  = [1, 2, 3]   # number of validators
    results  = {}

    for n_val in configs:
        label = f"4T + {n_val}V"
        print(f"\n  Running {label} ...")
        fw = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                         n_validators=n_val, sync_scheme=SyncScheme.BSP,
                         verbose=False)
        res = fw.train(n_rounds=N_ROUNDS)
        max_acc = max(res["accuracy_log"])
        max_iter = res["accuracy_log"].index(max_acc)
        print(f"  {label}: max_acc={max_acc:.4f}  at iteration {max_iter}")
        results[label] = {"accuracy_log": res["accuracy_log"],
                          "max_acc": max_acc, "max_iter": max_iter}

    # Plot: max accuracy and iteration per config (mirrors paper Fig. 4)
    labels   = list(results.keys())
    max_accs = [results[l]["max_acc"]  for l in labels]
    max_iters= [results[l]["max_iter"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(labels))
    axes[0].bar(x, max_accs, color="#1F76B4", width=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Max Accuracy"); axes[0].set_title("Maximum Accuracy per Config")
    axes[0].set_ylim([max(0, min(max_accs) - 0.05), 1.0])
    for xi, v in zip(x, max_accs):
        axes[0].text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, max_iters, color="#E05A2B", width=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Iteration Index"); axes[1].set_title("Iteration of Max Accuracy")
    for xi, v in zip(x, max_iters):
        axes[1].text(xi, v + 0.2, str(v), ha="center", fontsize=9)

    fig.suptitle("Exp 1: Trainer-to-Validator Ratio (§5.2)\n"
                 "Pneumonia X-ray | 4 Hospital Trainers | BSP | 30 Rounds")
    fig.tight_layout()
    p = os.path.join(OUT, "paper_exp1_ratio.png")
    fig.savefig(p, dpi=120); plt.close()
    print(f"\n  Chart saved -> {p}")

    # Also plot accuracy curves
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    colors = ["#1F76B4", "#E05A2B", "#2CA02C"]
    for (label, color) in zip(labels, colors):
        ax2.plot(results[label]["accuracy_log"], label=label, color=color, linewidth=2)
    ax2.set_xlabel("Training Iteration"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Exp 1: Accuracy per Trainer-to-Validator Config (§5.2)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(OUT, "paper_exp1_curves.png")
    fig2.savefig(p2, dpi=120); plt.close()
    print(f"  Chart saved -> {p2}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Reward-penalty policy (paper §5.3)
# Paper: 6 trainers, noise_std = k × trainer_index, k = 0.0545
# Adapted: 4 trainers (hospitals), same k value.
# Trainer 0 (Galle): 0, Trainer 1 (Colombo): 0.0545,
# Trainer 2 (Kandy): 0.1090, Trainer 3 (Jaffna): 0.1635
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2_reward_penalty(per_node_train, X_val, y_val, X_test, y_test):
    print(f"\n{DIV}")
    print("  EXP 2: Reward-Penalty Policy (paper §5.3)")
    print("  4 trainers | noise_std = 0.0545 × trainer_index | 3 validators | BSP")
    print(DIV)

    K = 0.0545
    nids = ["A", "B", "C", "D"]
    noise_profile = {nid: K * i for i, nid in enumerate(nids)}
    print(f"  Noise profile: { {nid: round(v,4) for nid,v in noise_profile.items()} }")

    # Scoring GROUP: trust scores adapt
    print("\n  [Scoring group] ...")
    fw_score = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                           n_validators=3, sync_scheme=SyncScheme.BSP,
                           noise_profile=noise_profile, verbose=False)
    res_score = fw_score.train(n_rounds=N_ROUNDS)

    # Control GROUP: uniform scores fixed throughout (no scoring)
    print("  [Control group - uniform trust] ...")
    fw_ctrl = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                          n_validators=3, sync_scheme=SyncScheme.BSP,
                          noise_profile=noise_profile,
                          uniform_trust=True, verbose=False)
    # Monkey-patch trust.update to be a no-op for control group
    def _no_op_update(tid, delta):
        n = len(fw_ctrl.trust.scores)
        return fw_ctrl.trust.scores[tid], fw_ctrl.trust.scores[tid]
    fw_ctrl.trust.update = _no_op_update
    res_ctrl = fw_ctrl.train(n_rounds=N_ROUNDS)

    print(f"\n  Scoring group   final accuracy : {res_score['final_accuracy']:.4f}")
    print(f"  Control group   final accuracy : {res_ctrl['final_accuracy']:.4f}")

    # Print trust score table (scoring group) — mirrors paper Table 1
    trust_log = res_score["trust_log"]
    sample_rounds = [0, 5, 10, 15, 20, 25, 29]
    print("\n  Trust scores (scoring group):")
    header = "  Trainer" + "".join(f"  Rnd {r:2d}" for r in sample_rounds)
    print(header)
    tid_map = {0: "Galle  ", 1: "Colombo", 2: "Kandy  ", 3: "Jaffna "}
    for tid in sorted(trust_log.keys()):
        row = [trust_log[tid][r] if r < len(trust_log[tid]) else 0.0
               for r in sample_rounds]
        print(f"  {tid_map.get(tid,'?')} " + "  ".join(f"{v:.4f}" for v in row))

    # Plot accuracy (mirrors paper Fig. 5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(res_score["accuracy_log"], label="Scoring (reward-penalty)",
                 color="#1F76B4", linewidth=2)
    axes[0].plot(res_ctrl["accuracy_log"],  label="Control (uniform trust)",
                 color="#E05A2B", linewidth=2, linestyle="--")
    axes[0].set_xlabel("Training Iteration"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Exp 2: Accuracy — Scoring vs Control (§5.3)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    colors = ["#2CA02C", "#1F76B4", "#E05A2B", "#9467BD"]
    for tid in sorted(trust_log.keys()):
        hosp = list(HOSP_NAMES.values())[tid]
        axes[1].plot(trust_log[tid], label=f"{hosp} (noise={noise_profile[nids[tid]]:.3f})",
                     color=colors[tid], linewidth=2)
    axes[1].set_xlabel("Training Iteration"); axes[1].set_ylabel("Trust Score")
    axes[1].set_title("Exp 2: Trust Score Evolution (§5.3)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    fig.suptitle("Exp 2: Reward-Penalty Policy — Pneumonia X-ray | 4 Hospitals")
    fig.tight_layout()
    p = os.path.join(OUT, "paper_exp2_reward_penalty.png")
    fig.savefig(p, dpi=120); plt.close()
    print(f"\n  Chart saved -> {p}")

    return {"scoring": res_score["accuracy_log"],
            "control": res_ctrl["accuracy_log"],
            "trust_log": {str(k): v for k, v in trust_log.items()},
            "noise_profile": {k: round(v, 4) for k, v in noise_profile.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Synchronisation schemes (paper §5.4)
# Paper: 6T + 3V | uniform trust | BSP, SSP, BAP_1.0, BAP_0.6 | 30 rounds
# Adapted: 4T + 3V | uniform trust | same 4 schemes
# Pace factors simulate different trainer speeds (key for SSP/BAP to differ from BSP)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3_sync(per_node_train, X_val, y_val, X_test, y_test):
    print(f"\n{DIV}")
    print("  EXP 3: Synchronisation Schemes (paper §5.4)")
    print("  4 trainers + 3 validators | uniform trust | 30 rounds")
    print("  Schemes: BSP | SSP | BAP_1.0 | BAP_0.6")
    print(DIV)

    # Simulated pace factors: A=fastest, D=slowest (different training speeds)
    pace = {"A": 1.4, "B": 1.1, "C": 0.8, "D": 0.5}
    print(f"  Trainer pace factors (simulated speeds): {pace}")

    schemes = [
        ("BSP",     SyncScheme.BSP, 1.0, 0.2),
        ("SSP",     SyncScheme.SSP, 1.0, 0.2),
        ("BAP_1.0", SyncScheme.BAP, 1.0, 0.2),
        ("BAP_0.6", SyncScheme.BAP, 0.6, 0.2),
    ]

    results = {}
    for name, scheme, bap_r, ssp_s in schemes:
        print(f"\n  Running {name} ...")
        fw = make_engine(per_node_train, X_val, y_val, X_test, y_test,
                         n_validators=3, sync_scheme=scheme,
                         uniform_trust=True, pace_factors=pace,
                         bap_ratio=bap_r, ssp_slack=ssp_s, verbose=False)
        t0  = time.time()
        res = fw.train(n_rounds=N_ROUNDS)
        elapsed = time.time() - t0
        print(f"  {name}: final_acc={res['final_accuracy']:.4f}  "
              f"max_acc={max(res['accuracy_log']):.4f}  t={elapsed:.1f}s")
        results[name] = {"accuracy_log": res["accuracy_log"],
                         "final_accuracy": res["final_accuracy"],
                         "max_accuracy": max(res["accuracy_log"]),
                         "round_times": res["round_times"]}

    # Plot: accuracy across 30 rounds (mirrors paper Fig. 6)
    colors = {"BSP": "#E05A2B", "SSP": "#1F76B4",
              "BAP_1.0": "#2CA02C", "BAP_0.6": "#9467BD"}
    styles = {"BSP": "-", "SSP": "--", "BAP_1.0": "-.", "BAP_0.6": ":"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["BSP", "SSP", "BAP_1.0", "BAP_0.6"]:
        ax.plot(results[name]["accuracy_log"],
                label=name, color=colors[name],
                linestyle=styles[name], linewidth=2)
    ax.set_xlabel("Training Iteration"); ax.set_ylabel("Accuracy")
    ax.set_title("Exp 3: Synchronisation Schemes (§5.4)\n"
                 "Pneumonia X-ray | 4 Hospital Nodes | Uniform Trust | 30 Rounds")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "paper_exp3_sync.png")
    fig.savefig(p, dpi=120); plt.close()
    print(f"\n  Chart saved -> {p}")

    # Summary table
    print("\n  Synchronisation scheme comparison:")
    print(f"  {'Scheme':<10}  {'Max Acc':>8}  {'Final Acc':>10}  {'Avg round (s)':>14}")
    print("  " + "-"*48)
    for name in ["BSP", "SSP", "BAP_1.0", "BAP_0.6"]:
        r = results[name]
        avg_t = np.mean(r["round_times"]) if r["round_times"] else 0
        print(f"  {name:<10}  {r['max_accuracy']:>8.4f}  "
              f"{r['final_accuracy']:>10.4f}  {avg_t:>14.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print(f"\n{DIV}")
    print("  FLoBC - Paper Experiment Replication")
    print("  Real chest X-ray data | 4 Sri Lankan hospitals")
    print("  Abuzied et al., Cluster Computing 2024")
    print(DIV)

    node_data = load_all_nodes()
    per_node_train, X_val, y_val, X_test, y_test, _ = build_splits(
        node_data, val_ratio=0.15, test_ratio=0.10, seed=42)

    # Speed optimisations for pure-NumPy SGD on large chest X-ray features
    rng = np.random.default_rng(42)

    def _downsample(X):
        """2x pixel stride: (n, 64*64) -> (n, 32*32). No PIL needed."""
        return X.reshape(-1, 64, 64)[:, ::DOWNSAMPLE, ::DOWNSAMPLE].reshape(len(X), -1).astype(np.float32)

    for nid in per_node_train:
        X_tr, y_tr = per_node_train[nid]
        if len(X_tr) > MAX_PER_HOSPITAL:
            idx = rng.choice(len(X_tr), MAX_PER_HOSPITAL, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        per_node_train[nid] = (_downsample(X_tr), y_tr)

    X_val   = _downsample(X_val)
    X_test  = _downsample(X_test)

    print(f"\n  Pooled validation set : {len(X_val)} samples")
    print(f"  Pooled test set       : {len(X_test)} samples")
    print(f"  Feature dimension     : {X_val.shape[1]}  (downsampled from 4096)")
    print(f"  Train samples/hospital: {MAX_PER_HOSPITAL} (subsampled)")
    print(f"  Rounds per experiment : {N_ROUNDS} | Epochs/round: {LOC_EPOCHS}")

    all_results = {}

    all_results["benchmark"]   = run_benchmark(
        per_node_train, X_val, y_val, X_test, y_test)

    all_results["exp1_ratio"]  = run_exp1_ratio(
        per_node_train, X_val, y_val, X_test, y_test)

    all_results["exp2_reward"] = run_exp2_reward_penalty(
        per_node_train, X_val, y_val, X_test, y_test)

    all_results["exp3_sync"]   = run_exp3_sync(
        per_node_train, X_val, y_val, X_test, y_test)

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_json = os.path.join(ROOT, "results", "paper_experiments.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x)
                  if hasattr(x, '__float__') else str(x))
    print(f"\n{DIV}")
    print(f"  All results saved -> {out_json}")
    print(f"  Charts saved      -> {OUT}/paper_*.png")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n  EXPERIMENT SUMMARY")
    print(f"  {'-'*64}")
    bm = all_results["benchmark"]
    print(f"  Benchmark   Centralized   : {bm['centralized'][-1]:.4f}")
    print(f"  Benchmark   Decentralized : {bm['decentralized'][-1]:.4f}")

    for label, r in all_results["exp1_ratio"].items():
        print(f"  Exp1 {label:<12} max={r['max_acc']:.4f}  @iter {r['max_iter']}")

    sc = all_results["exp2_reward"]
    print(f"  Exp2 Scoring  final={sc['scoring'][-1]:.4f}  "
          f"Control final={sc['control'][-1]:.4f}")

    for name, r in all_results["exp3_sync"].items():
        print(f"  Exp3 {name:<10} max={r['max_accuracy']:.4f}  "
              f"final={r['final_accuracy']:.4f}")
    print(f"  {'-'*64}\n")


if __name__ == "__main__":
    main()
