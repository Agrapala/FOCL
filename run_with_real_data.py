"""
FLoBC — Run All Experiments with REAL Datasets
================================================
Replaces the synthetic data with the actual paper datasets:
  - MNIST      → Experiments 0, 1, 2, 3, 8
  - Alarm Net  → Experiments 4, 5, 6
  - CIFAR-10   → Experiment 7 (vs Dis-PFL)

Run:
    python run_with_real_data.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from core.flobc_engine    import FLoBC, SyncScheme, SimpleModel
from core.real_data_loader import (load_mnist, load_alarm,
                                    load_cifar10, split_for_flobc)

os.makedirs("results", exist_ok=True)
DIV  = "═" * 62
DIV2 = "─" * 62


def _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
         n_tr, n_val, n_rounds=30,
         sync=SyncScheme.BSP, reputation=True,
         noise=None, bap_ratio=1.0, ssp_slack=0.2,
         verbose=False, export=None):
    fw = FLoBC(X_tr, y_tr, X_val, y_val, X_te, y_te,
               n_trainers=n_tr, n_validators=n_val,
               sync_scheme=sync, use_reputation=reputation,
               noise_profile=noise,
               bap_majority_ratio=bap_ratio,
               ssp_slack_ratio=ssp_slack)
    r = fw.train(n_rounds=n_rounds, verbose=verbose)
    if export:
        fw.export_chain(export)
    return r, fw


# ─────────────────────────────────────────────────────────────────────────────
# Load datasets once
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{DIV}")
print("  FLoBC — Real Dataset Experiments")
print(DIV)

# MNIST
print("\n  Loading datasets ...")
X_mnist_tr_raw, y_mnist_tr, X_mnist_te, y_mnist_te = load_mnist()
mnist_data = split_for_flobc(X_mnist_tr_raw, y_mnist_tr,
                              X_mnist_te,    y_mnist_te,
                              val_ratio=0.15, seed=0)
X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m = mnist_data

# Alarm Network
X_alarm, y_alarm = load_alarm()
from core.data_utils import train_val_test_split
X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a = \
    train_val_test_split(X_alarm, y_alarm, val_ratio=0.15, test_ratio=0.15)

# CIFAR-10
try:
    X_cf_tr_raw, y_cf_tr, X_cf_te, y_cf_te = load_cifar10()
    cifar_data = split_for_flobc(X_cf_tr_raw, y_cf_tr,
                                  X_cf_te,    y_cf_te,
                                  val_ratio=0.10, seed=0)
    X_tr_c, y_tr_c, X_val_c, y_val_c, X_te_c, y_te_c = cifar_data
    has_cifar = True
except Exception as e:
    print(f"  ~ CIFAR-10 unavailable ({e}) — Exp 7 will use MNIST")
    has_cifar = False

print(f"\n  MNIST  train={X_tr_m.shape}  val={X_val_m.shape}  test={X_te_m.shape}")
print(f"  Alarm  train={X_tr_a.shape}  val={X_val_a.shape}  test={X_te_a.shape}")
if has_cifar:
    print(f"  CIFAR  train={X_tr_c.shape}  val={X_val_c.shape}  test={X_te_c.shape}")

results = {}
t_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Exp 0 — Centralized vs Decentralized  (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 0 — Centralized vs Decentralized  (REAL MNIST)")
print(DIV)

cent_m = SimpleModel(X_tr_m.shape[1],
                     max(32, min(128, X_tr_m.shape[1]//6)),
                     10)
cent_log = [cent_m.accuracy(X_te_m, y_te_m)]
for _ in range(30):
    idx = np.random.choice(len(X_tr_m), 256, replace=False)
    cent_m.sgd_step(X_tr_m[idx], y_tr_m[idx], lr=0.05)
    cent_log.append(cent_m.accuracy(X_te_m, y_te_m))

r0, fw0 = _run(X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m,
               n_tr=7, n_val=3, n_rounds=30,
               export="results/real_blockchain_exp0.json")

gap0 = abs(cent_log[-1] - r0["accuracy_log"][-1]) * 100
print(f"  Centralized  final: {cent_log[-1]:.4f}")
print(f"  Decentralized final: {r0['accuracy_log'][-1]:.4f}")
print(f"  Gap: {gap0:.2f}%  (paper: < 0.5%)")
print(f"  Chain: {r0['chain_length']} blocks  valid={r0['chain_valid']}")
results["exp0"] = {"centralized": cent_log,
                   "decentralized": r0["accuracy_log"],
                   "gap_pct": round(gap0, 4),
                   "chain_length": r0["chain_length"],
                   "chain_valid": r0["chain_valid"]}

# ─────────────────────────────────────────────────────────────────────────────
# Exp 1 — Trainer-Validator Ratio  (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 1 — Trainer-Validator Ratio  (REAL MNIST, N=10)")
print(DIV)
res1 = {}
for n_val in range(1, 10):
    n_tr = 10 - n_val
    print(f"  {n_tr:2d}T / {n_val:2d}V ...", end="  ", flush=True)
    r, _ = _run(X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m,
                n_tr=n_tr, n_val=n_val, n_rounds=15)
    ma = max(r["accuracy_log"])
    mi = r["accuracy_log"].index(ma)
    key = f"t{n_tr}_v{n_val}"
    res1[key] = {"max_accuracy": round(ma, 4), "max_iter": mi,
                 "chain_length": r["chain_length"]}
    star = "  ← BEST" if n_tr == 7 and n_val == 3 else ""
    print(f"max={ma:.4f} @{mi}  chain={r['chain_length']}{star}")
results["exp1"] = res1

# ─────────────────────────────────────────────────────────────────────────────
# Exp 2 — Reward-Penalty  (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 2 — Reward-Penalty  (REAL MNIST, 6T/3V, k=0.0545)")
print(DIV)
noise2 = [i * 0.0545 for i in range(6)]
r2s, _ = _run(X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m,
              n_tr=6, n_val=3, n_rounds=30,
              reputation=True, noise=noise2,
              export="results/real_blockchain_exp2.json")
r2c, _ = _run(X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m,
              n_tr=6, n_val=3, n_rounds=30,
              reputation=False, noise=noise2)
print(f"  Scoring group final : {r2s['accuracy_log'][-1]:.4f}")
print(f"  Control group final : {r2c['accuracy_log'][-1]:.4f}")
print(f"  Chain: {r2s['chain_length']} blocks  valid={r2s['chain_valid']}")
print("  Final trust scores:")
for tid, vals in r2s["trust_log"].items():
    s = vals[-1] if vals else 0
    bar = "█" * int(s * 35)
    tag = "  ← PENALISED" if s < 0.02 else ("  ← TRUSTED" if s > 0.25 else "")
    print(f"    Trainer {tid}: {s:.4f}  {bar}{tag}")
results["exp2"] = {"scoring": r2s["accuracy_log"],
                   "control": r2c["accuracy_log"],
                   "trust_log": {str(k): v for k, v in r2s["trust_log"].items()},
                   "chain_length": r2s["chain_length"],
                   "chain_valid": r2s["chain_valid"]}

# ─────────────────────────────────────────────────────────────────────────────
# Exp 3 — Sync Schemes  (MNIST)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 3 — Synchronization Schemes  (REAL MNIST)")
print(DIV)
configs3 = {
    "BSP":     dict(sync=SyncScheme.BSP, bap_ratio=1.0, ssp_slack=0.0),
    "SSP":     dict(sync=SyncScheme.SSP, bap_ratio=1.0, ssp_slack=0.2),
    "BAP_1.0": dict(sync=SyncScheme.BAP, bap_ratio=1.0, ssp_slack=0.0),
    "BAP_0.6": dict(sync=SyncScheme.BAP, bap_ratio=0.6, ssp_slack=0.0),
}
res3 = {}
for name, cfg in configs3.items():
    print(f"  {name} ...", end="  ", flush=True)
    r, _ = _run(X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m,
                n_tr=6, n_val=3, n_rounds=25, **cfg,
                export=f"results/real_blockchain_exp3_{name}.json")
    avg_t = round(float(np.mean(r["round_times"])), 4)
    res3[name] = {"accuracy_log": r["accuracy_log"],
                  "final_accuracy": r["final_accuracy"],
                  "avg_round_time": avg_t,
                  "chain_length": r["chain_length"]}
    print(f"final={r['final_accuracy']:.4f}  avg_t={avg_t}s  chain={r['chain_length']}")
results["exp3"] = res3

# ─────────────────────────────────────────────────────────────────────────────
# Exp 4 — Bayesian Net: Centralized vs Decentralized  (ALARM)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 4 — Bayesian Net: Centralized vs Decentralized  (REAL ALARM)")
print(DIV)
n_cls_a = int(np.max(y_tr_a)) + 1
m4 = SimpleModel(X_tr_a.shape[1], 32, n_cls_a)
m4.sgd_step(X_tr_a, y_tr_a, lr=0.1)
cent4 = m4.accuracy(X_te_a, y_te_a)

r4, _ = _run(X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a,
             n_tr=9, n_val=1, n_rounds=1,
             export="results/real_blockchain_exp4.json")
print(f"  Centralized   acc: {cent4:.4f}  (paper: 0.8383)")
print(f"  Decentralized acc: {r4['final_accuracy']:.4f}  (paper: 0.8386)")
print(f"  Chain valid: {r4['chain_valid']}")
results["exp4"] = {"centralized": round(cent4, 4),
                   "decentralized": round(r4["final_accuracy"], 4),
                   "chain_valid": r4["chain_valid"]}

# ─────────────────────────────────────────────────────────────────────────────
# Exp 5 — Bayesian Net: T/V Ratio  (ALARM)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 5 — Bayesian Net: T/V Ratio  (REAL ALARM)")
print(DIV)
res5 = {}
for n_val in range(1, 6):
    n_tr = 10 - n_val
    if n_tr < n_val: break
    r, _ = _run(X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a,
                n_tr=n_tr, n_val=n_val, n_rounds=1)
    acc = round(r["final_accuracy"], 4)
    res5[f"t{n_tr}_v{n_val}"] = acc
    star = "  ← BEST" if n_tr == 9 and n_val == 1 else ""
    print(f"  {n_tr}T/{n_val}V  acc={acc:.4f}{star}")
results["exp5"] = res5

# ─────────────────────────────────────────────────────────────────────────────
# Exp 6 — Bayesian Net: Reward-Penalty  (ALARM)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
print("  EXP 6 — Bayesian Net: Reward-Penalty  (REAL ALARM)")
print(DIV)
noise6 = [0.0, 0.0, 0.0, 0.5, 0.7, 1.0]
r6, _ = _run(X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a,
             n_tr=6, n_val=3, n_rounds=5,
             reputation=True, noise=noise6,
             export="results/real_blockchain_exp6.json")
print("  Trust scores:")
for tid, vals in r6["trust_log"].items():
    s = vals[-1] if vals else 0
    bar = "█" * int(s * 35)
    tag = "  ← TRUSTED" if s > 0.2 else ("  ← PENALISED" if s < 0.05 else "")
    print(f"    Trainer {tid}: {s:.4f}  {bar}{tag}")
results["exp6"] = {"trust_scores": {str(k): round(v[-1], 4) if v else 0
                                    for k, v in r6["trust_log"].items()},
                   "dec_acc": round(r6["final_accuracy"], 4),
                   "chain_valid": r6["chain_valid"]}

# ─────────────────────────────────────────────────────────────────────────────
# Exp 7 — FLoBC vs Dis-PFL  (CIFAR-10 or MNIST fallback)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIV}")
label7 = "REAL CIFAR-10" if has_cifar else "MNIST (CIFAR unavailable)"
print(f"  EXP 7 — FLoBC vs Dis-PFL  ({label7})")
print(DIV)
if has_cifar:
    Xtr7, ytr7, Xv7, yv7, Xte7, yte7 = X_tr_c, y_tr_c, X_val_c, y_val_c, X_te_c, y_te_c
else:
    Xtr7, ytr7, Xv7, yv7, Xte7, yte7 = X_tr_m, y_tr_m, X_val_m, y_val_m, X_te_m, y_te_m

r7, fw7 = _run(Xtr7, ytr7, Xv7, yv7, Xte7, yte7,
               n_tr=8, n_val=2, n_rounds=30,
               reputation=True,
               export="results/real_blockchain_exp7.json")
sim7 = round(r7["final_accuracy"] * 100, 2)

paper7 = {"Local": 86.48, "FedAvg": 54.53, "FedAvg-FT": 84.96,
           "Dis-PFL": 91.05, "FLoBC(paper)": 90.35}
print(f"\n  {'Method':<20} {'Accuracy':>10}")
print(f"  {DIV2[:33]}")
for k, v in paper7.items():
    print(f"  {k:<20} {v:>10.2f}%")
print(f"  {'FLoBC(real data)':<20} {sim7:>10.2f}%")
print(f"\n  Chain: {r7['chain_length']} blocks  valid={r7['chain_valid']}")
results["exp7"] = {**paper7, "FLoBC(real_data)": sim7,
                   "chain_length": r7["chain_length"]}

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
elapsed = round(time.time() - t_start, 1)
results["meta"] = {
    "total_time_sec": elapsed,
    "datasets": "REAL (MNIST + Alarm + CIFAR-10)",
    "blockchain": "real (SHA-256 + Merkle Tree + Proof-of-Stake)",
    "cifar_available": has_cifar,
}

out = "results/real_data_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{DIV}")
print(f"  All real-data experiments complete in {elapsed}s")
print(f"  Results → {out}")
print(f"  Blockchain chains → results/real_blockchain_exp*.json")
print(DIV + "\n")
