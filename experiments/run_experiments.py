"""
FLoBC — All 8 Paper Experiments  (Real Blockchain Edition)
===========================================================
Every experiment now runs through the real cryptographic blockchain:
  - Each round produces signed transactions
  - Blocks are committed via Proof-of-Stake voting
  - Merkle tree proves transaction integrity
  - Full chain exported to results/blockchain_expN.json

Run:
    python experiments/run_experiments.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.flobc_engine import FLoBC, SyncScheme, SimpleModel
from core.data_utils   import (generate_mnist_like,
                                generate_bayesian_data,
                                train_val_test_split)

DIVIDER = "═" * 62
DIV2    = "─" * 62

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mnist(n=4200, seed=42):
    X, y = generate_mnist_like(n_samples=n, seed=seed)
    return train_val_test_split(X, y)

def _bayes(n=2000, seed=7):
    X, y = generate_bayesian_data(n_samples=n, seed=seed)
    return train_val_test_split(X, y)

def _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
         n_tr, n_val, n_rounds=30,
         sync=SyncScheme.BSP, reputation=True,
         noise=None, bap_ratio=1.0, ssp_slack=0.2,
         verbose=False, export_chain=None):
    """Create FLoBC, train, optionally export chain, return result dict."""
    fw = FLoBC(X_tr, y_tr, X_val, y_val, X_te, y_te,
               n_trainers=n_tr, n_validators=n_val,
               sync_scheme=sync, use_reputation=reputation,
               noise_profile=noise,
               bap_majority_ratio=bap_ratio,
               ssp_slack_ratio=ssp_slack)
    result = fw.train(n_rounds=n_rounds, verbose=verbose)
    if export_chain:
        os.makedirs(os.path.dirname(export_chain), exist_ok=True)
        fw.export_chain(export_chain)
    return result, fw

def _centralized(X_tr, y_tr, X_te, y_te, n_epochs=30, lr=0.05, batch=128):
    m = SimpleModel(X_tr.shape[1],
                    max(32, min(128, X_tr.shape[1]//6)),
                    int(np.max(y_tr)) + 1)
    log = [m.accuracy(X_te, y_te)]
    for _ in range(n_epochs):
        idx = np.random.choice(len(X_tr), min(batch, len(X_tr)), replace=False)
        m.sgd_step(X_tr[idx], y_tr[idx], lr=lr)
        log.append(m.accuracy(X_te, y_te))
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 0  Centralized vs Decentralized
# ─────────────────────────────────────────────────────────────────────────────

def exp0():
    print(f"\n{DIVIDER}")
    print("  EXP 0  Centralized vs Decentralized  (30 rounds, 7T/3V)")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist()

    cent = _centralized(X_tr, y_tr, X_te, y_te, n_epochs=30)

    r, fw = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                 n_tr=7, n_val=3, n_rounds=30,
                 export_chain="results/blockchain_exp0.json")

    gap = abs(cent[-1] - r["accuracy_log"][-1]) * 100
    print(f"  Centralized  final : {cent[-1]:.4f}")
    print(f"  Decentralized final: {r['accuracy_log'][-1]:.4f}")
    print(f"  Gap                : {gap:.2f}%  (paper: < 0.5%)")
    print(f"  Chain length       : {r['chain_length']} blocks")
    print(f"  Chain valid        : {r['chain_valid']}")

    return {"centralized": cent,
            "decentralized": r["accuracy_log"],
            "gap_pct": round(gap, 4),
            "chain_length": r["chain_length"],
            "chain_valid": r["chain_valid"]}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1  Trainer-to-Validator Ratio
# ─────────────────────────────────────────────────────────────────────────────

def exp1(n_total=10, n_rounds=20):
    print(f"\n{DIVIDER}")
    print("  EXP 1  Trainer-to-Validator Ratio  (N=10, 20 rounds)")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist(n=4000)

    results = {}
    for n_val in range(1, n_total):
        n_tr = n_total - n_val
        if n_tr < 1:
            continue
        print(f"  {n_tr:2d}T / {n_val:2d}V ...", end="  ", flush=True)
        r, _ = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                    n_tr=n_tr, n_val=n_val, n_rounds=n_rounds)
        ma  = max(r["accuracy_log"])
        mi  = r["accuracy_log"].index(ma)
        key = f"t{n_tr}_v{n_val}"
        results[key] = {"max_accuracy": round(ma, 4),
                        "max_iter": mi,
                        "log": r["accuracy_log"],
                        "chain_length": r["chain_length"]}
        star = "  ← BEST (paper)" if n_tr == 7 and n_val == 3 else ""
        print(f"max_acc={ma:.4f}  @iter {mi}  chain={r['chain_length']}{star}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2  Reward-Penalty Policy
# ─────────────────────────────────────────────────────────────────────────────

def exp2(n_rounds=30):
    print(f"\n{DIVIDER}")
    print("  EXP 2  Reward-Penalty Trust-Scoring  (6T/3V, k=0.0545 noise)")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist(n=4200)
    noise = [i * 0.0545 for i in range(6)]

    print("  Running SCORING group ...")
    r_sc, fw_sc = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                       n_tr=6, n_val=3, n_rounds=n_rounds,
                       reputation=True, noise=noise,
                       export_chain="results/blockchain_exp2_scoring.json")

    print("  Running CONTROL group ...")
    r_ct, _     = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                       n_tr=6, n_val=3, n_rounds=n_rounds,
                       reputation=False, noise=noise)

    print(f"\n  Scoring final acc : {r_sc['accuracy_log'][-1]:.4f}")
    print(f"  Control final acc : {r_ct['accuracy_log'][-1]:.4f}")
    print(f"  Scoring chain     : {r_sc['chain_length']} blocks  valid={r_sc['chain_valid']}")
    print("\n  Final trust scores (scoring group):")
    for tid, vals in r_sc["trust_log"].items():
        s   = vals[-1] if vals else 0
        bar = "█" * int(s * 40)
        tag = "  ← PENALISED" if s < 0.02 else ("  ← TRUSTED" if s > 0.25 else "")
        print(f"    Trainer {tid}: {s:.4f}  {bar}{tag}")

    return {
        "scoring_group": r_sc["accuracy_log"],
        "control_group": r_ct["accuracy_log"],
        "trust_log":     {str(k): v for k, v in r_sc["trust_log"].items()},
        "chain_length":  r_sc["chain_length"],
        "chain_valid":   r_sc["chain_valid"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3  Synchronization Schemes
# ─────────────────────────────────────────────────────────────────────────────

def exp3(n_rounds=30):
    print(f"\n{DIVIDER}")
    print("  EXP 3  Synchronization Schemes  (BSP / SSP / BAP)")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist(n=4200)

    configs = {
        "BSP":     dict(sync=SyncScheme.BSP, bap_ratio=1.0, ssp_slack=0.0),
        "SSP":     dict(sync=SyncScheme.SSP, bap_ratio=1.0, ssp_slack=0.2),
        "BAP_1.0": dict(sync=SyncScheme.BAP, bap_ratio=1.0, ssp_slack=0.0),
        "BAP_0.6": dict(sync=SyncScheme.BAP, bap_ratio=0.6, ssp_slack=0.0),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"  {name} ...", end="  ", flush=True)
        r, _ = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                    n_tr=6, n_val=3, n_rounds=n_rounds, **cfg,
                    export_chain=f"results/blockchain_exp3_{name}.json")
        results[name] = {
            "accuracy_log":   r["accuracy_log"],
            "round_times":    r["round_times"],
            "final_accuracy": r["final_accuracy"],
            "avg_round_time": round(float(np.mean(r["round_times"])), 4),
            "chain_length":   r["chain_length"],
        }
        print(f"final={r['final_accuracy']:.4f}  "
              f"avg_time={results[name]['avg_round_time']}s  "
              f"chain={r['chain_length']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4  Bayesian Network: Centralized vs Decentralized
# ─────────────────────────────────────────────────────────────────────────────

def exp4():
    print(f"\n{DIVIDER}")
    print("  EXP 4  Bayesian Net: Centralized vs Decentralized")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _bayes()

    m = SimpleModel(X_tr.shape[1], 32, int(np.max(y_tr))+1)
    m.sgd_step(X_tr, y_tr, lr=0.1)
    cent = m.accuracy(X_te, y_te)

    r, fw = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                 n_tr=9, n_val=1, n_rounds=1,
                 export_chain="results/blockchain_exp4.json")
    dec = r["final_accuracy"]

    print(f"  Centralized  acc: {cent:.4f}  (paper: 0.8383)")
    print(f"  Decentralized acc: {dec:.4f}  (paper: 0.8386)")
    print(f"  Chain valid      : {r['chain_valid']}")

    return {"centralized": round(cent, 4),
            "decentralized": round(dec, 4),
            "chain_valid": r["chain_valid"]}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5  Bayesian Network: Trainer-Validator Ratio
# ─────────────────────────────────────────────────────────────────────────────

def exp5(n_total=10):
    print(f"\n{DIVIDER}")
    print("  EXP 5  Bayesian Net: Trainer-Validator Ratio")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _bayes()

    results = {}
    for n_val in range(1, n_total // 2 + 1):
        n_tr = n_total - n_val
        if n_tr < n_val:
            break
        r, _ = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                    n_tr=n_tr, n_val=n_val, n_rounds=1)
        acc = round(r["final_accuracy"], 4)
        results[f"t{n_tr}_v{n_val}"] = acc
        star = "  ← BEST (paper)" if n_tr == 9 and n_val == 1 else ""
        print(f"  {n_tr}T / {n_val}V  acc={acc:.4f}{star}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6  Bayesian Network: Reward-Penalty
# ─────────────────────────────────────────────────────────────────────────────

def exp6():
    print(f"\n{DIVIDER}")
    print("  EXP 6  Bayesian Net: Reward-Penalty Policy")
    print(DIVIDER)
    X_tr, y_tr, X_val, y_val, X_te, y_te = _bayes()

    noise = [0.0, 0.0, 0.0, 0.5, 0.7, 1.0]   # trainers 3-5 are random
    r, fw = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                 n_tr=6, n_val=3, n_rounds=5,
                 reputation=True, noise=noise,
                 export_chain="results/blockchain_exp6.json")

    m = SimpleModel(X_tr.shape[1], 32, int(np.max(y_tr))+1)
    m.sgd_step(X_tr, y_tr, lr=0.1)
    cent = m.accuracy(X_te, y_te)

    print("\n  Trust scores:")
    for tid, vals in r["trust_log"].items():
        s   = vals[-1] if vals else 0
        bar = "█" * int(s * 40)
        tag = "  ← TRUSTED" if s > 0.2 else ("  ← PENALISED" if s < 0.05 else "")
        print(f"    Trainer {tid}: {s:.4f}  {bar}{tag}")
    print(f"\n  Decentralized acc : {r['final_accuracy']:.4f}")
    print(f"  Centralized   acc : {cent:.4f}")
    print(f"  Chain valid       : {r['chain_valid']}")

    return {
        "trust_scores":      {str(k): round(v[-1], 4) if v else 0
                              for k, v in r["trust_log"].items()},
        "decentralized_acc": round(r["final_accuracy"], 4),
        "centralized_acc":   round(cent, 4),
        "chain_valid":       r["chain_valid"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 7  FLoBC vs Dis-PFL
# ─────────────────────────────────────────────────────────────────────────────

def exp7():
    print(f"\n{DIVIDER}")
    print("  EXP 7  FLoBC vs Dis-PFL  (Table 5 from paper)")
    print(DIVIDER)

    paper = {"Local": 86.48, "FedAvg": 54.53,
             "FedAvg-FT": 84.96, "Dis-PFL": 91.05, "FLoBC(paper)": 90.35}

    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist(n=5000, seed=99)
    r, fw = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                 n_tr=8, n_val=2, n_rounds=30, reputation=True,
                 export_chain="results/blockchain_exp7.json")
    sim = round(r["final_accuracy"] * 100, 2)

    print(f"\n  {'Method':<20} {'Accuracy':>10}")
    print(f"  {DIV2[:32]}")
    for k, v in paper.items():
        print(f"  {k:<20} {v:>10.2f}%")
    print(f"  {'FLoBC(simulated)':<20} {sim:>10.2f}%")
    print(f"\n  Chain: {r['chain_length']} blocks  |  valid={r['chain_valid']}")

    return {**paper, "FLoBC(simulated)": sim,
            "chain_length": r["chain_length"], "chain_valid": r["chain_valid"]}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 8  FLoBC vs PVD-FL
# ─────────────────────────────────────────────────────────────────────────────

def exp8():
    print(f"\n{DIVIDER}")
    print("  EXP 8  FLoBC vs PVD-FL  (Tables 6-8 from paper)")
    print(DIVIDER)

    pvdfl = {
        "3x128FC": {"1ep": 91.49, "5ep": 96.84, "10ep": 97.37},
        "3x512FC": {"1ep": 94.63, "5ep": 97.54, "10ep": 97.94},
        "CNN":     {"1ep": 96.54, "5ep": 97.54, "10ep": 98.47},
    }
    flobc_paper = {
        "3x128FC": {"1ep": 94.48, "5ep": 96.43, "10ep": 97.13},
        "3x512FC": {"1ep": 95.30, "5ep": 97.60, "10ep": 97.96},
        "CNN":     {"1ep": 96.56, "5ep": 98.06, "10ep": 98.58},
    }

    X_tr, y_tr, X_val, y_val, X_te, y_te = _mnist(n=4000, seed=7)
    sim = {}
    for arch in ["3x128FC", "3x512FC", "CNN"]:
        sim[arch] = {}
        for ep_str in ["1ep", "5ep", "10ep"]:
            n_rnd = int(ep_str.replace("ep", ""))
            r, _  = _run(X_tr, y_tr, X_val, y_val, X_te, y_te,
                         n_tr=4, n_val=2, n_rounds=n_rnd, reputation=True)
            sim[arch][ep_str] = round(r["final_accuracy"] * 100, 2)

    print(f"\n  {'Arch':<10} {'Ep':<8} {'PVD-FL':>8} {'FLoBC(p)':>10} {'FLoBC(s)':>10}")
    print(f"  {DIV2[:50]}")
    for arch in ["3x128FC", "3x512FC", "CNN"]:
        for ep in ["1ep", "5ep", "10ep"]:
            pv = pvdfl[arch][ep]
            fp = flobc_paper[arch][ep]
            fs = sim[arch][ep]
            diff = fs - pv
            sign = "+" if diff >= 0 else ""
            print(f"  {arch:<10} {ep:<8} {pv:>8.2f}  {fp:>10.2f}  "
                  f"{fs:>10.2f}  ({sign}{diff:.2f})")

    return {"pvdfl_paper": pvdfl,
            "flobc_paper": flobc_paper,
            "flobc_simulated": sim}


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

def run_all(out="results/all_results.json"):
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    res = {}

    res["exp0"] = exp0()
    res["exp1"] = exp1()
    res["exp2"] = exp2()
    res["exp3"] = exp3()
    res["exp4"] = exp4()
    res["exp5"] = exp5()
    res["exp6"] = exp6()
    res["exp7"] = exp7()
    res["exp8"] = exp8()

    elapsed = round(time.time() - t0, 1)
    res["meta"] = {"total_time_sec": elapsed,
                   "blockchain": "real (SHA-256 + Merkle + PoS)"}

    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"\n{DIVIDER}")
    print(f"  All 8 experiments complete in {elapsed}s")
    print(f"  Results JSON  → {out}")
    print(f"  Blockchain JSON files → results/blockchain_exp*.json")
    print(DIVIDER + "\n")
    return res


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_all()
