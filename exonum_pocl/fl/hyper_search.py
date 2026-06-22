"""
Hyperparameter Tuning for Healthcare CNN
=========================================
Grid search over key hyperparameters to close the accuracy gap.
Target: Node A >= 89% (up from ~75-80% baseline)

Search space:
  lr:             [1e-3, 5e-4, 2e-4]
  batch_size:     [32, 64]
  epochs_per_rnd: [3, 5, 8]
  hidden_dims:    [[256,128,64], [512,256,128]]
  dropout_rate:   [0.15, 0.20]

Runs a short 5-round FL session for each config, returns best config.
"""

import itertools
import numpy as np
from typing import List, Tuple, Dict, Any


SEARCH_SPACE = {
    "lr":             [1e-3, 5e-4, 2e-4],
    "batch_size":     [32, 64],
    "epochs_per_rnd": [3, 5, 8],
    "hidden_dims":    [[256, 128, 64], [512, 256, 128]],
    "dropout_rate":   [0.15, 0.20],
}

# Lightweight search — fewer combos for speed
FAST_SEARCH_SPACE = {
    "lr":             [5e-4, 2e-4],
    "batch_size":     [32],
    "epochs_per_rnd": [5, 8],
    "hidden_dims":    [[512, 256, 128]],
    "dropout_rate":   [0.20],
}


def run_quick_trial(hp_dict: dict, n_rounds: int = 5) -> Tuple[float, float]:
    """
    Runs a short FL trial. Returns (global_acc, node_A_acc).
    Import engine here to avoid circular imports at module load time.
    """
    from fl.engine import FLoCBPoCL, HyperParams
    hp = HyperParams(
        lr             = hp_dict["lr"],
        batch_size     = hp_dict["batch_size"],
        epochs_per_rnd = hp_dict["epochs_per_rnd"],
        hidden_dims    = hp_dict["hidden_dims"],
        dropout_rate   = hp_dict["dropout_rate"],
    )
    engine = FLoCBPoCL(hp=hp, consensus_mode="pocl_pbft", verbose=False)
    result = engine.train(n_rounds=n_rounds)
    g_acc  = result["global_acc_final"]
    nA_acc = float(result["per_node_acc_final"].get("0", 0.0))
    return g_acc, nA_acc


def grid_search(fast: bool = True,
                n_trial_rounds: int = 5) -> Tuple[dict, dict]:
    """
    Returns (best_hp_dict, all_trial_results_dict).
    """
    space  = FAST_SEARCH_SPACE if fast else SEARCH_SPACE
    keys   = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"\n  [HyperSearch] {len(combos)} configurations | "
          f"{n_trial_rounds} rounds each")
    print(f"  {'─'*60}")

    best_score = -1.0
    best_hp    = None
    all_results: List[dict] = []

    for i, vals in enumerate(combos):
        hp_dict = dict(zip(keys, vals))
        print(f"  Trial {i+1:2d}/{len(combos)} | {hp_dict}", end=" -> ", flush=True)
        try:
            g_acc, nA_acc = run_quick_trial(hp_dict, n_trial_rounds)
            # Score: 40% global + 60% Node A (prioritise Node A target)
            score = 0.40 * g_acc + 0.60 * nA_acc
            print(f"G={g_acc:.4f} NodeA={nA_acc:.4f} score={score:.4f}")
            all_results.append({
                "hp": {k: (list(v) if isinstance(v, list) else v)
                       for k, v in hp_dict.items()},
                "global": round(g_acc, 6),
                "node_a": round(nA_acc, 6),
                "score":  round(score, 6),
            })
            if score > best_score:
                best_score = score
                best_hp    = hp_dict
        except Exception as e:
            print(f"ERROR: {e}")

    if best_hp is None:
        # Fallback to safe defaults
        best_hp = {
            "lr": 2e-4, "batch_size": 32, "epochs_per_rnd": 8,
            "hidden_dims": [512, 256, 128], "dropout_rate": 0.20,
        }

    print(f"\n  [HyperSearch] Best HP: {best_hp}")
    print(f"  [HyperSearch] Best score: {best_score:.4f}")
    return best_hp, {
        "trials":     all_results,
        "best":       {k: (list(v) if isinstance(v, list) else v)
                       for k, v in best_hp.items()},
        "best_score": round(best_score, 6),
    }
