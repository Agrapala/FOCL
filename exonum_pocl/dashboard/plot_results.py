"""
Dashboard — Plot All Results
==============================
Generates publication-quality charts from experiment results.

Charts:
  1. Global accuracy curves: pBFT vs PoCL-pBFT vs PoS vs Tuned
  2. Per-node accuracy progression (4 nodes, tuned run)
  3. Consensus comparison bar chart (accuracy / delay / gas)
  4. FL improvement vs baseline (per node)
  5. Trust score evolution per node (PoCL-pBFT)
  6. IPFS gas savings visualisation
"""

import json
import os
import sys

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(ROOT, "results")
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)


def load(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"  [Dashboard] WARNING: {name}.json not found, skipping")
        return {}
    with open(path) as f:
        return json.load(f)


def savefig(fig, name: str):
    path = os.path.join(DASHBOARD_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  [Dashboard] {name}")


def plot_all():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [Dashboard] matplotlib not found — skipping charts")
        print("  Install: pip install matplotlib")
        return

    plt.rcParams.update({
        "figure.dpi":     150,
        "font.size":      11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
    })

    COLORS = {
        "pBFT":            "#2196F3",
        "PoCL-pBFT":       "#4CAF50",
        "PoS":             "#FF9800",
        "PoCL-pBFT-Tuned": "#9C27B0",
    }
    NODE_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
    NODE_NAMES  = ["Node A", "Node B", "Node C", "Node D"]

    pbft   = load("exp1_pbft")
    pocl   = load("exp2_pocl_pbft")
    pos    = load("exp3_pos")
    tuned  = load("exp5_best_hp_pocl_pbft")
    master = load("master_results")

    # ── Fig 1: Global accuracy curves ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res, col in [
        ("pBFT",             pbft,  COLORS["pBFT"]),
        ("PoCL-pBFT",        pocl,  COLORS["PoCL-pBFT"]),
        ("PoS",              pos,   COLORS["PoS"]),
        ("PoCL-pBFT (Tuned)",tuned, COLORS["PoCL-pBFT-Tuned"]),
    ]:
        if not res:
            continue
        log = res.get("global_acc_log", [])
        if log:
            ax.plot(range(len(log)), [v*100 for v in log], label=name,
                    color=col, marker="o", markersize=3, markevery=5)
    ax.axhline(89, color="red", linestyle="--", linewidth=1.2,
               label="Target 89% (Node A)")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Global Accuracy — Exonum Blockchain: pBFT / PoCL-pBFT / PoS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig1_global_accuracy.png")
    plt.close(fig)

    # ── Fig 2: Per-node accuracy (tuned run) ───────────────────────────────
    if tuned:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axes.flat):
            log  = tuned.get("per_node_acc_log", {}).get(str(i), [])
            base = float(tuned.get("baseline_accs", {}).get(str(i), 0))
            if log:
                ax.plot(range(len(log)), [v*100 for v in log],
                        color=NODE_COLORS[i], label=NODE_NAMES[i])
            ax.axhline(base*100, color="gray", linestyle=":",
                       linewidth=1.5, label=f"Baseline {base*100:.1f}%")
            if i == 0:
                ax.axhline(89, color="red", linestyle="--",
                           linewidth=1.2, label="Target 89%")
            ax.set_title(NODE_NAMES[i])
            ax.set_xlabel("FL Round")
            ax.set_ylabel("Accuracy (%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Per-Node Accuracy — PoCL-pBFT Tuned (Exonum BC)", fontsize=14)
        fig.tight_layout()
        savefig(fig, "fig2_per_node_accuracy.png")
        plt.close(fig)

    # ── Fig 3: Consensus comparison bar charts ─────────────────────────────
    if master:
        exp        = master.get("experiments", {})
        mechanisms = list(exp.keys())
        if mechanisms:
            accs   = [exp[m]["global_acc_final"]*100 for m in mechanisms]
            delays = [exp[m]["avg_consensus_delay_ms"] for m in mechanisms]
            gases  = [exp[m]["avg_gas_equivalent"] for m in mechanisms]

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for ax, vals, title, ylabel in [
                (axes[0], accs,   "FL Accuracy (%)",              "Accuracy (%)"),
                (axes[1], delays, "Avg Consensus Delay (ms)",     "Delay (ms)"),
                (axes[2], gases,  "Avg Gas Equivalent (tx count)","Gas Equiv."),
            ]:
                bars = ax.bar(mechanisms, vals,
                              color=[COLORS.get(m, "#888") for m in mechanisms])
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() * 1.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=9)
                ax.tick_params(axis="x", rotation=15)
                ax.grid(True, axis="y", alpha=0.3)

            fig.suptitle("Consensus Mechanism Comparison — Exonum Blockchain", fontsize=13)
            fig.tight_layout()
            savefig(fig, "fig3_consensus_comparison.png")
            plt.close(fig)

    # ── Fig 4: FL improvement over baseline ────────────────────────────────
    if tuned:
        imps = tuned.get("fl_improvement_pct", {})
        vals = [float(imps.get(str(i), 0)) for i in range(4)]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(NODE_NAMES, vals, color=NODE_COLORS)
        ax.axhline(5, color="red", linestyle="--", linewidth=1.2,
                   label="Target: +5pp improvement")
        ax.set_title("FL Improvement over Local Baseline — PoCL-pBFT Tuned")
        ax.set_ylabel("Accuracy Improvement (percentage points)")
        ax.legend()
        for bar, v in zip(bars, vals):
            sign = "+" if v >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.1 if v >= 0 else -0.3),
                    f"{sign}{v:.2f}pp", ha="center", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        savefig(fig, "fig4_fl_improvement.png")
        plt.close(fig)

    # ── Fig 5: Trust score evolution ───────────────────────────────────────
    if pocl:
        trust = pocl.get("trust_log", {})
        if trust:
            fig, ax = plt.subplots(figsize=(9, 4))
            for i in range(4):
                log = trust.get(str(i), [])
                if log:
                    ax.plot(range(len(log)), log, label=NODE_NAMES[i],
                            color=NODE_COLORS[i], linewidth=2)
            ax.set_title("Trust Score Evolution per Node — PoCL-pBFT")
            ax.set_xlabel("FL Round")
            ax.set_ylabel("Trust Weight")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            savefig(fig, "fig5_trust_scores.png")
            plt.close(fig)

    # ── Fig 6: IPFS gas savings ────────────────────────────────────────────
    ipfs_data = load("exp6_ipfs_analysis")
    if ipfs_data:
        gs = ipfs_data.get("gas_savings", {})
        if gs:
            labels = ["On-Chain\n(full model)", "IPFS\n(CID only)"]
            values = [gs.get("onchain_gas_total", 0),
                      gs.get("ipfs_gas_total", 0)]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(labels, values, color=["#E53935", "#43A047"], width=0.4)
            ax.set_title(
                f"Gas Cost: On-Chain vs IPFS Off-Chain\n"
                f"Savings: {gs.get('savings_pct','?')}%  |  "
                f"Model: {gs.get('model_size_bytes','?')} bytes vs "
                f"{gs.get('cid_size_bytes','?')} bytes CID"
            )
            ax.set_ylabel("Total Gas Units")
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{v:,.0f}", ha="center", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            savefig(fig, "fig6_ipfs_gas_savings.png")
            plt.close(fig)

    print(f"\n  [Dashboard] All charts saved to {DASHBOARD_DIR}")


if __name__ == "__main__":
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    plot_all()
