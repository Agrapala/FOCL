"""
FLoBC — Generate All Charts from Experiment Results
=====================================================
Reads results/all_results.json and saves 6 PNG charts to dashboard/

Run:
    python dashboard/plot_results.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "all_results.json")
OUT = os.path.dirname(os.path.abspath(__file__))


def load():
    if not os.path.exists(RESULTS):
        print("ERROR: results/all_results.json not found.")
        print("Run:   python experiments/run_experiments.py   first.")
        sys.exit(1)
    with open(RESULTS) as f:
        return json.load(f)


def plot_all(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    C = ["#00d4b4", "#4a9eff", "#f59e0b", "#f43f5e",
         "#a855f7", "#84cc16", "#fb923c", "#22d3ee"]
    STYLE = dict(linewidth=2.2)

    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor":   "#0f172a",
        "axes.edgecolor":   "#1e3a5f",
        "axes.labelcolor":  "#94a3b8",
        "xtick.color":      "#64748b",
        "ytick.color":      "#64748b",
        "text.color":       "#e2e8f0",
        "grid.color":       "#1e3a5f",
        "legend.facecolor": "#0a1628",
        "legend.edgecolor": "#1e3a5f",
    })

    # ── 1. Exp 0 — Centralized vs Decentralized ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    cent = res["exp0"]["centralized"]
    dec  = res["exp0"]["decentralized"]
    ax.plot(cent, label="Centralized",   color=C[1], **STYLE)
    ax.plot(dec,  label="Decentralized (FLoBC)", color=C[0],
            **STYLE, linestyle="--")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Exp 0 — Centralized vs Decentralized\n"
                 f"Gap = {res['exp0']['gap_pct']:.2f}%  |  "
                 f"Blockchain: {res['exp0'].get('chain_length','?')} blocks",
                 pad=10)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out0 = os.path.join(OUT, "exp0_cent_vs_dec.png")
    fig.savefig(out0, dpi=150); plt.close(fig)
    print(f"  Saved: {out0}")

    # ── 2. Exp 1 — T/V Ratio bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    keys  = list(res["exp1"].keys())
    accs  = [res["exp1"][k]["max_accuracy"] for k in keys]
    cols  = [C[0] if k == "t7_v3" else C[1] for k in keys]
    ax.bar(range(len(keys)), accs, color=cols)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "/") for k in keys],
                       rotation=30, ha="right")
    ax.set_ylabel("Max Accuracy"); ax.set_ylim(0.96, 1.005)
    ax.set_title("Exp 1 — Trainer-to-Validator Ratio\n"
                 "Teal bar = paper's best: 7T/3V")
    ax.grid(axis="y", alpha=0.3)
    # Chain lengths as text on bars
    for i, k in enumerate(keys):
        cl = res["exp1"][k].get("chain_length", "")
        ax.text(i, accs[i] + 0.001, f"⛓{cl}", ha="center",
                fontsize=7, color="#64748b")
    fig.tight_layout()
    out1 = os.path.join(OUT, "exp1_tv_ratio.png")
    fig.savefig(out1, dpi=150); plt.close(fig)
    print(f"  Saved: {out1}")

    # ── 3. Exp 2 — Scoring vs Control ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(res["exp2"]["scoring_group"], label="Scoring Group",
            color=C[0], **STYLE)
    ax.plot(res["exp2"]["control_group"], label="Control Group",
            color=C[3], **STYLE, linestyle="--")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
    ax.set_title("Exp 2 — Reward-Penalty: Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    tlog = res["exp2"]["trust_log"]
    for i, (tid, vals) in enumerate(tlog.items()):
        if vals:
            ax.plot(vals, label=f"Trainer {tid}", color=C[i % len(C)],
                    linewidth=1.8)
    ax.set_xlabel("Round"); ax.set_ylabel("Trust Score")
    ax.set_title("Exp 2 — Trust Score Evolution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    out2 = os.path.join(OUT, "exp2_reward_penalty.png")
    fig.savefig(out2, dpi=150); plt.close(fig)
    print(f"  Saved: {out2}")

    # ── 4. Exp 3 — Sync Schemes ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scheme_colors = {"BSP": C[1], "SSP": C[0],
                     "BAP_1.0": C[2], "BAP_0.6": C[3]}
    ax = axes[0]
    for name, col in scheme_colors.items():
        if name in res["exp3"]:
            ax.plot(res["exp3"][name]["accuracy_log"], label=name,
                    color=col, **STYLE)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Accuracy")
    ax.set_title("Exp 3 — Sync Schemes: Accuracy vs Iterations")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    names = [n for n in scheme_colors if n in res["exp3"]]
    times = [res["exp3"][n]["avg_round_time"] for n in names]
    bars  = ax.bar(names, times, color=[scheme_colors[n] for n in names])
    ax.set_ylabel("Avg Round Time (s)")
    ax.set_title("Exp 3 — Sync Schemes: Speed")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f"{t:.4f}s", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out3 = os.path.join(OUT, "exp3_sync_schemes.png")
    fig.savefig(out3, dpi=150); plt.close(fig)
    print(f"  Saved: {out3}")

    # ── 5. Exp 7 — vs Dis-PFL ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    e7      = {k: v for k, v in res["exp7"].items()
               if k not in ("chain_length", "chain_valid")}
    methods = list(e7.keys())
    accs    = [e7[m] for m in methods]
    colors  = [C[0] if "FLoBC" in m else C[1] for m in methods]
    bars    = ax.barh(range(len(methods)), accs, color=colors)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(40, 96)
    ax.set_title("Exp 7 — FLoBC vs Dis-PFL\nTeal = FLoBC variants")
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                f"{acc:.2f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out7 = os.path.join(OUT, "exp7_vs_dispfl.png")
    fig.savefig(out7, dpi=150); plt.close(fig)
    print(f"  Saved: {out7}")

    # ── 6. Exp 8 — vs PVD-FL ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    ep_labels = ["1 epoch", "5 epochs", "10 epochs"]
    archs     = ["3x128FC", "3x512FC", "CNN"]
    x = np.arange(len(ep_labels)); w = 0.28

    for i, (arch, ax) in enumerate(zip(archs, axes)):
        pv = [res["exp8"]["pvdfl_paper"][arch][e] for e in ["1ep","5ep","10ep"]]
        fp = [res["exp8"]["flobc_paper"][arch][e] for e in ["1ep","5ep","10ep"]]
        fs = [res["exp8"]["flobc_simulated"][arch][e] for e in ["1ep","5ep","10ep"]]
        ax.bar(x - w,   pv, w, label="PVD-FL",      color=C[3])
        ax.bar(x,       fp, w, label="FLoBC(paper)", color=C[1])
        ax.bar(x + w,   fs, w, label="FLoBC(sim)",   color=C[0])
        ax.set_title(arch)
        ax.set_xticks(x)
        ax.set_xticklabels(ep_labels, rotation=15)
        ax.set_ylim(88, 100)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=7)

    fig.suptitle("Exp 8 — FLoBC vs PVD-FL", fontsize=13)
    fig.tight_layout()
    out8 = os.path.join(OUT, "exp8_vs_pvdfl.png")
    fig.savefig(out8, dpi=150); plt.close(fig)
    print(f"  Saved: {out8}")

    print(f"\n  All 6 charts saved to: {OUT}")


if __name__ == "__main__":
    plot_all(load())
