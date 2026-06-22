"""
plot_pneumonia.py  —  All charts for Pneumonia BC-FL experiments
=================================================================
Run AFTER run_pneumonia.py AND evaluate_objectives.py complete.

    python dashboard/plot_pneumonia.py

Saves 7 PNG charts to dashboard/:
  exp1_global_accuracy.png       global FL accuracy over 30 rounds
  exp1_local_accuracy.png        per-hospital local train accuracy per round
  exp1_trust_evolution.png       trust score evolution over rounds (all 4)
  exp2_byzantine_trust.png       final trust scores after Byzantine attack
  exp3_sync_compare.png          sync scheme accuracy + block count
  obj2_before_after.png          Objective 2: before vs after FL per hospital
  obj2_gain.png                  Objective 2: accuracy gain per hospital vs 5% target
"""

import os, sys, json
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dashboard")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy"); sys.exit(1)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         130,
})

HOSP_COLORS = {
    "Hospital_Galle":   "#2E75B6",
    "Hospital_Colombo": "#00B050",
    "Hospital_Kandy":   "#FF8C00",
    "Hospital_Jaffna":  "#C00000",
}
NID_LIST  = ["A", "B", "C", "D"]
BAR_COLS  = list(HOSP_COLORS.values())

# ── Load results ─────────────────────────────────────────────────────────────
RES = os.path.join(ROOT, "results", "pneumonia_results.json")
OBJ = os.path.join(ROOT, "results", "objective_verification.json")

if not os.path.exists(RES):
    print("Run python run_pneumonia.py first."); sys.exit(1)

with open(RES) as f: data = json.load(f)
obj_data = json.load(open(OBJ)) if os.path.exists(OBJ) else None

e1   = data["experiment_1_BSP"]
hosp = data.get("hospitals", {
    "A": "Hospital_Galle", "B": "Hospital_Colombo",
    "C": "Hospital_Kandy", "D": "Hospital_Jaffna"})

saved = []

# ── Chart 1: Global accuracy over rounds ─────────────────────────────────────
acc_log = e1["accuracy_log"]
rounds  = list(range(len(acc_log)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rounds, acc_log, color="#2E75B6", lw=2.5, marker="o", ms=4,
        label="Global model accuracy")
ax.fill_between(rounds, acc_log, alpha=0.08, color="#2E75B6")
ax.axhline(acc_log[-1], color="#C00000", lw=1.2, ls="--",
           label=f"Final: {acc_log[-1]:.4f}")
if len(acc_log) > 1:
    ax.annotate(f"Start: {acc_log[0]:.4f}",
                xy=(0, acc_log[0]), xytext=(2, acc_log[0]-0.03),
                fontsize=9, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
ax.set_title("BC-FL Global Model Accuracy — 4 Hospital Pneumonia\n"
             "(BSP sync, 30 rounds, warm-started from local best models)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("FL Round"); ax.set_ylabel("Test Accuracy")
ax.set_ylim(max(0, min(acc_log)-0.05), 1.05); ax.legend(fontsize=10)
fig.tight_layout()
p = os.path.join(OUT_DIR, "exp1_global_accuracy.png")
fig.savefig(p); plt.close(fig); saved.append(p)

# ── Chart 2: Per-hospital local train accuracy ────────────────────────────────
loc_tr = e1.get("local_train_log", {})
loc_vl = e1.get("local_val_log", {})
if loc_tr:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nid in NID_LIST:
        name  = hosp.get(nid, f"Hospital_{nid}")
        color = HOSP_COLORS.get(name, "#888")
        tr    = loc_tr.get(nid, [])
        vl    = loc_vl.get(nid, [])
        rnd   = list(range(1, len(tr)+1))
        lbl   = f"Node {nid} ({name.split('_')[1]})"
        if tr: ax1.plot(rnd, tr, color=color, lw=2, marker="o", ms=3, label=lbl)
        if vl: ax2.plot(rnd, vl, color=color, lw=2, marker="s", ms=3, label=lbl)
    for ax, ttl in [(ax1, "Local TRAIN Accuracy"), (ax2, "Local VAL Accuracy")]:
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel("FL Round"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
    fig.suptitle("Per-Hospital Local Training Metrics per FL Round",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "exp1_local_accuracy.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)

# ── Chart 3: Trust score EVOLUTION over rounds (Exp 1) ───────────────────────
trust_log = e1.get("trust_log", {})
if trust_log:
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, vs in sorted(trust_log.items()):
        idx   = int(k)
        nid   = NID_LIST[idx] if idx < len(NID_LIST) else str(k)
        name  = hosp.get(nid, f"Hospital_{nid}")
        color = HOSP_COLORS.get(name, "#888")
        rnd   = list(range(1, len(vs)+1))
        lbl   = f"Node {nid} ({name.split('_')[-1]})"
        ax.plot(rnd, vs, color=color, lw=2.2, marker="o", ms=3.5, label=lbl)
    ax.axhline(0.25, color="grey", lw=1, ls=":", alpha=0.6,
               label="Equal share (0.25)")
    ax.set_title("Trust Score Evolution — All 4 Hospitals (Exp 1, BSP)\n"
                 "Healthy hospitals converge to balanced trust scores",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("FL Round"); ax.set_ylabel("Trust Score")
    ax.set_ylim(0, 0.7); ax.legend(fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "exp1_trust_evolution.png")
    fig.savefig(p); plt.close(fig); saved.append(p)

# ── Chart 4: Byzantine trust scores (Exp 2) ──────────────────────────────────
e2     = data.get("experiment_2_byzantine", {})
trust2 = e2.get("final_trust", {})
if trust2:
    keys   = sorted(trust2.keys())
    labels = [k for k in keys]
    scores = [trust2[k] for k in keys]
    colors = []
    for k in keys:
        parts = k.split("_")
        nid   = parts[1] if len(parts) > 1 else "?"
        name  = hosp.get(nid, "")
        colors.append(HOSP_COLORS.get(name, "#888888"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white", height=0.5)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width()+0.006, bar.get_y()+bar.get_height()/2,
                f"{s:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, max(scores)*1.35 if scores else 1)
    ax.set_xlabel("Final Trust Score", fontsize=11)
    ax.set_title("Trust Scores After Byzantine Attack\n"
                 "Hospital Jaffna (Node D) injected noisy model updates",
                 fontsize=12, fontweight="bold")
    for i, k in enumerate(keys):
        if "Jaffna" in k and scores[i] < max(scores)*0.3:
            ax.annotate("← Penalised by reputation system",
                        xy=(scores[i], i), xytext=(scores[i]+0.02, i),
                        fontsize=9, color="#C00000",
                        arrowprops=dict(arrowstyle="->", color="#C00000"))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "exp2_byzantine_trust.png")
    fig.savefig(p); plt.close(fig); saved.append(p)

# ── Chart 5: Sync scheme comparison (Exp 3) ──────────────────────────────────
sync = data.get("experiment_3_sync", {})
if sync:
    slabels = list(sync.keys())
    accs    = [sync[k]["final_accuracy"] for k in slabels]
    lengths = [sync[k]["chain_length"]   for k in slabels]
    cols    = ["#2E75B6","#00B050","#FF8C00","#9B59B6"][:len(slabels)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    b1 = ax1.bar(slabels, accs, color=cols, edgecolor="white", width=0.5)
    for bar, a in zip(b1, accs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                 f"{a:.4f}", ha="center", fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Final Accuracy by Sync Scheme", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy"); ax1.set_xlabel("Scheme")

    b2 = ax2.bar(slabels, lengths, color=cols, edgecolor="white", width=0.5)
    for bar, l in zip(b2, lengths):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 str(l), ha="center", fontsize=10)
    ax2.set_title("Blockchain Blocks Created", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Blocks"); ax2.set_xlabel("Scheme")

    fig.suptitle("Synchronisation Scheme Comparison — Pneumonia BC-FL\n"
                 "4 Hospital Nodes | 20 Rounds",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "exp3_sync_compare.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)

# ── Charts 6 & 7: Objective 2 — before vs after FL ───────────────────────────
if obj_data:
    gains_raw = obj_data.get("objective_2", {}).get("per_hospital", {})
    if gains_raw:
        nids    = NID_LIST
        befores = [gains_raw.get(n, {}).get("before_fl", 0) for n in nids]
        afters  = [gains_raw.get(n, {}).get("after_fl",  0) for n in nids]
        gains_v = [gains_raw.get(n, {}).get("gain",      0) for n in nids]
        names   = [hosp.get(n, n) for n in nids]
        colors  = [HOSP_COLORS.get(nm, "#888") for nm in names]
        x       = np.arange(len(nids))
        short   = [nm.split("_")[1] for nm in names]

        # Chart 6: Before vs After
        fig, ax = plt.subplots(figsize=(10, 5.5))
        w = 0.35
        b1 = ax.bar(x - w/2, befores, w, label="Before FL (local only)",
                    color=[c+"88" for c in colors], edgecolor="white")
        b2 = ax.bar(x + w/2, afters,  w, label="After FL (global model)",
                    color=colors, edgecolor="white")
        for bar, v in zip(b1, befores):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=9)
        for bar, v in zip(b2, afters):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_xlabel("Hospital")
        ax.set_title("Objective 2: Per-Hospital Accuracy Before vs After Federated Learning\n"
                     "Each hospital's test set accuracy — no raw data shared",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10); fig.tight_layout()
        p = os.path.join(OUT_DIR, "obj2_before_after.png")
        fig.savefig(p); plt.close(fig); saved.append(p)

        # Chart 7: Gain vs 5% target
        fig, ax = plt.subplots(figsize=(9, 5))
        bar_cols2 = ["#00B050" if g >= 0.05 else "#C00000" for g in gains_v]
        bars = ax.bar(short, [g*100 for g in gains_v],
                      color=bar_cols2, edgecolor="white", width=0.5)
        for bar, g in zip(bars, gains_v):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (0.3 if g >= 0 else -1.5),
                    f"{g*100:+.2f}pp", ha="center", fontsize=10,
                    color="black")
        ax.axhline(5.0, color="#FF8C00", lw=2, ls="--",
                   label="Target: +5 percentage points")
        ax.set_ylabel("Accuracy Gain (percentage points)")
        ax.set_xlabel("Hospital")
        ax.set_title("Objective 2: Accuracy Gain per Hospital via Federation\n"
                     "Green = target met (≥5pp), Red = below target",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10); ax.set_ylim(-5, max(max([g*100 for g in gains_v])+5, 15))
        green_patch = mpatches.Patch(color="#00B050", label="≥ 5% gain (target met)")
        red_patch   = mpatches.Patch(color="#C00000", label="< 5% gain")
        ax.legend(handles=[green_patch, red_patch,
                            mpatches.Patch(color="#FF8C00", label="5pp target line")],
                  fontsize=9)
        fig.tight_layout()
        p = os.path.join(OUT_DIR, "obj2_gain.png")
        fig.savefig(p); plt.close(fig); saved.append(p)

# ── Chart 8: PoCL winner selection frequency ─────────────────────────────────
if obj_data:
    winner_log = obj_data.get("winner_log", [])
    reward_log = obj_data.get("reward_log", {})
    if winner_log:
        # Count how many times each hospital (tid 0-3) was selected as winner
        tid_map  = {0: "A", 1: "B", 2: "C", 3: "D"}
        win_cnt  = {0: 0, 1: 0, 2: 0, 3: 0}
        for round_winners in winner_log:
            for tid in round_winners:
                win_cnt[tid] = win_cnt.get(tid, 0) + 1
        n_rounds_pocl = len(winner_log)

        nids    = [tid_map[t] for t in sorted(win_cnt)]
        counts  = [win_cnt[t] for t in sorted(win_cnt)]
        pcts    = [c / n_rounds_pocl * 100 for c in counts]
        names   = [hosp.get(n, f"Hospital_{n}") for n in nids]
        colors  = [HOSP_COLORS.get(nm, "#888") for nm in names]
        short   = [nm.split("_")[1] for nm in names]

        # Mean reward per winner (contribution R_i)
        mean_rewards = []
        for t in sorted(win_cnt):
            vs = reward_log.get(str(t), [])
            mean_rewards.append(float(np.mean(vs)) if vs else 0.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Left: selection frequency
        bars = ax1.bar(short, pcts, color=colors, edgecolor="white", width=0.5)
        for bar, c, p in zip(bars, counts, pcts):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f"{c}/{n_rounds_pocl}\n({p:.1f}%)", ha="center", fontsize=9)
        ax1.axhline(75.0, color="grey", lw=1.2, ls="--",
                    label="75% line (k=3 of 4 = expected)")
        ax1.set_ylim(0, 115); ax1.legend(fontsize=9)
        ax1.set_ylabel("Selected as Winner (% of rounds)")
        ax1.set_xlabel("Hospital")
        ax1.set_title("PoCL Winner Selection Frequency\n"
                      "(top-3 of 4 per round, voted by accuracy + timeliness)",
                      fontsize=11, fontweight="bold")

        # Right: mean contribution reward R_i
        bars2 = ax2.bar(short, mean_rewards, color=colors, edgecolor="white", width=0.5)
        for bar, v in zip(bars2, mean_rewards):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0002,
                     f"{v:.4f}", ha="center", fontsize=9)
        ax2.set_ylabel("Mean Contribution Score R_i")
        ax2.set_xlabel("Hospital")
        ax2.set_title("PoCL Mean Reward (Contribution R_i)\n"
                      "Mean |local_layer – global_layer| averaged across layers",
                      fontsize=11, fontweight="bold")

        fig.suptitle("FLoBC-PoCL Consensus — Winner Statistics",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        p = os.path.join(OUT_DIR, "pocl_winner_stats.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)

print(f"\n  {'='*55}")
print("  All charts saved:")
for p in saved:
    print(f"    {os.path.basename(p)}")
print(f"  Location: {OUT_DIR}")
print(f"  {'='*55}\n")
