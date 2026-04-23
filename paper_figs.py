"""Generate all figures for the paper."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import os, base64

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/tables/train_and_test_results.csv", parse_dates=["date"])
df = df[df["date"] >= "2012-01-01"].copy()
ev = df[df["bdfr"] == 1].copy()
ev["label_ret"] = np.exp(ev["label"]) - 1
ev["ym"] = ev["date"].dt.to_period("M")
top_n = 30

results = []
for ym, grp in ev.groupby("ym"):
    grp = grp.sort_values("pred", ascending=False)
    top = grp.head(top_n)
    results.append({
        "ym": pd.Timestamp(str(ym)),
        "top": top["label_ret"].mean(),
        "bench": grp["label_ret"].mean(),
        "n": len(grp),
        "ic": spearmanr(grp["pred"], grp["label_ret"])[0] if len(grp) > 5 else np.nan,
    })

res = pd.DataFrame(results)
cum_top   = (1 + res["top"]).cumprod()
cum_bench = (1 + res["bench"]).cumprod()

# ── Figure 1: Cumulative wealth + rolling top-30 excess return ───────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(res["ym"], cum_top,   color="#1a5c9c", lw=1.8, label="Top-30 Strategy")
axes[0].plot(res["ym"], cum_bench, color="#888888", lw=1.3, ls="--", label="Universe Benchmark")
axes[0].set_yscale("log")
axes[0].set_ylabel("Cumulative Wealth (log scale, $1 start)", fontsize=10)
axes[0].set_title("Figure 1. Cumulative Wealth: Strategy vs. Benchmark (2012–2023, test phase)",
                  fontsize=11, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].fill_between(res["ym"], cum_top, cum_bench,
                     where=cum_top >= cum_bench, alpha=0.09, color="green")
axes[0].fill_between(res["ym"], cum_top, cum_bench,
                     where=cum_top < cum_bench,  alpha=0.09, color="red")

# Bottom panel: rolling 12-period win rate of top-30 vs bench
ROLL = 12
rolling_win = pd.Series(
    [(res["top"].iloc[i:i+ROLL] > res["bench"].iloc[i:i+ROLL]).mean() * 100
     for i in range(len(res) - ROLL + 1)],
    index=res["ym"].iloc[ROLL-1:].values
)
rolling_excess = (res["top"] - res["bench"]).rolling(ROLL, min_periods=6).mean() * 100
rolling_excess.index = res["ym"]

ax2 = axes[1]
bar_colors = ["#3BB273" if v > 0 else "#E05252" for v in (res["top"] - res["bench"])]
ax2.bar(res["ym"], (res["top"] - res["bench"]) * 100,
        color=bar_colors, alpha=0.35, width=25)
ax2.plot(res["ym"], rolling_excess, color="#1a5c9c", lw=1.8,
         label=f"{ROLL}-period rolling mean excess return")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Top-30 excess return over benchmark (%)", fontsize=10)
ax2.set_title("Figure 1 (cont.). Monthly Excess Return of Top-30 Strategy over Benchmark",
              fontsize=10, style="italic")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/paper_fig1_wealth.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")

# ── Figure 2: Prediction decile bar chart ────────────────────────────────────
N_BUCKETS = 10
bucket_means = []
for ym, grp in ev.groupby("ym"):
    if len(grp) < N_BUCKETS * 2:
        continue
    grp = grp.copy()
    try:
        grp["bucket"] = pd.qcut(grp["pred"], N_BUCKETS, labels=False, duplicates="drop")
    except Exception:
        continue
    bucket_means.append(grp.groupby("bucket")["label_ret"].mean())

bucket_df = pd.concat(bucket_means, axis=1).T.mean().sort_index()
xs = bucket_df.index.tolist()
ys = bucket_df.values * 100

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#3BB273" if x == xs[-1] else ("#E05252" if x == xs[0] else "#9DB4CC") for x in xs]
ax.bar(range(len(xs)), ys, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Prediction Rank Decile (1 = lowest, 10 = highest)", fontsize=10)
ax.set_ylabel("Mean 60-day Forward Return (%)", fontsize=10)
ax.set_title("Figure 2. Mean 60-day Return by Prediction Decile (2012–2023, averaged across all periods)",
             fontsize=10, fontweight="bold")
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([str(int(x)+1) for x in xs])
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(ys):
    ax.text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
plt.savefig("results/figures/paper_fig2_buckets.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ── Figure 3: TSCV design illustration ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(-0.4, 3.8)
ax.axis("off")

colors_block = {"val": "#F5A623", "train": "#4A90D9", "gap": "#e8e8e8", "test": "#27AE60"}
labels_block = {"val": "Validation", "train": "Training", "gap": "Buffer\n(61 d)", "test": "Test"}

def draw_row(y, val_x, val_w, train_x, train_w, gap_x, gap_w, test_x, test_w, label=""):
    for key, x, w in [("val", val_x, val_w), ("train", train_x, train_w),
                      ("gap", gap_x, gap_w), ("test", test_x, test_w)]:
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.32), w, 0.64,
            boxstyle="round,pad=0.02",
            facecolor=colors_block[key],
            edgecolor="white", linewidth=1.5, alpha=0.88
        )
        ax.add_patch(rect)
        if w > 0.25:
            ax.text(x + w/2, y, labels_block[key],
                    ha="center", va="center", fontsize=8.5,
                    color="white" if key != "gap" else "#666",
                    fontweight="bold" if key != "gap" else "normal")
    if label:
        ax.text(-0.08, y, label, ha="right", va="center", fontsize=8.5, color="#333")

# Time axis arrow
ax.annotate("", xy=(10.1, -0.1), xytext=(0, -0.1),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
ax.text(5, -0.27, "Time", ha="center", va="top", fontsize=9, color="#444")

# Three example windows
draw_row(3.2, 0.2, 1.6, 2.1, 2.8, 5.1, 0.4, 5.7, 1.5, label="Window t₁")
draw_row(2.1, 0.2, 1.6, 2.5, 2.8, 5.5, 0.4, 6.1, 1.5, label="Window t₂")
draw_row(1.0, 0.2, 1.6, 2.9, 2.8, 5.9, 0.4, 6.5, 1.5, label="Window t₃")

# "..." dots
ax.text(5.0, 0.45, "· · ·", ha="center", va="center", fontsize=14, color="#aaa")

# Arrows above showing temporal ordering within one window
ax.annotate("", xy=(2.0, 3.65), xytext=(0.15, 3.65),
            arrowprops=dict(arrowstyle="->", color=colors_block["val"], lw=1.2))
ax.annotate("", xy=(5.0, 3.65), xytext=(2.15, 3.65),
            arrowprops=dict(arrowstyle="->", color=colors_block["train"], lw=1.2))
ax.annotate("", xy=(7.3, 3.65), xytext=(5.1, 3.65),
            arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.0))
ax.text(1.1,  3.75, "1. Fit early stopping", fontsize=7.5, color=colors_block["val"],  ha="center")
ax.text(3.6,  3.75, "2. Train model", fontsize=7.5, color=colors_block["train"], ha="center")
ax.text(6.2,  3.75, "3. Evaluate", fontsize=7.5, color=colors_block["test"],  ha="center")

# Legend
leg_items = [mpatches.Patch(facecolor=colors_block[k], label=labels_block[k], alpha=0.88)
             for k in ["val", "train", "gap", "test"]]
ax.legend(handles=leg_items, loc="lower right", fontsize=8.5,
          framealpha=0.9, ncol=4, bbox_to_anchor=(1.0, 0.02))

ax.set_title("Figure 3. Three-Block Walk-Forward TSCV Design\n"
             "The validation block precedes the training window, decoupling tree-count selection from the test period.",
             fontsize=10, fontweight="bold", pad=6)

plt.tight_layout()
plt.savefig("results/figures/paper_fig3_tscv.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")

# ── Encode all 3 figures ──────────────────────────────────────────────────────
imgs = {}
for k, path in [("fig1", "results/figures/paper_fig1_wealth.png"),
                ("fig2", "results/figures/paper_fig2_buckets.png"),
                ("fig3", "results/figures/paper_fig3_tscv.png")]:
    with open(path, "rb") as f:
        imgs[k] = base64.b64encode(f.read()).decode()
    print(f"{k}: {len(imgs[k])} chars")

with open("_imgs.py", "w") as f:
    for k, v in imgs.items():
        f.write(f"{k} = '{v}'\n")
print("done")
