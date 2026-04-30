"""
Regenerate all paper figures and compute exact metrics using:
  - Dividend-adjusted returns (closeadj/closeunadj from prices.csv)
  - S&P 500 Total Return as the benchmark (^GSPC via yfinance)
Outputs:
  - results/figures/paper_fig1_wealth.png  (updated)
  - results/figures/paper_fig2_buckets.png (updated)
  - _imgs.py                               (updated)
  - prints exact table metrics for paper_gen.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import os, base64, sys
import yfinance as yf

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("results/figures", exist_ok=True)

# ── 1. Load predictions ───────────────────────────────────────────────────────
df = pd.read_csv("results/tables/train_and_test_results.csv", parse_dates=["date"])
df = df[df["date"] >= "2012-01-01"].copy()
ev = df[df["bdfr"] == 1].copy()
ev["label_ret_unadj"] = np.exp(ev["label"]) - 1
ev["ym"] = ev["date"].dt.to_period("M")
test_tickers = set(ev["ticker"].unique())
print(f"Test tickers: {len(test_tickers):,}  |  Test rows (bdfr=1): {len(ev):,}")

# ── 2. Load and join adj factors ──────────────────────────────────────────────
print("Loading prices for adj factors...")
prices = pd.read_csv(
    "data/raw/prices.csv",
    usecols=["ticker", "date", "closeadj", "closeunadj"],
    parse_dates=["date"],
)
prices = prices[prices["ticker"].isin(test_tickers) & (prices["closeunadj"] > 0)].copy()
prices["adj_factor"] = (prices["closeadj"] / prices["closeunadj"]).clip(0.01, 100)
prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
prices["adj_factor_future"] = prices.groupby("ticker")["adj_factor"].shift(-60)

af = prices.dropna(subset=["adj_factor_future"])[
    ["ticker", "date", "adj_factor", "adj_factor_future"]
]
ev = ev.merge(af, on=["ticker", "date"], how="left")
log_adj = np.log((ev["adj_factor_future"] / ev["adj_factor"]).clip(0.001, 1000))
ev["label_ret"] = np.exp(ev["label"] + log_adj) - 1  # dividend-adjusted return
valid_frac = ev["label_ret"].notna().mean()
print(f"Adj return coverage: {valid_frac*100:.1f}%")

# ── 3. Load S&P 500 (TR, auto-adjusted) ──────────────────────────────────────
print("Downloading S&P 500...")
sp_raw = yf.download("^GSPC", start="2012-01-01", end="2023-10-01",
                     auto_adjust=True, progress=False)
sp = sp_raw[("Close", "^GSPC")].copy()
sp.index = pd.to_datetime(sp.index).tz_localize(None)
sp_dates = sp.index.values
sp_vals  = sp.values
print(f"S&P 500: {sp.index[0].date()} to {sp.index[-1].date()}")

# ── 4. Build per-period portfolio results ─────────────────────────────────────
top_n = 30
rows = []
for ym, grp in ev.groupby("ym"):
    gv = grp.dropna(subset=["label_ret"]).sort_values("pred", ascending=False)
    if len(gv) == 0:
        continue
    d   = pd.Timestamp(str(ym))
    idx = np.searchsorted(sp_dates, np.datetime64(d))
    sp_ret = float(sp_vals[idx + 60] / sp_vals[idx] - 1) if idx + 60 < len(sp_vals) else np.nan
    ic = float(spearmanr(gv["pred"], gv["label_ret"])[0]) if len(gv) > 5 else np.nan
    rows.append({
        "ym":    d,
        "top":   float(gv.head(top_n)["label_ret"].mean()),
        "bench": sp_ret,
        "n":     len(gv),
        "ic":    ic,
    })

res = pd.DataFrame(rows).sort_values("ym").reset_index(drop=True)
res_v = res.dropna(subset=["bench"])   # periods where both exist
n     = len(res_v)
ppy   = 252 / 60                        # holding periods per year (≈4.2)

# ── 5. Compute all exact metrics ──────────────────────────────────────────────
def ann(col):
    s = res_v[col].dropna()
    return (1 + s).prod() ** (ppy / len(s)) - 1

def sh(col):
    return res_v[col].mean() / res_v[col].std() * np.sqrt(ppy)

ann_top   = ann("top")
ann_bench = ann("bench")
mean_top   = res_v["top"].mean()
mean_bench = res_v["bench"].mean()
std_top    = res_v["top"].std()
std_bench  = res_v["bench"].std()
sharpe_top   = sh("top")
sharpe_bench = sh("bench")
worst_top    = res_v["top"].min()
best_top     = res_v["top"].max()
worst_bench  = res_v["bench"].min()
best_bench   = res_v["bench"].max()
prob_single  = (res_v["top"] > res_v["bench"]).mean()
mean_ic      = res_v["ic"].mean()

# Rolling 12-period probability
ROLL = 12
roll_wins = []
for i in range(len(res_v) - ROLL + 1):
    s  = res_v["top"].iloc[i:i+ROLL]
    b  = res_v["bench"].iloc[i:i+ROLL]
    roll_wins.append((1 + s).prod() > (1 + b).prod())
prob_roll12 = np.mean(roll_wins)

print("\n" + "="*60)
print("EXACT METRICS FOR PAPER TABLE")
print("="*60)
print(f"Evaluation period: Jan 2012 – Sep 2023  ({n} periods)")
print(f"Annualized return  — Strategy:  {ann_top*100:.1f}%")
print(f"Annualized return  — S&P 500:   {ann_bench*100:.1f}%")
print(f"Mean 60d return    — Strategy:  {mean_top*100:.1f}%")
print(f"Mean 60d return    — S&P 500:   {mean_bench*100:.1f}%")
print(f"Std dev            — Strategy:  {std_top*100:.1f}%")
print(f"Std dev            — S&P 500:   {std_bench*100:.1f}%")
print(f"Sharpe             — Strategy:  {sharpe_top:.2f}")
print(f"Sharpe             — S&P 500:   {sharpe_bench:.2f}")
print(f"Prob beat S&P (single period):  {prob_single*100:.1f}%")
print(f"Prob beat S&P (rolling 12):     {prob_roll12*100:.1f}%")
print(f"Mean IC:                        {mean_ic:.3f}")
print(f"Worst period       — Strategy:  {worst_top*100:.1f}%")
print(f"Best period        — Strategy:  {best_top*100:.1f}%")
print(f"Worst period       — S&P 500:   {worst_bench*100:.1f}%")
print(f"Best period        — S&P 500:   {best_bench*100:.1f}%")

# ── 6. Figure 1: Cumulative wealth + rolling excess ───────────────────────────
cum_top   = (1 + res_v["top"]).cumprod()
cum_bench = (1 + res_v["bench"]).cumprod()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

ax0 = axes[0]
ax0.plot(res_v["ym"], cum_top,   color="#1a5c9c", lw=1.8, label="Top-30 Strategy (div. adj.)")
ax0.plot(res_v["ym"], cum_bench, color="#888888", lw=1.3, ls="--", label="S&P 500 (Total Return)")
ax0.set_yscale("log")
ax0.set_ylabel("Cumulative Wealth (log scale, $1 start)", fontsize=10)
ax0.set_title("Figure 1. Cumulative Wealth: Strategy vs. S&P 500 (2012–2023, test phase)",
              fontsize=11, fontweight="bold")
ax0.legend(fontsize=10)
ax0.grid(alpha=0.3)
ax0.fill_between(res_v["ym"], cum_top, cum_bench,
                 where=cum_top >= cum_bench, alpha=0.09, color="green")
ax0.fill_between(res_v["ym"], cum_top, cum_bench,
                 where=cum_top <  cum_bench, alpha=0.09, color="red")

ax1 = axes[1]
excess = (res_v["top"] - res_v["bench"]) * 100
roll_exc = excess.rolling(ROLL, min_periods=6).mean()
bar_colors = ["#3BB273" if v > 0 else "#E05252" for v in excess]
ax1.bar(res_v["ym"].values, excess.values, color=bar_colors, alpha=0.6,
        width=pd.Timedelta(days=25))
ax1.plot(res_v["ym"].values, roll_exc.values, color="#1a5c9c", lw=1.8,
         label=f"{ROLL}-period rolling mean excess return")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_ylabel("Top-30 excess return over S&P 500 (%)", fontsize=10)
ax1.set_title("Figure 1 (cont.). Monthly Excess Return of Top-30 Strategy over S&P 500",
              fontsize=10, style="italic")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/paper_fig1_wealth.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig1")

# ── 7. Figure 2: Return by prediction decile (adj returns) ───────────────────
N_BUCKETS = 10
bucket_means = []
for ym, grp in ev.groupby("ym"):
    gv = grp.dropna(subset=["label_ret"])
    if len(gv) < N_BUCKETS * 2:
        continue
    gv = gv.copy()
    try:
        gv["bucket"] = pd.qcut(gv["pred"], N_BUCKETS, labels=False, duplicates="drop")
    except Exception:
        continue
    bucket_means.append(gv.groupby("bucket")["label_ret"].mean())

bucket_df = pd.concat(bucket_means, axis=1).T.mean().sort_index()
xs = bucket_df.index.tolist()
ys = bucket_df.values * 100

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#3BB273" if x == xs[-1] else ("#E05252" if x == xs[0] else "#9DB4CC") for x in xs]
ax.bar(range(len(xs)), ys, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Prediction Rank Decile (1 = lowest, 10 = highest)", fontsize=10)
ax.set_ylabel("Mean 60-day Forward Return, div. adj. (%)", fontsize=10)
ax.set_title(
    "Figure 2. Mean 60-day Return by Prediction Decile (2012–2023, averaged across all periods)",
    fontsize=10, fontweight="bold")
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([str(int(x) + 1) for x in xs])
ax.grid(axis="y", alpha=0.3)
for i, v in enumerate(ys):
    ax.text(i, v + 0.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
plt.tight_layout()
plt.savefig("results/figures/paper_fig2_buckets.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ── 8. Figure 3: TSCV diagram (unchanged — copy existing if present) ──────────
fig3_path = "results/figures/paper_fig3_tscv.png"
if not os.path.exists(fig3_path):
    # Regenerate TSCV diagram (from original paper_figs.py)
    colors_block = {"val": "#F5A623", "train": "#4A90D9", "gap": "#e8e8e8", "test": "#27AE60"}
    labels_block = {"val": "Validation", "train": "Training", "gap": "Buffer\n(61 d)", "test": "Test"}

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 3.8); ax.axis("off")

    def draw_row(y, val_x, val_w, train_x, train_w, gap_x, gap_w, test_x, test_w, label=""):
        for key, x, w in [("val", val_x, val_w), ("train", train_x, train_w),
                          ("gap", gap_x, gap_w), ("test", test_x, test_w)]:
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.32), w, 0.64, boxstyle="round,pad=0.02",
                facecolor=colors_block[key], edgecolor="white", linewidth=1.5, alpha=0.88)
            ax.add_patch(rect)
            if w > 0.25:
                ax.text(x + w/2, y, labels_block[key], ha="center", va="center", fontsize=8.5,
                        color="white" if key != "gap" else "#666",
                        fontweight="bold" if key != "gap" else "normal")
        if label:
            ax.text(-0.08, y, label, ha="right", va="center", fontsize=8.5, color="#333")

    ax.annotate("", xy=(10.1, -0.1), xytext=(0, -0.1),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
    ax.text(5, -0.27, "Time", ha="center", va="top", fontsize=9, color="#444")
    draw_row(3.2, 0.2, 1.6, 2.1, 2.8, 5.1, 0.4, 5.7, 1.5, label="Window t₁")
    draw_row(2.1, 0.2, 1.6, 2.5, 2.8, 5.5, 0.4, 6.1, 1.5, label="Window t₂")
    draw_row(1.0, 0.2, 1.6, 2.9, 2.8, 5.9, 0.4, 6.5, 1.5, label="Window t₃")
    ax.text(5.0, 0.45, "· · ·", ha="center", va="center", fontsize=14, color="#aaa")
    ax.annotate("", xy=(2.0, 3.65), xytext=(0.15, 3.65),
                arrowprops=dict(arrowstyle="->", color=colors_block["val"], lw=1.2))
    ax.annotate("", xy=(5.0, 3.65), xytext=(2.15, 3.65),
                arrowprops=dict(arrowstyle="->", color=colors_block["train"], lw=1.2))
    ax.annotate("", xy=(7.3, 3.65), xytext=(5.1, 3.65),
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.0))
    ax.text(1.1, 3.75, "1. Fit early stopping", fontsize=7.5, color=colors_block["val"], ha="center")
    ax.text(3.6, 3.75, "2. Train model",        fontsize=7.5, color=colors_block["train"], ha="center")
    ax.text(6.2, 3.75, "3. Evaluate",           fontsize=7.5, color=colors_block["test"], ha="center")
    leg_items = [mpatches.Patch(facecolor=colors_block[k], label=labels_block[k], alpha=0.88)
                 for k in ["val", "train", "gap", "test"]]
    ax.legend(handles=leg_items, loc="lower right", fontsize=8.5,
              framealpha=0.9, ncol=4, bbox_to_anchor=(1.0, 0.02))
    ax.set_title(
        "Figure 3. Three-Block Walk-Forward TSCV Design\n"
        "The validation block precedes the training window, decoupling tree-count selection from the test period.",
        fontsize=10, fontweight="bold", pad=6)
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3 (regenerated)")
else:
    print("fig3 already exists — keeping")

# ── 9. Update _imgs.py ────────────────────────────────────────────────────────
imgs = {}
for k, path in [("fig1", "results/figures/paper_fig1_wealth.png"),
                ("fig2", "results/figures/paper_fig2_buckets.png"),
                ("fig3", fig3_path)]:
    with open(path, "rb") as f:
        imgs[k] = base64.b64encode(f.read()).decode()
    print(f"{k}: {len(imgs[k]):,} chars")

with open("_imgs.py", "w") as f:
    for k, v in imgs.items():
        f.write(f"{k} = '{v}'\n")
print("_imgs.py updated")
