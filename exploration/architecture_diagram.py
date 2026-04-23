"""
Model Architecture Diagram
Fundamentals Based, Mid-Term Equity Return Prediction

Coordinate system: xlim=(0, 22), ylim=(0, 20), figure 22 × 20 in.

Vertical layout (bottom → top):
  1.80 – 11.46  Section 3 panel background
  3.72 – 4.27   flow arrows + notes
  4.55 –  5.27  step annotation boxes
  6.08 – 10.58  four period rectangles  (BY=6.08, BH=4.50)
 11.08           time axis
 11.80           Section 3 header
 12.15 – 16.13  Section 2 loop background  (H=3.98)
 16.44           Section 2 header
 16.80 – 18.50  pipeline boxes  (PY=16.80, PH=1.70)
 18.72           Section 1 header
 19.65           title
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette ───────────────────────────────────────────────────────────────────
C_DATA  = "#2E86AB"
C_VAL   = "#E8A838"
C_TRAIN = "#3BB273"
C_TEST  = "#7B2D8B"
C_BT    = "#C0392B"
C_LOOP  = "#EEF2F7"
C_ARROW = "#2C3E50"
C_TEXT  = "#1A1A2E"
C_MUTED = "#8E9BAB"
C_PANEL = "#FAFBFC"
C_GRID  = "#CFD8DC"


# ── drawing helpers ───────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, ec="white", lw=1.8,
         zorder=3, alpha=1.0, r=0.35):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, alpha=alpha, clip_on=False,
    ))


def t(ax, x, y, s, fs=9, fw="bold", c="white",
      ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight=fw, color=c, zorder=z, **kw)


def arr(ax, x1, y1, x2, y2, c=C_ARROW, lw=1.8,
        z=6, style="-|>", rad=None):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=c, lw=lw,
            connectionstyle=f"arc3,rad={rad or 0}",
        ),
        zorder=z,
    )


# ── figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 20), facecolor="white")
ax  = fig.add_axes([0.015, 0.015, 0.97, 0.97])
ax.set_xlim(0, 22)
ax.set_ylim(0, 20)
ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# TITLE + DATASET INFO
# ─────────────────────────────────────────────────────────────────────────────

t(ax, 11, 19.65,
  "Fundamentals Based, Mid-Term Equity Return Prediction",
  fs=14.5, c=C_TEXT)



# ═════════════════════════════════════════════════════════════════════════════
# ① DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

t(ax, 0.40, 18.72, "① DATA PIPELINE", fs=10, c=C_TEXT, ha="left")

PY, PH = 16.80, 1.70
GAP    = 0.46
PW     = (21.28 - 3 * GAP) / 4   # ≈ 4.97

PIPE = [
    ("Sharadar Raw Data",
     "17,000+ US-listed equities  ·  25 years\n"
     "no survivorship bias"),
    ("Data Preparation",
     "min. price history: 180 days\n"
     "fund. snapshot  ·  daily panel"),
    ("Feature Engineering",
     "filter: revenue > 0  &  net income > 0\n"
     "log transforms  ·  valuation ratios\n"
     "sector & market-wide earnings signals"),
    ("Modeling and Strategy Simulation",
     "filter: market cap  ≥  $100M\n"
     "walk-forward TSCV  →  predictions"),
]

for i, (title, sub) in enumerate(PIPE):
    x  = 0.36 + i * (PW + GAP)
    cx = x + PW / 2
    cy = PY + PH / 2
    rbox(ax, x, PY, PW, PH, C_DATA)
    t(ax, cx, cy + 0.38, title, fs=9.0)
    t(ax, cx, cy - 0.32, sub, fs=7.2, fw="normal", style="italic")
    if i < 3:
        arr(ax, x + PW, cy, x + PW + GAP, cy)


# ═════════════════════════════════════════════════════════════════════════════
# ② WALK-FORWARD TSCV LOOP
# ═════════════════════════════════════════════════════════════════════════════

t(ax, 0.40, 16.44, "② WALK-FORWARD TSCV LOOP", fs=10, c=C_TEXT, ha="left")

# loop background
rbox(ax, 0.25, 12.15, 21.50, 3.98,
     C_LOOP, ec="#B0BEC5", lw=1.5, zorder=1, r=0.5)

# ── sliding mini-windows ──────────────────────────────────────────────────────
def mini(ax, ox, oy, sc, alpha, tag):
    vw = 2.0*sc; tw = 2.6*sc; xw = 1.1*sc; g = 0.30*sc; bh = 0.66
    rbox(ax, ox,           oy, vw, bh, C_VAL,   zorder=3, alpha=alpha, r=0.18)
    rbox(ax, ox+vw+g,      oy, tw, bh, C_TRAIN, zorder=3, alpha=alpha, r=0.18)
    rbox(ax, ox+vw+g+tw+g, oy, xw, bh, C_TEST,  zorder=3, alpha=alpha, r=0.18)
    for gxc in [ox+vw+g/2, ox+vw+g+tw+g/2]:
        ax.text(gxc, oy+bh/2, "gap", ha="center", va="center",
                fontsize=5.5, color=C_MUTED, zorder=4)
    for cx_, lbl in [
        (ox+vw/2,            "VAL"),
        (ox+vw+g+tw/2,       "TRAIN"),
        (ox+vw+g+tw+g+xw/2, f"TEST\n{tag}"),
    ]:
        t(ax, cx_, oy+bh/2, lbl, fs=7.0*sc, z=4, alpha=alpha)
    return ox + vw + g + tw + g + xw

mini(ax, 0.70, 15.42, 0.60, 0.25, "t−2")
mini(ax, 1.72, 14.52, 0.76, 0.52, "t−1")
re = mini(ax, 3.00, 13.28, 1.00, 1.00, "  t  ")

# "slides forward" arrow + label
arr(ax, re + 0.45, 13.61, 14.65, 13.61, c=C_ARROW, lw=2.0)
t(ax, re + 0.45 + (14.65 - re - 0.45)/2, 13.96,
  "slides forward one month per iteration",
  fs=8.2, c=C_TEXT, fw="normal", style="italic")

# repeat arrow on left
arr(ax, 0.47, 15.72, 0.47, 13.61, c="#E74C3C", lw=2.5, rad=-0.5)
t(ax, 0.20, 14.66, "monthly\nrepeat", fs=7.0, c="#E74C3C", fw="bold", rotation=90)

# ── phase-split info box ──────────────────────────────────────────────────────
PS_X, PS_Y = 14.95, 12.32
PS_W, PS_H = 6.70, 3.60

rbox(ax, PS_X, PS_Y, PS_W, PS_H,
     "white", ec="#B0BEC5", lw=1.3, zorder=2, r=0.45)

t(ax, PS_X + PS_W/2, PS_Y + PS_H - 0.40,
  "Phase Split  (cutoff: Jan 2012)", fs=9.0, c=C_TEXT)

# horizontal divider
dv_y = PS_Y + PS_H / 2 + 0.06
ax.plot([PS_X + 0.30, PS_X + PS_W - 0.30], [dv_y, dv_y],
        color=C_GRID, lw=1.2, zorder=3)

for i, (col, phase, yrs, desc) in enumerate([
    (C_TEST, "Model Dev",
     "2004 – 2011",
     "TSCV validation  ·  model building"),
    (C_BT,   "Unseen Data Test",
     "2012 – 2025",
     "True out-of-sample backtest"),
]):
    row_cy = PS_Y + PS_H * 0.745 - i * (PS_H * 0.49)
    # coloured badge
    rbox(ax, PS_X + 0.28, row_cy - 0.30, 2.20, 0.62, col,
         zorder=4, r=0.15)
    t(ax, PS_X + 1.38, row_cy + 0.01, phase, fs=8.2, z=5)
    # year + description to the right
    t(ax, PS_X + 2.74, row_cy + 0.13, yrs,
      fs=9.2, c=C_TEXT, fw="bold", ha="left", z=5)
    t(ax, PS_X + 2.74, row_cy - 0.18, desc,
      fs=7.4, c=C_MUTED, fw="normal", ha="left", style="italic", z=5)


# ═════════════════════════════════════════════════════════════════════════════
# ③ ONE TSCV STEP — DETAILED TIMELINE
# ═════════════════════════════════════════════════════════════════════════════

t(ax, 0.40, 11.80,
  "③ ONE TSCV STEP — DETAILED TIMELINE  "
  "(validation period placed earlier in time than training period)",
  fs=10, c=C_TEXT, ha="left")

# section background
rbox(ax, 0.25, 1.80, 21.50, 9.66,
     C_PANEL, ec="#B0BEC5", lw=1.5, zorder=1, r=0.5)

# time axis
TL_Y = 11.08
arr(ax, 0.55, TL_Y, 21.20, TL_Y, c=C_ARROW, lw=1.8)
t(ax, 0.50, TL_Y, "Older", fs=8.5, c=C_TEXT, ha="right")
t(ax, 21.28, TL_Y, "Newer →", fs=8.5, c=C_TEXT, ha="left")

# ── period geometry ───────────────────────────────────────────────────────────
BH  = 4.50    # height of each period rectangle
BY  = 6.08    # bottom y of periods

VX  = 0.55;   VW  = 5.10    # validation
G1  = 0.88
TX  = VX + VW + G1;  TW = 5.90    # training
G2  = 0.88
XX  = TX + TW + G2;  XW = 3.20    # model dev test
SEP = 0.16
BX  = XX + XW + SEP; BW = 3.20    # unseen data test
# right edge: 0.55+5.10+0.88+5.90+0.88+3.20+0.16+3.20 = 19.87 < 22 ✓

rbox(ax, VX, BY, VW, BH, C_VAL)
rbox(ax, TX, BY, TW, BH, C_TRAIN)
rbox(ax, XX, BY, XW, BH, C_TEST)
rbox(ax, BX, BY, BW, BH, C_BT)

# period header labels
for bx, bw, lbl in [
    (VX, VW, "VALIDATION\nPERIOD"),
    (TX, TW, "TRAINING\nPERIOD"),
    (XX, XW, "MODEL DEV\nTEST"),
    (BX, BW, "UNSEEN DATA\nTEST"),
]:
    t(ax, bx + bw/2, BY + BH - 0.45, lbl, fs=9.5)

# dashed separator between the two test periods
ax.plot([BX - SEP/2, BX - SEP/2], [BY + 0.16, BY + BH - 0.16],
        color="#B0BEC5", lw=1.8, ls="--", zorder=4)

# gap annotations + dashed borders
for gx, gw in [(VX + VW, G1), (TX + TW, G2)]:
    t(ax, gx + gw/2, BY + BH/2, "≥ 61\ntrading\ndays",
      fs=7.8, c=C_MUTED, fw="normal", style="italic")
    for lx in [gx + 0.09, gx + gw - 0.09]:
        ax.plot([lx, lx], [BY + 0.20, BY + BH - 0.20],
                color=C_GRID, lw=1.2, ls="--", zorder=2)

# ── bullet text inside periods ────────────────────────────────────────────────
def buls(ax, x, ytop, lines, fs=7.9, c="white", gap=0.40):
    for i, ln in enumerate(lines):
        ax.text(x, ytop - i * gap, ln,
                ha="left", va="top", fontsize=fs, color=c, zorder=5)

buls(ax, VX + 0.22, BY + BH - 0.86, [
    "• Restricted subset only",
    "  (bdfr = 1, days_from_report = 1–5)",
    "• Capped at 5,000 rows",
    "  (most recent observations)",
    "• XGBoost trains with eval watchlist",
    "• Early stopping: patience = 30 rounds",
    "→  Selects best num_boost_round",
])

buls(ax, TX + 0.22, BY + BH - 0.86, [
    "• All observations  (no subset filter)",
    "• Window ≤ 90 calendar days",
    "• Max 100,000 rows  (most recent)",
    "• Time-decay weights: 1 / √(age + λ)",
    "• Z-score features & label",
    "• XGBoost: eta=0.02, max_depth=3",
    "→  Trains on selected num_boost_round",
])

for bx, yrs in [(XX, "2004 – 2011"), (BX, "2012 – 2025")]:
    buls(ax, bx + 0.20, BY + BH - 0.86, [
        "• Restricted subset",
        f"• One forward month  ({yrs})",
        "",
        "Evaluation metrics:",
        "  · Compounded Return",
        "  · Quarters > Benchmark",
        "  · Max Drawdown",
        "  · Max Time Below Benchmark",
    ], gap=0.38)

# ── step annotation boxes ─────────────────────────────────────────────────────
STEP_Y = 4.55   # bottom of annotation boxes; height=0.72 → top=5.27; gap to BY=0.81

for cx, col, lbl in [
    (VX  + VW/2,  C_VAL,   "① Optimize\nnum_boost_round"),
    (TX  + TW/2,  C_TRAIN, "② Train\nXGBoost model"),
    (XX  + XW/2,  C_TEST,  "③a Evaluate\n(Model Dev)"),
    (BX  + BW/2,  C_BT,    "③b Evaluate\n(Unseen Data)"),
]:
    arr(ax, cx, BY, cx, STEP_Y + 0.78, c=col, lw=1.8)
    rbox(ax, cx - 1.22, STEP_Y, 2.44, 0.72, col, zorder=4)
    t(ax, cx, STEP_Y + 0.36, lbl, fs=8.5)

# ── flow notes ────────────────────────────────────────────────────────────────
FLOW_Y = 3.72

for cx1, cx2, note in [
    (VX + VW/2,  TX + TW/2, "best num_boost_round"),
    (TX + TW/2,  XX + XW/2, "trained model"),
]:
    arr(ax, cx1, FLOW_Y, cx2 - 0.20, FLOW_Y, c=C_MUTED, lw=1.5, style="-|>")
    t(ax, (cx1 + cx2)/2, FLOW_Y - 0.32, note,
      fs=7.5, c=C_MUTED, fw="normal", style="italic")


# ── save ──────────────────────────────────────────────────────────────────────
out = "architecture_diagram.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.show()
