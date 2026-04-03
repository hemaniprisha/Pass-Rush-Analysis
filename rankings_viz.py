import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")
import os

os.makedirs("rankings_viz", exist_ok=True)

df   = pd.read_csv("data_proc/pass_rusher_scores.csv")
edge = df[df["alignment"] == "edge"].copy()
edge["sack_rank"]      = edge["sacks"].rank(ascending=False, method="min")
edge["composite_rank"] = edge["composite"].rank(ascending=False, method="min")
edge["rank_gap"]       = edge["sack_rank"] - edge["composite_rank"]

e_med_pr  = edge["pressure_rate"].median()
e_med_p2s = edge["p2s"].median()

BG    = "#0f1923"
PANEL = "#141d27"
WHITE = "#f0f0f0"
SILVER= "#a8b2bc"
CYAN  = "#3eb8e0"
GOLD  = "#e8b84b"
GREEN = "#5ce0a0"
GRAY  = "#4a5568"
DIM   = "#2d3748"

def quad_color(r):
    if r.pressure_rate > e_med_pr and r.p2s > e_med_p2s:  return GREEN
    if r.pressure_rate > e_med_pr and r.p2s <= e_med_p2s: return CYAN
    if r.pressure_rate <= e_med_pr and r.p2s > e_med_p2s: return GOLD
    return DIM


# viz 1: archetype scatter
# Breeze and Stroman are genuinely finishers (correct quadrant = gold)
# diamond marker flags them as sack-inflated without misrepresenting their position
label_config = {
    "Jalen Wilson":      {"color": GREEN, "offset": ( 6,  7),  "marker": "o", "inflate": False},
    "Devonte Hubbard":   {"color": GREEN, "offset": ( 6, -13), "marker": "o", "inflate": False},
    "Jihad Green":       {"color": GREEN, "offset": ( 6,  7),  "marker": "o", "inflate": False},
    "Jonathan Phillips": {"color": CYAN,  "offset": ( 6,  7),  "marker": "o", "inflate": False},
    "Joe Williams":      {"color": CYAN,  "offset": ( 6,  7),  "marker": "o", "inflate": False},
    "Dicaprio Hall":     {"color": GOLD,  "offset": ( 6,  7),  "marker": "o", "inflate": False},
    "Arthur Turner":     {"color": GOLD,  "offset": ( 6, -13), "marker": "o", "inflate": False},
    "T.Y. Breeze":       {"color": GOLD,  "offset": (-56, 10), "marker": "D", "inflate": True},
    "Jamal Stroman":     {"color": GOLD,  "offset": (  6,  7), "marker": "D", "inflate": True},
}

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)

ax.axvspan(e_med_pr, edge["pressure_rate"].max()*1.15, ymin=0, ymax=1, alpha=0.03, color=GREEN)
ax.axhspan(e_med_p2s, edge["p2s"].max()*1.1,           xmin=0, xmax=1, alpha=0.03, color=GOLD)
ax.axvline(e_med_pr,  color=GRAY, lw=1.0, ls="--")
ax.axhline(e_med_p2s, color=GRAY, lw=1.0, ls="--")

xmax = edge["pressure_rate"].max() * 1.10
ymax = edge["p2s"].max()           * 1.04
xmin = edge["pressure_rate"].min() * 0.88
ymin = 0.0

ax.text(xmax, ymax,  "ELITE",     ha="right", va="top",    fontsize=9, color=GREEN,  fontweight="bold", alpha=0.75)
ax.text(xmin, e_med_p2s + (ymax - e_med_p2s)*0.35, "FINISHER",
        ha="left",  va="center",  fontsize=9, color=GOLD,   fontweight="bold", alpha=0.75)
ax.text(xmax, ymin,  "DISRUPTOR", ha="right", va="bottom", fontsize=9, color=CYAN,   fontweight="bold", alpha=0.75)
ax.text(xmin, ymin,  "LIMITED",   ha="left",  va="bottom", fontsize=9, color=SILVER, fontweight="bold", alpha=0.4)

for _, r in edge.iterrows():
    if r["player"] in label_config: continue
    c    = quad_color(r)
    size = (r["snaps"] / edge["snaps"].max()) * 130 + 15
    ax.scatter(r["pressure_rate"], r["p2s"], s=size, c=c, alpha=0.18, zorder=2, edgecolors="none")

for name, cfg in label_config.items():
    row = edge[edge["player"] == name]
    if row.empty: continue
    r    = row.iloc[0]
    size = (r["snaps"] / edge["snaps"].max()) * 130 + 15
    ec   = GOLD if cfg["inflate"] else WHITE
    ax.scatter(r["pressure_rate"], r["p2s"], s=size*2.2, c=cfg["color"],
               alpha=1.0, zorder=5, marker=cfg["marker"], edgecolors=ec, linewidths=1.2)
    dx, dy = cfg["offset"]
    ax.annotate(name.split()[-1], (r["pressure_rate"], r["p2s"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=WHITE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc=BG, ec="none", alpha=0.75))

ax.set_xlabel("Pressure Rate  (pressures per snap)", color=SILVER, fontsize=10, labelpad=8)
ax.set_ylabel("Pressure-to-Sack Conversion", color=SILVER, fontsize=10, labelpad=8)
ax.tick_params(colors=SILVER, labelsize=8)
for sp in ax.spines.values(): sp.set_color(GRAY)
ax.grid(color="#1e2a38", lw=0.4)

diamond_handle = mlines.Line2D([], [], color=GOLD, marker="D", linestyle="None",
                                markersize=7, markeredgecolor=GOLD,
                                label="Sack total overstates their impact")
legend_handles = [
    mpatches.Patch(color=GREEN, label="Elite: above median on both pressure rate and conversion"),
    mpatches.Patch(color=CYAN,  label="Disruptor: generates pressure frequently but rarely finishes"),
    mpatches.Patch(color=GOLD,  label="Finisher: efficient when they get there, but low volume"),
    diamond_handle,
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True,
          framealpha=0.25, edgecolor=GRAY, fontsize=8.5, labelcolor=WHITE)

ax.set_title("Pass Rush Archetypes among Edge Rushers in the 2023 NFL Season",
             color=WHITE, fontsize=13, fontweight="bold", pad=14)
fig.text(0.5, 0.01,
    "127 edge rushers qualified (100+ snaps, 10+ pressures).  "
    "Bubble size = snaps played.  Dashed lines = position group medians.",
    ha="center", fontsize=7.5, color=SILVER)

plt.tight_layout(rect=[0, 0.035, 1, 1])
plt.savefig("rankings_viz/archetypes.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()


# viz 2: rank displacement
underrated_names = ["Jonathan Phillips", "Joe Williams", "Denico Harris", "Jordan Nixon"]
overrated_names  = ["T.Y. Breeze", "Jamal Stroman", "Nate Kinlaw", "Josh Tindall"]

under = edge[edge["player"].isin(underrated_names)].sort_values("rank_gap", ascending=False)
over  = edge[edge["player"].isin(overrated_names)].sort_values("rank_gap",  ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.97, wspace=0.35)

MAX_RANK = int(edge["sack_rank"].max()) + 5

def arrow_panel(ax, data, title, arrow_color):
    ax.set_facecolor(PANEL)
    y = np.arange(len(data))

    sack_vals = data["sack_rank"].values
    comp_vals = data["composite_rank"].values

    for i, (sv, cv) in enumerate(zip(sack_vals, comp_vals)):
        ax.annotate("", xy=(cv, i), xytext=(sv, i),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.8, mutation_scale=14))
        ax.plot([sv, cv], [i, i], color=arrow_color, lw=1.4, alpha=0.35, zorder=1)

    ax.scatter(sack_vals, y, s=85, c=SILVER,      zorder=4, marker="o")
    ax.scatter(comp_vals, y, s=85, c=arrow_color, zorder=4, marker="D")

    ax.set_yticks(y)
    ax.set_yticklabels(data["player"].values, fontsize=10, color=WHITE)
    ax.invert_yaxis()
    ax.set_xlabel("Rank among edge rushers  (rank 1 = best)", color=SILVER, fontsize=9)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=SILVER, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRAY)
    ax.grid(axis="x", color="#1e2a38", lw=0.4)
    ax.set_xlim(0, MAX_RANK)

    for i, (sv, cv) in enumerate(zip(sack_vals, comp_vals)):
        gap = int(abs(sv - cv))
        ax.text((sv + cv) / 2, i + 0.38, f"{gap} spots",
                ha="center", fontsize=7.5, color=arrow_color,
                fontweight="bold", alpha=0.9, clip_on=True)

    handles = [
        mpatches.Patch(color=SILVER,      label="Rank by sacks alone"),
        mpatches.Patch(color=arrow_color, label="Rank by composite score"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.2, edgecolor=GRAY, fontsize=8, labelcolor=WHITE)

arrow_panel(axes[0], under, "Hidden by Sack Totals",   GREEN)
arrow_panel(axes[1], over,  "Inflated by Sack Totals", RED := "#e05c5c")

fig.suptitle("What Sack Totals Get Wrong among Edge Rushers in 2023",
             color=WHITE, fontsize=13, fontweight="bold")
fig.text(0.5, 0.02,
    "Top 10 by sacks and top 10 by composite share only 5 players.  "
    "Arrows show how a player's rank changes when the full pressure profile is considered.  "
    "127 edge rushers qualified (100+ snaps, 10+ pressures).",
    ha="center", fontsize=7.5, color=SILVER)

plt.savefig("rankings_viz/rank_displacement.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()


# viz 3: run vs pass situation pressure rate
edge2     = edge.dropna(subset=["run_prate", "pass_prate", "sit_uplift"]).copy()
plot_data = edge2.sort_values("composite", ascending=False).head(14).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(PANEL)
fig.subplots_adjust(bottom=0.28)

y = np.arange(len(plot_data))
H = 0.28

for i, (_, row) in enumerate(plot_data.iterrows()):
    uplift = row["sit_uplift"]
    run_pr = row["run_prate"]
    pas_pr = row["pass_prate"]

    # every-down threat: less than 5pp difference across situations
    # passing-down specialist: 5pp or more uplift in obvious pass situations
    color = GREEN if uplift < 0.05 else GOLD

    ax.barh(i + H/2, run_pr, H, color=color, alpha=0.35, zorder=2)
    ax.barh(i - H/2, pas_pr, H, color=color, alpha=1.0,  zorder=2)

    right = max(run_pr, pas_pr) + 0.003
    ax.plot([right, right], [i - H/2, i + H/2], color=color, lw=1.2, alpha=0.5)

    sign = "+" if uplift >= 0 else ""
    ax.text(right + 0.004, i, f"{sign}{uplift*100:.1f}pp in pass situations",
            va="center", fontsize=7.5, color=color, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(plot_data["player"], fontsize=9.5, color=WHITE)
ax.invert_yaxis()
ax.set_xlabel("Pressure Rate", color=SILVER, fontsize=10, labelpad=8)
ax.set_title("Top Edge Rushers: Pressure Rate in Run Situations vs. Obvious Pass Situations",
             color=WHITE, fontsize=12, fontweight="bold", pad=12)
ax.tick_params(colors=SILVER, labelsize=8)
for sp in ax.spines.values(): sp.set_color(GRAY)
ax.grid(axis="x", color="#1e2a38", lw=0.4)

legend_handles = [
    mpatches.Patch(color=WHITE, alpha=0.35, label="Duller bar = run situations (1st/2nd down + 3rd & short)"),
    mpatches.Patch(color=WHITE, alpha=1.0,  label="Brighter bar = obvious pass situations (3rd & 7+)"),
    mpatches.Patch(color=GREEN, label="Every-down threat: less than 5pp difference across situations"),
    mpatches.Patch(color=GOLD,  label="Passing-down specialist: 5pp or more uplift in pass situations"),
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.10), ncol=2,
           frameon=True, framealpha=0.25, edgecolor=GRAY,
           fontsize=8, labelcolor=WHITE)

fig.text(0.5, 0.02,
    "Run situations = 1st/2nd down + 3rd & short (under 4 yards).  "
    "Pass situations = 3rd & long (7+ yards).  "
    "127 edge rushers qualified (100+ snaps, 10+ pressures).",
    ha="center", fontsize=7.5, color=SILVER)

plt.savefig("rankings_viz/consistency.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()