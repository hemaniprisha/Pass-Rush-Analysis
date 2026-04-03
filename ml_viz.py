import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("ml_viz", exist_ok=True)

pkl_path = "data_proc/models.pkl"
if not os.path.exists(pkl_path):
    raise FileNotFoundError(
        f"Could not find {pkl_path}, run ml_pipeline.py first from the same directory"
    )

with open(pkl_path, "rb") as f:
    d = pickle.load(f)

m1, m2, m3 = d["m1"], d["m2"], d["m3"]

# fix XGBoost base_score deserialization issue
import json
def _fix_base_score(model):
    booster = model.get_booster()
    cfg = json.loads(booster.save_config())

    bs = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(bs, str):
        bs = bs.strip("[]")
        cfg["learner"]["learner_model_param"]["base_score"] = str(float(bs))

    booster.load_config(json.dumps(cfg))
    return model

m1 = _fix_base_score(m1)
m2 = _fix_base_score(m2)
m3 = _fix_base_score(m3)

# load saved model data
F1           = d["features1"]
F3           = d["features3"]
X_te1        = d["X_te1"]
X_te3        = d["X_te3"]
auc1, ap1    = d["auc1"], d["ap1"]
auc2, ap2    = d["auc2"], d["ap2"]
r2, rmse     = d["r2"],   d["rmse"]
r2_base      = d["r2_base"]

# feature name maps
name_map1 = {
    "down": "Down",
    "distance": "Yards to Go",
    "field_position": "Field Position",
    "expected_points": "Expected Points",
    "def_pass_rusher_count": "# Pass Rushers",
    "def_box_count": "Defenders in Box",
    "def_high_safety_count": "High Safeties",
    "off_rb_count": "RBs on Field",
    "off_te_count": "TEs on Field",
    "off_wr_count": "WRs on Field",
    "def_dl_count": "DL Count",
    "def_lb_count": "LB Count",
    "def_db_count": "DB Count",
    "no_huddle": "No Huddle",
    "is_play_action": "Play Action",
    "quarter": "Quarter",
    "score_diff": "Score Differential",
    "is_third_down": "3rd Down Flag",
    "is_obvious_pass": "Obvious Pass (3rd & 7+)",
    "is_shotgun": "QB in Shotgun",
    "is_rollout": "Rollout",
    "is_edge": "Edge Rusher",
    "rusher_share": "Rusher Share",
    "blitz": "Blitz (5+ Rushers)",
    "late_game": "Late Game (Q4+)",
    "formation_enc": "Formation",
    "week": "Week",
}

name_map3 = {
    "avg_down": "Avg Down Faced",
    "avg_distance": "Avg Distance Faced",
    "avg_ep": "Avg Expected Points",
    "avg_rushers": "Avg Pass Rushers",
    "avg_box": "Avg Box Count",
    "avg_safeties": "Avg High Safeties",
    "pct_edge": "% Snaps at Edge",
    "pct_blitz": "% Blitz Snaps",
    "pct_shotgun": "% vs Shotgun",
    "pct_obvious_pass": "% Obvious Pass Downs",
    "pct_play_action": "% Play Action Faced",
    "avg_rb": "Avg RBs on Field",
    "avg_te": "Avg TEs on Field",
    "avg_wr": "Avg WRs on Field",
    "snaps": "Total Snaps",
}

labels1 = [name_map1.get(f, f) for f in F1]
labels3 = [name_map3.get(f, f) for f in F3]

# colors
BG    = "#0f1923"
PANEL = "#141d27"
WHITE = "#f0f0f0"
SILVER= "#a8b2bc"
CYAN  = "#3eb8e0"
GOLD  = "#e8b84b"
GREEN = "#5ce0a0"
GRAY  = "#4a5568"
RED   = "#e05c5c"

# model performance cards
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor(BG)

cards = [
    ("Model 1\nPressure Probability",  auc1,  0.5,    "AUC-ROC", ap1,  "Avg Precision",
     "Does pre-snap context predict\nwhether pressure will occur?"),
    ("Model 2\nPressure to Sack",      auc2,  0.5,    "AUC-ROC", ap2,  "Avg Precision",
     "Given pressure, does situation\npredict whether it becomes a sack?"),
    ("Model 3\nRusher Rate Regression",r2,    r2_base,"R\u00b2",  rmse, "RMSE",
     "How much of a rusher's pressure\nrate is explained by snap context?"),
]

for ax, (title, main_val, base_val, m1_label, sec_val, m2_label, question) in zip(axes, cards):
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.4, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=WHITE)
    ax.text(5, 8.2, question, ha="center", va="top", fontsize=7.5, color=SILVER, style="italic")

    ax.text(5, 6.4, f"{main_val:.3f}", ha="center", fontsize=30, fontweight="bold", color=CYAN)
    ax.text(5, 5.4, m1_label, ha="center", fontsize=9, color=SILVER)

    delta = main_val - base_val
    ax.text(5, 4.6, f"{'+' if delta>=0 else ''}{delta:.3f} vs baseline",
            ha="center", fontsize=8.5, color=GREEN if delta > 0 else RED)

    ax.text(5, 3.4, f"{sec_val:.3f}", ha="center", fontsize=20, fontweight="bold", color=GOLD)
    ax.text(5, 2.6, m2_label, ha="center", fontsize=9, color=SILVER)

fig.suptitle("Three-Stage Pass Rush Model",
             color=WHITE, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("ml_viz/model_performance.png", dpi=160, facecolor=BG)
plt.close()

# shap feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor(BG)

for ax, (model, X_te, labels, n_feats, title) in zip(axes, [
    (m1, X_te1, labels1, 12, "Model 1: What Drives Pressure?"),
    (m3, X_te3, labels3, 12, "Model 3: What Shapes Rusher Rate?"),
]):
    ax.set_facecolor(PANEL)

    np.random.seed(42)
    idx = np.random.choice(len(X_te), min(2000, len(X_te)), replace=False)

    # shap workaround for classification vs regression
    if hasattr(model, "predict_proba"):  # classification
        explainer = shap.Explainer(model.predict_proba, X_te[idx])
        sv = explainer(X_te[idx]).values[:, :, 1]
    else:  # regression
        explainer = shap.Explainer(model.predict, X_te[idx])
        sv = explainer(X_te[idx]).values

    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:n_feats]

    vals  = mean_abs[order]
    names = [labels[i] for i in order]

    colors = [
        CYAN if v >= vals[0]*0.6 else
        GOLD if v >= vals[0]*0.3 else
        SILVER
        for v in vals
    ]

    ax.barh(range(len(vals))[::-1], vals, color=colors, height=0.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names[::-1], fontsize=9.5, color=WHITE)

    ax.set_xlabel("Mean |SHAP Value|", color=SILVER)
    ax.set_title(title, color=WHITE, fontsize=10.5, fontweight="bold")

    ax.tick_params(colors=SILVER)
    for sp in ax.spines.values():
        sp.set_color(GRAY)

fig.suptitle("SHAP Feature Importance",
             color=WHITE, fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("ml_viz/shap_importance.png", dpi=160, facecolor=BG)
plt.close()