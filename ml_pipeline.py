import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

os.makedirs("data_proc", exist_ok=True)

df = pd.read_csv("data_proc/merged_rushers.csv")

# situational flags engineered from raw columns
df["score_diff"]       = df["def_score"] - df["off_score"]
df["is_third_down"]    = (df["down"] == 3).astype(int)
df["is_obvious_pass"]  = ((df["down"] == 3) & (df["distance"] >= 7)).astype(int)
df["is_shotgun"]       = (df["off_qb_alignment"] == "SHOTGUN").astype(int)
df["is_rollout"]       = (df["dropback_type"]    == "Rollout").astype(int)
df["is_edge"]          = (df["alignment"]        == "edge").astype(int)
df["rusher_share"]     = df["def_pass_rusher_count"] / (df["def_dl_count"] + df["def_lb_count"] + 1)
df["blitz"]            = (df["def_pass_rusher_count"] >= 5).astype(int)
df["late_game"]        = (df["quarter"] >= 4).astype(int)

# ordinal encode formation by frequency, rare formations collapse to highest index
fe_map = {v: i for i, v in enumerate(df["offensive_formation_group"].value_counts().index)}
df["formation_enc"] = df["offensive_formation_group"].map(fe_map).fillna(0)

FEATURES = [
    "down", "distance", "field_position", "expected_points",
    "def_pass_rusher_count", "def_box_count", "def_high_safety_count",
    "off_rb_count", "off_te_count", "off_wr_count",
    "def_dl_count", "def_lb_count", "def_db_count",
    "no_huddle", "is_play_action", "quarter",
    "score_diff", "is_third_down", "is_obvious_pass",
    "is_shotgun", "is_rollout", "is_edge",
    "rusher_share", "blitz", "late_game", "formation_enc", "week"
]

X          = df[FEATURES].values
y_pressure = (df["pressure"] >= 1).astype(int).values


# model 1: pressure probability on a given player-play
# 9% positive rate requires scale_pos_weight to prevent the model collapsing to all-negative
print("Model 1: pressure probability (play-level)")

X_tr, X_te, y_tr, y_te = train_test_split(X, y_pressure, test_size=0.2,
                                            random_state=42, stratify=y_pressure)
scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

m1 = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=scale_pos, eval_metric="logloss",
                        base_score=0.5, random_state=42, n_jobs=-1, verbosity=0)
m1.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

proba1   = m1.predict_proba(X_te)[:, 1]
dummy    = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
auc1     = roc_auc_score(y_te, proba1)
ap1      = average_precision_score(y_te, proba1)
auc_base = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])

print(f"  AUC:           {auc1:.4f}")
print(f"  baseline AUC:  {auc_base:.4f}")
print(f"  avg precision: {ap1:.4f}")
print(f"  positive rate: {y_pressure.mean():.3f}")


# model 2: sack probability conditioned on pressure already occurring
# this isolates finishing ability from pressure generation
print("\nModel 2: sack probability given pressure")

pressure_df = df[df["pressure"] >= 1].copy()
X2 = pressure_df[FEATURES].values
y2 = (pressure_df["sack"] > 0).astype(int).values

print(f"  pressure plays: {len(pressure_df)}, sack rate: {y2.mean():.3f}")

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2,
                                                random_state=42, stratify=y2)
scale2 = (y2_tr == 0).sum() / (y2_tr == 1).sum()

m2 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=scale2, eval_metric="logloss",
                        base_score=0.5, random_state=42, n_jobs=-1, verbosity=0)
m2.fit(X2_tr, y2_tr, eval_set=[(X2_te, y2_te)], verbose=False)

proba2 = m2.predict_proba(X2_te)[:, 1]
auc2   = roc_auc_score(y2_te, proba2)
ap2    = average_precision_score(y2_te, proba2)

print(f"  AUC:           {auc2:.4f}")
print(f"  avg precision: {ap2:.4f}")


# model 3: rusher-level pressure rate regression
# aggregating to the player level and asking how much snap context predicts
# observed pressure rate, the residual is closer to true individual skill
print("\nModel 3: rusher-level pressure rate regression")

rusher_agg = df.groupby("player").agg(
    snaps            = ("pressure",              "count"),
    pressures        = ("pressure",              "sum"),
    avg_down         = ("down",                  "mean"),
    avg_distance     = ("distance",              "mean"),
    avg_ep           = ("expected_points",       "mean"),
    avg_rushers      = ("def_pass_rusher_count", "mean"),
    avg_box          = ("def_box_count",         "mean"),
    avg_safeties     = ("def_high_safety_count", "mean"),
    pct_edge         = ("is_edge",               "mean"),
    pct_blitz        = ("blitz",                 "mean"),
    pct_shotgun      = ("is_shotgun",            "mean"),
    pct_obvious_pass = ("is_obvious_pass",       "mean"),
    pct_play_action  = ("is_play_action",        "mean"),
    avg_rb           = ("off_rb_count",          "mean"),
    avg_te           = ("off_te_count",          "mean"),
    avg_wr           = ("off_wr_count",          "mean"),
).reset_index()

rusher_agg = rusher_agg[rusher_agg["snaps"] >= 50]
rusher_agg["pressure_rate"] = rusher_agg["pressures"] / rusher_agg["snaps"]

FEAT3 = ["avg_down", "avg_distance", "avg_ep", "avg_rushers", "avg_box", "avg_safeties",
         "pct_edge", "pct_blitz", "pct_shotgun", "pct_obvious_pass", "pct_play_action",
         "avg_rb", "avg_te", "avg_wr", "snaps"]

X3 = rusher_agg[FEAT3].values
y3 = rusher_agg["pressure_rate"].values

X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)

m3 = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       base_score=0.5, random_state=42, n_jobs=-1, verbosity=0)
m3.fit(X3_tr, y3_tr, eval_set=[(X3_te, y3_te)], verbose=False)

pred3   = m3.predict(X3_te)
r2      = r2_score(y3_te, pred3)
rmse    = np.sqrt(mean_squared_error(y3_te, pred3))
r2_base = r2_score(y3_te, np.full_like(y3_te, y3_tr.mean()))

print(f"  rushers qualified: {len(rusher_agg)}")
print(f"  R2:                {r2:.4f}")
print(f"  baseline R2:       {r2_base:.4f}")
print(f"  RMSE:              {rmse:.4f}")

with open("data_proc/models.pkl", "wb") as f:
    pickle.dump({
        "m1": m1, "m2": m2, "m3": m3,
        "features1": FEATURES, "features3": FEAT3,
        "X_te1": X_te,  "y_te1": y_te,
        "X_te2": X2_te, "y_te2": y2_te,
        "X_te3": X3_te, "y_te3": y3_te,
        "rusher_agg": rusher_agg,
        "auc1": auc1, "ap1": ap1,
        "auc2": auc2, "ap2": ap2,
        "r2": r2, "rmse": rmse, "r2_base": r2_base,
        "fe_map": fe_map,
    }, f)
