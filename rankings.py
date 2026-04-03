import pandas as pd
import numpy as np
import os

os.makedirs("data_proc", exist_ok=True)

merged = pd.read_csv("data_proc/merged_rushers.csv")

# run situations: early downs + 3rd and short (< 4 yards to go)
# these are downs where the offense can credibly run the ball
# pass situations: 3rd and long (7+ yards), offense almost certainly passing
# 3rd and medium (4-6 yards) is ambiguous and excluded from the split
merged["sit_group"] = merged.apply(lambda r:
    "run_situation"  if r["down"] in [1, 2] or (r["down"] == 3 and r["distance"] < 4) else
    "pass_situation" if r["down"] == 3 and r["distance"] >= 7 else
    "ambiguous", axis=1)

def build_stats(df, min_snaps=50, min_pressures=5):
    base = df.groupby(["player", "alignment"]).agg(
        snaps      = ("gp_key",        "count"),
        pressures  = ("pressure",      "sum"),
        sacks      = ("sack",          "sum"),
        forced_fum = ("forced_fumble", "sum"),
        batted     = ("batted_pass",   "sum"),
    ).reset_index()

    # avg TTP only on plays where this specific player got to the QB
    ttp = (df[df["pressure"] == 1]
           .groupby("player")["time_to_pressure"]
           .mean()
           .reset_index()
           .rename(columns={"time_to_pressure": "avg_ttp"}))

    base = base.merge(ttp, on="player", how="left")
    base["pressure_rate"] = base["pressures"] / base["snaps"]
    base["sack_rate"]     = base["sacks"]     / base["snaps"]
    base["p2s"]           = base["sacks"]     / base["pressures"].replace(0, np.nan)

    return base[(base["snaps"] >= min_snaps) & (base["pressures"] >= min_pressures)].copy()


overall = build_stats(merged, min_snaps=100, min_pressures=10)

run_raw  = build_stats(merged[merged["sit_group"] == "run_situation"],  50, 5)
pass_raw = build_stats(merged[merged["sit_group"] == "pass_situation"], 20, 3)

run_sit  = (run_raw[["player", "snaps", "pressures", "pressure_rate"]]
            .rename(columns={"snaps": "run_snaps", "pressures": "run_pressures",
                              "pressure_rate": "run_prate"}))
pass_sit = (pass_raw[["player", "snaps", "pressures", "pressure_rate"]]
            .rename(columns={"snaps": "pass_snaps", "pressures": "pass_pressures",
                              "pressure_rate": "pass_prate"}))

full = overall.merge(run_sit,  on="player", how="left")
full = full.merge(pass_sit, on="player", how="left")

# positive = more effective in obvious pass situations than run situations
# large positive = passing-down specialist; near zero = every-down threat
full["sit_uplift"] = full["pass_prate"] - full["run_prate"]


def normalize(s):
    mn, mx = s.min(), s.max()
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.5, index=s.index)


# normalize within position groups separately so edge and DI are on their own scale
for pos in ["edge", "defensive_interior"]:
    idx = full["alignment"] == pos
    sub = full[idx].copy()

    full.loc[idx, "n_prate"] = normalize(sub["pressure_rate"])

    # lower TTP is better (faster to the QB), so invert after normalizing
    ttp_filled = sub["avg_ttp"].fillna(sub["avg_ttp"].median())
    full.loc[idx, "n_speed"] = 1 - normalize(ttp_filled)

    # missing p2s filled with position median rather than zero
    full.loc[idx, "n_conv"] = normalize(sub["p2s"].fillna(sub["p2s"].median()))


# pressure rate carries the most weight,  it's the most stable signal
# conversion second, speed third (TTP has the narrowest spread of the three)
full["composite"] = (
    0.45 * full["n_prate"] +
    0.30 * full["n_conv"]  +
    0.25 * full["n_speed"]
) * 100

full["composite"] = full["composite"].round(1)
full = full.drop_duplicates(subset=["player", "alignment"], keep="first")
full = full.sort_values(["alignment", "composite"], ascending=[True, False])

edge_out = full[full["alignment"] == "edge"].reset_index(drop=True)
di_out   = full[full["alignment"] == "defensive_interior"].reset_index(drop=True)

cols = ["player", "snaps", "pressures", "pressure_rate", "sacks",
        "p2s", "avg_ttp", "composite", "run_prate", "pass_prate", "sit_uplift"]

print("Top 15 Edge Rushers")
print(edge_out.head(15)[cols].round(3).to_string(index=False))

print("\nTop 15 Interior Linemen")
print(di_out.head(15)[cols].round(3).to_string(index=False))

print(f"\nedge qualified:     {len(edge_out)}")
print(f"interior qualified: {len(di_out)}")

print("\nEdge Position Medians (100+ snaps, 10+ pressures)")
print(f"  pressure rate:  {edge_out['pressure_rate'].median():.3f}")
print(f"  p2s conversion: {edge_out['p2s'].median():.3f}")
print(f"  avg TTP:        {edge_out['avg_ttp'].median():.3f}s")

full.to_csv("data_proc/pass_rusher_scores.csv", index=False)