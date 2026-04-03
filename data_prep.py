import pandas as pd
import numpy as np
import os

os.makedirs("data_proc", exist_ok=True)

plays   = pd.read_csv("data_raw/plays.csv")
defense = pd.read_csv("data_raw/defense.csv")

# play_id resets each game, so compositing with game_id gives a globally unique key
plays["gp_key"]   = plays["game_id"]   + "_" + plays["play_id"]
defense["gp_key"] = defense["game_id"] + "_" + defense["play_id"]

# edge and interior linemen only, LBs/DBs blitzing are captured via the
# blitz flag engineered later, keeping this focused on true pass rush alignments
rushers = defense[defense["alignment"].isin(["edge", "defensive_interior"])].copy()

play_cols = [
    "gp_key", "down", "distance", "field_position",
    "def_pass_rusher_count", "def_box_count", "def_high_safety_count",
    "off_rb_count", "off_te_count", "off_wr_count", "off_qb_alignment",
    "no_huddle", "is_play_action", "dropback_type",
    "def_dl_count", "def_lb_count", "def_db_count",
    "is_no_play", "expected_points", "quarter",
    "off_score", "def_score", "week",
    "time_to_throw", "pass_rush_result", "time_to_pressure",
    "offensive_formation_group"
]

merged = rushers.merge(plays[play_cols], on="gp_key", how="inner")

# drop nullified plays and snaps where no one was actually rushing
merged = merged[merged["is_no_play"] == 0]
merged = merged[merged["def_pass_rusher_count"] > 0]

print(f"player-play rows: {len(merged)}")
print(f"unique players:   {merged['player'].nunique()}")
print(f"pressure rate:    {merged['pressure'].mean():.3f}")
print(f"sack rate:        {merged['sack'].mean():.4f}")

merged.to_csv("data_proc/merged_rushers.csv", index=False)