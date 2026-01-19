# ============================================================
# dp.py
# MAHARASHTRA UIDAI – LGD STANDARDIZED PIPELINE (FINAL, STABLE)
# CSV + POSTMASTER + GEOJSON
# ============================================================

import pandas as pd
import unicodedata
import json
import re
from datetime import datetime

STATE_FILTER = "maharashtra"

# ------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ------------------------------------------------------------
# NORMALIZER (USED EVERYWHERE)
# ------------------------------------------------------------
def normalize(val):
    if pd.isna(val):
        return None
    val = unicodedata.normalize("NFKD", str(val))
    val = val.encode("ascii", "ignore").decode("utf-8")
    val = val.lower()
    val = re.sub(r"\(.*?\)", "", val)
    val = re.sub(r"district", "", val)
    val = re.sub(r"[-_.]", " ", val)
    val = re.sub(r"\s+", " ", val).strip()
    return val

# ------------------------------------------------------------
# GEOJSON → LGD NAME ALIASES (AUTHORITATIVE & COMPLETE)
# ------------------------------------------------------------
GEOJSON_LGD_ALIASES = {

    # Renamed districts (LGD official)
    "ahmadnagar": "ahilyanagar",              # 466
    "ahmednagar": "ahilyanagar",
    "aurangabad": "chhatrapati sambhajinagar",# 468
    "osmanabad": "dharashiv",                  # 472

    # Spelling / census differences
    "garhchiroli": "gadchiroli",               # 486
    "gondiya": "gondia",                       # 475
    "raigarh": "raigad",                       # 488
    "buldana": "buldhana",                     # 469

    # Defensive
    "washim district": "washim",
    "hingoli district": "hingoli",
}

# ------------------------------------------------------------
# LOAD LGD MASTER (AUTHORITATIVE)
# ------------------------------------------------------------
log("Loading LGD master")

lgd = pd.read_csv("mh_district_lgd_master.csv")
lgd.columns = lgd.columns.str.lower().str.replace(" ", "_")

lgd["district_lgd_code"] = lgd["district_lgd_code"].astype(str)
lgd["district_name_norm"] = lgd["district_name_(in_english)"].apply(normalize)

# Normalized name → LGD code
LGD_NAME_TO_CODE = dict(
    zip(lgd["district_name_norm"], lgd["district_lgd_code"])
)

# LGD code → Display label
LGD_CODE_TO_LABEL = {
    row["district_lgd_code"]:
        "Ahilyanagar (Ahmednagar)"
        if row["district_lgd_code"] == "466"
        else row["district_name_(in_english)"]
    for _, row in lgd.iterrows()
}

log(f"LGD districts loaded: {len(LGD_CODE_TO_LABEL)}")

# ------------------------------------------------------------
# LOAD POST-MASTER (PINCODE → LGD)
# ------------------------------------------------------------
log("Loading Post-Master")

post = pd.read_csv("Post-Master.csv")
post.columns = post.columns.str.lower()

post["statename"] = post["statename"].apply(normalize)
post["district_norm"] = post["district"].apply(normalize)
post["pincode"] = pd.to_numeric(post["pincode"], errors="coerce")

post = post[post["statename"] == STATE_FILTER]

post["district_lgd_code"] = post["district_norm"].map(LGD_NAME_TO_CODE)

post_lgd = (
    post[["pincode", "district_lgd_code"]]
    .dropna()
    .drop_duplicates()
)

del post

# ------------------------------------------------------------
# CLEAN & AGGREGATE UIDAI FILE
# ------------------------------------------------------------
def clean_uidai(input_file, output_file):
    log(f"Processing {input_file}")

    df = pd.read_csv(input_file)
    df.columns = df.columns.str.lower()

    df["district_norm"] = df["district"].apply(normalize)
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])

    df["month_year"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
        .pipe(pd.to_datetime)
        .dt.strftime("%b-%y")
    )

    # 1️⃣ PINCODE → LGD (highest authority)
    df = df.merge(post_lgd, on="pincode", how="left")

    # 2️⃣ DISTRICT NAME → LGD fallback
    df["district_lgd_code"] = df["district_lgd_code"].combine_first(
        df["district_norm"].map(LGD_NAME_TO_CODE)
    )

    df = df.dropna(subset=["district_lgd_code"])

    metric_cols = [c for c in df.columns if c.startswith("age")]

    df = (
        df.groupby(
            ["month_year", "district_lgd_code", "pincode"],
            as_index=False
        )[metric_cols]
        .sum()
    )

    df["district_label"] = df["district_lgd_code"].map(LGD_CODE_TO_LABEL)
    df["state"] = "Maharashtra"

    df.to_csv(output_file, index=False)
    log(f"Saved {output_file} | Rows: {len(df)}")

# ------------------------------------------------------------
# RUN UIDAI PIPELINE
# ------------------------------------------------------------
clean_uidai("MHEnrol.csv", "MHEnrol_clean_agg.csv")
clean_uidai("MHDemo.csv",  "MHDemo_clean_agg.csv")
clean_uidai("MHBio.csv",   "MHBio_clean_agg.csv")

# ------------------------------------------------------------
# FIX MAHARASHTRA GEOJSON (LGD AUTHORITATIVE)
# ------------------------------------------------------------
log("Fixing Maharashtra GeoJSON (LGD authoritative)")

with open("Maharashtra.geojson", "r", encoding="utf-8") as f:
    geo = json.load(f)

fixed_features = []

for feat in geo["features"]:
    props = feat.get("properties", {})

    raw = (
        props.get("Dist_Name")
        or props.get("DIST_NAME")
        or props.get("district")
        or props.get("name")
    )

    if not raw:
        continue

    norm = normalize(raw)
    norm = GEOJSON_LGD_ALIASES.get(norm, norm)

    lgd_code = LGD_NAME_TO_CODE.get(norm)

    if not lgd_code:
        log(f"⚠️ GeoJSON unmatched district: {raw}")
        continue

    props["district_lgd_code"] = lgd_code
    props["district_label"] = LGD_CODE_TO_LABEL.get(lgd_code, raw)
    props["district_original"] = raw

    fixed_features.append(feat)

geo["features"] = fixed_features

with open("maharashtra_districts_lgd_ready.geojson", "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, indent=2)

log(f"GeoJSON fixed | Districts: {len(fixed_features)}")

# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------
csv_codes = set(
    pd.read_csv("MHEnrol_clean_agg.csv")["district_lgd_code"].astype(str)
)

geo_codes = {
    f["properties"]["district_lgd_code"]
    for f in geo["features"]
}

print("❌ CSV − GEOJSON mismatch:", csv_codes - geo_codes)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
