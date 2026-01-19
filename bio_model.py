import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv("MHBio_clean_agg.csv")
print("Initial rows:", df.shape[0])

# =====================================================
# 2. SAFE MONTH PARSING
# =====================================================
def parse_month(val):
    for fmt in ("%b-%y", "%Y-%m", "%Y/%m", "%Y%m", "%b %Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(val)
    except:
        return np.nan

df["month_parsed"] = df["month_year"].apply(parse_month)
df = df.dropna(subset=["month_parsed"])
df["month_index"] = df["month_parsed"].dt.year * 12 + df["month_parsed"].dt.month

print("Rows after month parsing:", df.shape[0])

# =====================================================
# 3. TARGET CLEANING
# =====================================================
TARGET_COLS = ["bio_age_5_17", "bio_age_17_"]
df = df.dropna(subset=TARGET_COLS)

# =====================================================
# 4. SORT FOR TIME SERIES
# =====================================================
df = df.sort_values(
    ["district_lgd_code", "pincode", "month_index"]
)

# =====================================================
# 5. CREATE LAG FEATURES (KEY STEP)
# =====================================================
for col in TARGET_COLS:
    df[f"{col}_lag1"] = (
        df.groupby(["district_lgd_code", "pincode"])[col]
          .shift(1)
    )

df = df.dropna()
print("Rows after lag features:", df.shape[0])

# =====================================================
# 6. ENCODE CATEGORICAL FEATURES
# =====================================================
le_pin = LabelEncoder()
le_dist_label = LabelEncoder()

df["pincode_enc"] = le_pin.fit_transform(df["pincode"].astype(str))
df["district_label_enc"] = le_dist_label.fit_transform(df["district_label"].astype(str))

# =====================================================
# 7. FEATURES & TARGETS
# =====================================================
FEATURE_COLS = [
    "month_index",
    "district_lgd_code",
    "pincode_enc",
    "district_label_enc",
    "bio_age_5_17_lag1",
    "bio_age_17__lag1"
]

X = df[FEATURE_COLS]
Y = df[TARGET_COLS]

print("Final X shape:", X.shape)
print("Final Y shape:", Y.shape)

# =====================================================
# 8. TIME-AWARE TRAIN / TEST SPLIT
# =====================================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

# =====================================================
# 9. MODEL PIPELINE (SGD – GRADIENT DESCENT)
# =====================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(
        SGDRegressor(
            loss="huber",
            learning_rate="adaptive",
            max_iter=3000,
            alpha=0.0001,
            random_state=42
        )
    ))
])

# =====================================================
# 10. TRAIN
# =====================================================
model.fit(X_train, Y_train)

# =====================================================
# 11. PREDICT
# =====================================================
Y_pred = model.predict(X_test)

# =====================================================
# 12. METRICS
# =====================================================
print("\n===== BIOMETRIC MODEL PERFORMANCE =====")
print("R2 :", round(r2_score(Y_test, Y_pred), 4))
print("MAE:", round(mean_absolute_error(Y_test, Y_pred), 2))
print("MSE:", round(mean_squared_error(Y_test, Y_pred), 2))

# =====================================================
# 13. SAVE MODEL & ENCODERS
# =====================================================
joblib.dump(model, "bio_agewise_model.pkl")
# joblib.dump(le_pin, "bio_pincode_encoder.pkl")
# joblib.dump(le_dist_label, "bio_district_label_encoder.pkl")

print("\n✅ Biometric age-wise model trained and saved")
