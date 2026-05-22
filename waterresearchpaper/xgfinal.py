# ==========================================================
# Group-Aware + Regime-Weighted XGBoost (FINAL ATTEMPT)
# ==========================================================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ==========================================================
# 1. Load data
# ==========================================================
df = pd.read_excel("data_2023(1).xlsx")
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df = df.rename(columns={"RE (%)": "RE"})

# ==========================================================
# 2. Robust numeric cleaning
# ==========================================================
numeric_cols = [
    "Surface area(m2/g)",
    "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)",
    "RE"
]

def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if re.match(r"^\d+,\d+$", x):
        x = x.replace(",", ".")
    else:
        x = x.split(",")[0]
    x = x.replace("±", "").replace("~", "").replace("%", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", x)
    return float(m.group()) if m else np.nan

for c in numeric_cols:
    df[c] = df[c].apply(clean_numeric)

df = df.dropna(subset=numeric_cols)
df = df[(df["RE"] >= 0) & (df["RE"] <= 100)].reset_index(drop=True)

# ==========================================================
# 3. Feature matrix (PLAIN — no engineering)
# ==========================================================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)

base_features = [
    "Surface area(m2/g)",
    "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)"
]

adsorbate_features = [c for c in df.columns if c.startswith("Adsorbate_")]
FEATURES = base_features + adsorbate_features

X = df[FEATURES].values
y = df["RE"].values

# ==========================================================
# 4. Regime definition (for weighting)
# ==========================================================
C0_MEDIAN = df["Intial Concentration(ppm)"].median()

def assign_regime(row):
    if row["Initial pH"] < 7 and row["Intial Concentration(ppm)"] < C0_MEDIAN:
        return "Acidic_LowC0"
    elif row["Initial pH"] < 7:
        return "Acidic_HighC0"
    elif row["Intial Concentration(ppm)"] < C0_MEDIAN:
        return "Neutral_LowC0"
    else:
        return "Neutral_HighC0"

df["Regime"] = df.apply(assign_regime, axis=1)

# ==========================================================
# 5. Group-aware split (by Adsorbate)
# ==========================================================
# Use original Adsorbate column BEFORE one-hot
groups = df[[c for c in df.columns if c.startswith("Adsorbate_")]].idxmax(axis=1)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
r_train = df.loc[train_idx, "Regime"]

# ==========================================================
# 6. Regime-based sample weights
# ==========================================================
# Downweight unstable regime
regime_weights = {
    "Acidic_HighC0": 1.0,
    "Acidic_LowC0": 1.0,
    "Neutral_HighC0": 0.9,
    "Neutral_LowC0": 0.5   # noisy regime
}

sample_weights = r_train.map(regime_weights).values

# ==========================================================
# 7. XGBoost (same hyperparameters as best model)
# ==========================================================
xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

# ==========================================================
# 8. Evaluation
# ==========================================================
y_pred = xgb_model.predict(X_test)

print("\n=========== GROUP-AWARE + WEIGHTED XGBOOST ===========")
print("Test R²  :", round(r2_score(y_test, y_pred), 4))
print("Test RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 4))
