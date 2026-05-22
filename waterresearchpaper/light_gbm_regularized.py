# ==========================================================
# FINAL LIGHTGBM MODEL
# GLOBAL + REGIME-WISE R² EVALUATION
# ==========================================================

import pandas as pd
import numpy as np
import re

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_excel("data_2023(1).xlsx")
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df = df.rename(columns={"RE (%)": "RE"})

# ==========================================================
# 2. ROBUST NUMERIC CLEANING
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
    x = x.replace("±", "").replace("~", "").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+", x)
    return float(m.group()) if m else np.nan

for c in numeric_cols:
    df[c] = df[c].apply(clean_numeric)

df = df.dropna(subset=numeric_cols)
df = df[(df["RE"] >= 0) & (df["RE"] <= 100)].reset_index(drop=True)

# ==========================================================
# 3. PHYSICS-INFORMED FEATURES
# ==========================================================
df["Site_Density"] = (
    df["Dose(g/L)"] * df["Surface area(m2/g)"]
) / df["Intial Concentration(ppm)"]

df["Cap_Load_Ratio"] = (
    df["Adsorption capacity(mg/g)"]
) / df["Intial Concentration(ppm)"]

df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]

# ==========================================================
# 4. ENCODE ADSORBATE (SAFE)
# ==========================================================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)

# Drop identifiers (non-physical)
df = df.drop(columns=["Adsorbent", "Ref."], errors="ignore")

# ==========================================================
# 5. DEFINE CHEMICAL REGIMES
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
# 6. FEATURE MATRIX
# ==========================================================
FEATURES = [c for c in df.columns if c not in ["RE", "Regime"]]

X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["RE"]
regimes = df["Regime"]

# ==========================================================
# 7. TRAIN-TEST SPLIT (STRATIFIED BY REGIME)
# ==========================================================
X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, regimes,
    test_size=0.2,
    random_state=42,
    stratify=regimes
)

# ==========================================================
# 8. FINAL LIGHTGBM MODEL (BEST CONFIG)
# ==========================================================
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# 9. GLOBAL PERFORMANCE
# ==========================================================
y_pred = model.predict(X_test)

global_r2 = r2_score(y_test, y_pred)
global_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n================ GLOBAL PERFORMANCE ================")
print("Global Test R² :", round(global_r2, 4))
print("Global RMSE    :", round(global_rmse, 4))

# ==========================================================
# 10. REGIME-WISE PERFORMANCE
# ==========================================================
rows = []

for reg in sorted(r_test.unique()):
    idx = r_test == reg
    n = idx.sum()

    if n < 5:
        r2 = np.nan
        rmse = np.nan
    else:
        r2 = r2_score(y_test[idx], y_pred[idx])
        rmse = np.sqrt(mean_squared_error(y_test[idx], y_pred[idx]))

    rows.append({
        "Regime": reg,
        "Samples": n,
        "R²": r2,
        "RMSE": rmse
    })

regime_df = pd.DataFrame(rows)

print("\n================ REGIME-WISE PERFORMANCE ================")
print(regime_df.to_string(index=False))

# ==========================================================
# 11. WEIGHTED REGIME R² (IMPORTANT)
# ==========================================================
valid = regime_df.dropna(subset=["R²"])
weighted_r2 = np.average(
    valid["R²"],
    weights=valid["Samples"]
)

print("\n================ WEIGHTED REGIME R² ================")
print("Weighted Regime R² :", round(weighted_r2, 4))
