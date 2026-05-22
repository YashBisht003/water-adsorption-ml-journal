# ==========================================================
# LightGBM Restricted-Domain Model for Removal Efficiency
# Fair comparison with XGBoost
# ==========================================================

# ===============================
# 0. Imports
# ===============================
import pandas as pd
import numpy as np
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

# ===============================
# 1. Load Data
# ===============================
df = pd.read_excel("data_2023(1).xlsx")
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df = df.rename(columns={"RE (%)": "RE"})

# ===============================
# 2. Numeric Cleaning
# ===============================
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

# ===============================
# 3. Physics-Informed Features
# ===============================
df["Site_Density"] = (df["Dose(g/L)"] * df["Surface area(m2/g)"]) / df["Intial Concentration(ppm)"]
df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / df["Intial Concentration(ppm)"]
df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]

# ===============================
# 4. Encode Adsorbate
# ===============================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)

# ===============================
# 5. Regime Definition
# ===============================
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

# ===============================
# 6. Restrict Domain
# ===============================
MIN_SAMPLES = 10
valid_regimes = df["Regime"].value_counts()
valid_regimes = valid_regimes[valid_regimes >= MIN_SAMPLES].index
df = df[df["Regime"].isin(valid_regimes)].reset_index(drop=True)

# ===============================
# 7. Feature Matrix
# ===============================
base_features = [
    "Surface area(m2/g)",
    "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)",
    "Site_Density",
    "Cap_Load_Ratio",
    "LogTime",
    "Mixing_Index",
    "Thermo_Capacity"
]

adsorbate_features = [c for c in df.columns if c.startswith("Adsorbate_")]
FEATURES = base_features + adsorbate_features

X = df[FEATURES]
y = df["RE"]

# ===============================
# 8. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=df["Regime"]
)

# ===============================
# 9. LightGBM Model (SAFE CONFIG)
# ===============================
lgb_model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=20,          # VERY important (low!)
    max_depth=5,
    min_data_in_leaf=25,    # strong regularization
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42
)

lgb_model.fit(X_train, y_train)

# ===============================
# 10. Evaluation
# ===============================
y_pred = lgb_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=========== LIGHTGBM RESULTS ===========")
print("Test R²  :", round(r2, 4))
print("Test RMSE:", round(rmse, 4))
