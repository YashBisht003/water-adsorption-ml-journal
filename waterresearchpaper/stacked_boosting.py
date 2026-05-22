# ==========================================================
# STACKED ENSEMBLE MODEL
# XGBoost + LightGBM + CatBoost
# FIXED: numeric-only, no leakage, research-grade
# ==========================================================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

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
    "Surface area(m2/g)", "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)", "Contact Time (min.)",
    "Dose(g/L)", "RPM", "Initial pH", "T(K)", "RE"
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
# 4. ENCODE ADSORBATE (ONLY CATEGORICAL WE KEEP)
# ==========================================================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)

# ==========================================================
# 5. DROP NON-PHYSICAL IDENTIFIERS (CRITICAL FIX)
# ==========================================================
df = df.drop(columns=["Adsorbent", "Ref."], errors="ignore")

# ==========================================================
# 6. FEATURE MATRIX
# ==========================================================
FEATURES = [c for c in df.columns if c != "RE"]

X = df[FEATURES]
y = df["RE"]

# Ensure numeric-only matrix (XGBoost safe)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0.0)

# ==========================================================
# 7. TRAIN-TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# 8. BASE MODELS
# ==========================================================
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgbm = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=20,
    min_data_in_leaf=25,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=False
)

models = [xgb, lgbm, cat]

# ==========================================================
# 9. OOF PREDICTIONS (LEAKAGE-FREE STACKING)
# ==========================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((X_train.shape[0], len(models)))
test_preds = np.zeros((X_test.shape[0], len(models)))

for i, model in enumerate(models):
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_preds[val_idx, i] = model.predict(X_train.iloc[val_idx])

    # Retrain on full training set
    model.fit(X_train, y_train)
    test_preds[:, i] = model.predict(X_test)

# ==========================================================
# 10. META-LEARNER
# ==========================================================
meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_train)

y_pred = meta.predict(test_preds)

# ==========================================================
# 11. EVALUATION
# ==========================================================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=========== STACKED ENSEMBLE RESULTS ===========")
print("Test R²  :", round(r2, 4))
print("Test RMSE:", round(rmse, 4))
