# ==========================================================
# XGBoost Regime-Aware Model for Removal Efficiency (RE)
# Physics-informed, journal-grade, reproducible
# ==========================================================

# ===============================
# 0. Imports (COMPLETE)
# ===============================
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor

# ===============================
# 1. Load Data
# ===============================
df = pd.read_excel("data_2023(1).xlsx")

# Drop junk column if present
df = df.drop(columns=["Unnamed: 12"], errors="ignore")

# Rename target column
df = df.rename(columns={"RE (%)": "RE"})

# ===============================
# 2. Numeric Columns
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

# ===============================
# 3. Robust Numeric Cleaner
# ===============================
def clean_numeric(x):
    """
    Handles:
    - decimal commas (130,4 → 130.4)
    - uncertainty formats (80 ± 2 → 80)
    - approximations (~50 → 50)
    - multiple values (30, 0.1 → 30)
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # If comma is decimal separator
    if re.match(r"^\d+,\d+$", x):
        x = x.replace(",", ".")
    else:
        x = x.split(",")[0]

    # Remove symbols
    x = x.replace("±", "").replace("~", "").replace("%", "").strip()

    # Extract number
    match = re.search(r"[-+]?\d*\.?\d+", x)
    return float(match.group()) if match else np.nan


for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Drop invalid rows
df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# Physical bounds
df = df[(df["RE"] >= 0) & (df["RE"] <= 100)]

# ===============================
# 4. Physics-Informed Features
# ===============================
df["Site_Density"] = (
    df["Dose(g/L)"] * df["Surface area(m2/g)"]
) / df["Intial Concentration(ppm)"]

df["Cap_Load_Ratio"] = (
    df["Adsorption capacity(mg/g)"]
) / df["Intial Concentration(ppm)"]

df["LogTime"] = np.log1p(df["Contact Time (min.)"])

df["Mixing_Index"] = df["RPM"] * df["LogTime"]

df["Thermo_Capacity"] = (
    df["Adsorption capacity(mg/g)"] * df["T(K)"]
)

# ===============================
# 5. One-Hot Encode Adsorbate
# ===============================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)

# ===============================
# 6. Feature Matrix & Target
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
# 7. Regime Definition
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
# 8. Train-Test Split (Stratified)
# ===============================
X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, df["Regime"],
    test_size=0.2,
    random_state=42,
    stratify=df["Regime"]
)

# ===============================
# 9. Sample Weights (Imbalance Fix)
# ===============================
regime_counts = r_train.value_counts()
sample_weights = r_train.map(lambda r: 1 / regime_counts[r])

# ===============================
# 10. Global XGBoost Model
# ===============================
xgb_global = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)

xgb_global.fit(X_train, y_train, sample_weight=sample_weights)

# ===============================
# 11. Residual Learning (Regime-wise)
# ===============================
train_residuals = y_train - xgb_global.predict(X_train)

residual_models = {}

for reg in r_train.unique():
    idx = r_train == reg
    if idx.sum() < 40:
        continue

    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    model.fit(X_train[idx], train_residuals[idx])
    residual_models[reg] = model

# ===============================
# 12. Final Prediction
# ===============================
def predict_xgb(X_input, regimes):
    base = xgb_global.predict(X_input)
    final = base.copy()

    for i, r in enumerate(regimes):
        if r in residual_models:
            final[i] += residual_models[r].predict(
                X_input.iloc[i:i+1]
            )[0]
    return final

y_pred = predict_xgb(X_test, r_test)

# ===============================
# 13. Evaluation
# ===============================
print("\n================ XGBOOST RESULTS ================")
print("Test R²  :", round(r2_score(y_test, y_pred), 4))
print("Test RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 4))

print("\nRegime-wise Performance:")
rows = []
for r in r_test.unique():
    idx = r_test == r
    rows.append({
        "Regime": r,
        "Samples": idx.sum(),
        "R²": r2_score(y_test[idx], y_pred[idx]),
        "RMSE": mean_squared_error(y_test[idx], y_pred[idx], squared=False)
    })

print(pd.DataFrame(rows).sort_values("Regime"))
