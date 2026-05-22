# ==========================================================
# XGBoost Restricted-Domain Model for Removal Efficiency (RE)
# ==========================================================

# ===============================
# 0. Imports
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
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
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
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # Handle decimal comma
    if re.match(r"^\d+,\d+$", x):
        x = x.replace(",", ".")
    else:
        x = x.split(",")[0]

    x = x.replace("±", "").replace("~", "").replace("%", "").strip()

    match = re.search(r"[-+]?\d*\.?\d+", x)
    return float(match.group()) if match else np.nan


for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

df = df.dropna(subset=numeric_cols).reset_index(drop=True)
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
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]


# ===============================
# 5. One-Hot Encode Adsorbate
# ===============================
df = pd.get_dummies(df, columns=["Adsorbate"], drop_first=True)


# ===============================
# 6. Regime Definition
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
# 7. RESTRICT DOMAIN (IMPORTANT)
# ===============================
MIN_SAMPLES = 10
regime_counts = df["Regime"].value_counts()
valid_regimes = regime_counts[regime_counts >= MIN_SAMPLES].index

df = df[df["Regime"].isin(valid_regimes)].reset_index(drop=True)

print("\nRestricted-domain regime counts:")
print(df["Regime"].value_counts())


# ===============================
# 8. Feature Matrix & Target
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
# 9. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, df["Regime"],
    test_size=0.2,
    random_state=42,
    stratify=df["Regime"]
)


# ===============================
# 10. Sample Weights
# ===============================
regime_counts = r_train.value_counts()
sample_weights = r_train.map(lambda r: 1 / regime_counts[r])


# ===============================
# 11. Global XGBoost
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
# 12. Residual Learning
# ===============================
train_residuals = y_train - xgb_global.predict(X_train)

residual_models = {}

for r in r_train.unique():
    idx = r_train == r
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
    residual_models[r] = model


# ===============================
# 13. Final Prediction
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
# 14. Evaluation
# ===============================
print("\n================ RESTRICTED-DOMAIN RESULTS ================")
print("Test R²  :", round(r2_score(y_test, y_pred), 4))
print("Test RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 4))
