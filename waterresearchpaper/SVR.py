# ==========================================================
# XGBoost Restricted-Domain Model for Removal Efficiency (RE)
# COMPLETE END-TO-END SCRIPT (FINAL, ERROR-FREE)
# ==========================================================

# ===============================
# 0. Imports
# ===============================
import pandas as pd
import numpy as np
import re
import os

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# ===============================
# 1. Output directory
# ===============================
OUTPUT_DIR = "analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. Load Data
# ===============================
df = pd.read_excel("data_2023(1).xlsx")
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df = df.rename(columns={"RE (%)": "RE"})

# ===============================
# 3. Numeric Cleaning
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
# 4. Physics-Informed Features
# ===============================
df["Site_Density"] = (df["Dose(g/L)"] * df["Surface area(m2/g)"]) / df["Intial Concentration(ppm)"]
df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / df["Intial Concentration(ppm)"]
df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]

# ===============================
# 5. Encode Adsorbate
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
# 7. Restrict Domain
# ===============================
MIN_SAMPLES = 10
valid_regimes = df["Regime"].value_counts()
valid_regimes = valid_regimes[valid_regimes >= MIN_SAMPLES].index
df = df[df["Regime"].isin(valid_regimes)].reset_index(drop=True)

# ===============================
# 8. Feature Matrix
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
# 13. Prediction
# ===============================
def predict_xgb(X_input, regimes):
    base = xgb_global.predict(X_input)
    final = base.copy()
    for i, r in enumerate(regimes):
        if r in residual_models:
            final[i] += residual_models[r].predict(X_input.iloc[i:i+1])[0]
    return final

y_pred = predict_xgb(X_test, r_test)

# ===============================
# 14. Metrics (ROBUST)
# ===============================
print("\n================ FINAL RESULTS ================")
print("Test R²  :", round(r2_score(y_test, y_pred), 4))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", round(rmse, 4))

# ===============================
# 15. Diagnostic Plots
# ===============================
residuals = y_test - y_pred

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
plt.plot([0,100],[0,100],'--r')
plt.xlabel("Actual RE (%)")
plt.ylabel("Predicted RE (%)")
plt.title("Parity Plot")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/parity_global.png", dpi=300)
plt.close()

plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted RE (%)")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residuals_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(6,5))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/residual_distribution.png", dpi=300)
plt.close()

for reg in r_test.unique():
    idx = r_test == reg
    if idx.sum() < 10:
        continue
    plt.figure(figsize=(5,5))
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.8, edgecolor="k")
    plt.plot([0,100],[0,100],'--r')
    plt.title(f"Parity – {reg}")
    plt.xlabel("Actual RE (%)")
    plt.ylabel("Predicted RE (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/parity_{reg}.png", dpi=300)
    plt.close()

# ===============================
# 16. SHAP Analysis (FIXED)
# ===============================
X_train_np = X_train.values.astype(np.float64)
X_test_np  = X_test.values.astype(np.float64)

explainer = shap.TreeExplainer(xgb_global)
shap_values = explainer.shap_values(X_test_np)

shap.summary_plot(shap_values, X_test_np, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=300)
plt.close()

shap.summary_plot(
    shap_values,
    X_test_np,
    feature_names=FEATURES,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=300)
plt.close()

# ===============================
# 17. Save Predictions
# ===============================
out = X_test.copy()
out["Actual_RE"] = y_test.values
out["Predicted_RE"] = y_pred
out["Residual"] = residuals.values
out["Regime"] = r_test.values
out.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)
