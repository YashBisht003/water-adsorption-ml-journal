import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# 1. Load data
# =====================================================
df = pd.read_excel("data_2023(1).xlsx")
df = df.rename(columns={"RE (%)": "RE"})

# =====================================================
# 2. Numeric columns
# =====================================================
numeric_cols_full = [
    "Surface area(m2/g)",
    "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)"
]

# =====================================================
# 3. Fix comma decimal issue
# =====================================================
for col in numeric_cols_full + ["RE"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols_full + ["RE"])

# =====================================================
# 4. Physics-informed features (FULL MODEL)
# =====================================================
df["Site_Density"] = (
    df["Dose(g/L)"] * df["Surface area(m2/g)"]
) / df["Intial Concentration(ppm)"]

df["Cap_Load_Ratio"] = (
    df["Adsorption capacity(mg/g)"]
) / df["Intial Concentration(ppm)"]

df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]

# =====================================================
# 5. -------- FULL RANDOM FOREST MODEL --------
# =====================================================
features_full = numeric_cols_full + [
    "Site_Density",
    "Cap_Load_Ratio",
    "LogTime",
    "Mixing_Index",
    "Thermo_Capacity",
    "Adsorbate"
]

X_full = df[features_full].copy()
X_full["Adsorbate"] = X_full["Adsorbate"].astype("category").cat.codes
y = df["RE"]

Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

rf_full = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

rf_full.fit(Xf_tr, yf_tr)

ytr_full = rf_full.predict(Xf_tr)
yte_full = rf_full.predict(Xf_te)

r2_full_train = r2_score(yf_tr, ytr_full)
r2_full_test = r2_score(yf_te, yte_full)

rmse_full_train = mean_squared_error(yf_tr, ytr_full, squared=False)
rmse_full_test = mean_squared_error(yf_te, yte_full, squared=False)

# =====================================================
# 6. -------- CONTROL RANDOM FOREST (NO CAPACITY) -----
# =====================================================
numeric_cols_control = [
    "Surface area(m2/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)"
]

features_control = numeric_cols_control + [
    "Site_Density",
    "LogTime",
    "Mixing_Index",
    "Adsorbate"
]

X_ctrl = df[features_control].copy()
X_ctrl["Adsorbate"] = X_ctrl["Adsorbate"].astype("category").cat.codes

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    X_ctrl, y, test_size=0.2, random_state=42
)

rf_ctrl = RandomForestRegressor(
    n_estimators=400,
    max_depth=8,
    random_state=42
)

rf_ctrl.fit(Xc_tr, yc_tr)

ytr_ctrl = rf_ctrl.predict(Xc_tr)
yte_ctrl = rf_ctrl.predict(Xc_te)

r2_ctrl_train = r2_score(yc_tr, ytr_ctrl)
r2_ctrl_test = r2_score(yc_te, yte_ctrl)

rmse_ctrl_train = mean_squared_error(yc_tr, ytr_ctrl, squared=False)
rmse_ctrl_test = mean_squared_error(yc_te, yte_ctrl, squared=False)

# =====================================================
# 7. Print comparison
# =====================================================
print("\n================ MODEL COMPARISON ================\n")

print("FULL RANDOM FOREST (with capacity)")
print(f"Train R² : {r2_full_train:.3f}")
print(f"Test  R² : {r2_full_test:.3f}")
print(f"Train RMSE : {rmse_full_train:.3f}")
print(f"Test  RMSE : {rmse_full_test:.3f}\n")

print("CONTROL RANDOM FOREST (capacity removed)")
print(f"Train R² : {r2_ctrl_train:.3f}")
print(f"Test  R² : {r2_ctrl_test:.3f}")
print(f"Train RMSE : {rmse_ctrl_train:.3f}")
print(f"Test  RMSE : {rmse_ctrl_test:.3f}")

# =====================================================
# 8. Plot comparison (TRAIN vs TEST)
# =====================================================
models = ["Full RF", "Control RF"]
train_scores = [r2_full_train, r2_ctrl_train]
test_scores = [r2_full_test, r2_ctrl_test]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, train_scores, width, label="Train R²")
plt.bar(x + width/2, test_scores, width, label="Test R²")

plt.xticks(x, models)
plt.ylabel("R² Score")
plt.title("Effect of Adsorption Capacity on RE (%) Prediction")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()
