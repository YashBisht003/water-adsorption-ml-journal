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
numeric_cols = [
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
# 3. Fix comma-decimal issue
# =====================================================
for col in numeric_cols + ["RE"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols + ["RE"])

# =====================================================
# 4. Physics-informed engineered features
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
# 5. Adsorbate physicochemical descriptors (APPROXIMATE)
# =====================================================
adsorbate_props = {
    "Cr3+":  {"charge": 3,  "radius": 0.615, "class": "metal"},
    "Pb2+":  {"charge": 2,  "radius": 1.19,  "class": "metal"},
    "Cd2+":  {"charge": 2,  "radius": 0.95,  "class": "metal"},
    "Cu2+":  {"charge": 2,  "radius": 0.73,  "class": "metal"},
    "Ni2+":  {"charge": 2,  "radius": 0.69,  "class": "metal"},
    "Zn2+":  {"charge": 2,  "radius": 0.74,  "class": "metal"},
    "As(V)": {"charge": -3, "radius": 0.46,  "class": "anion"},
    "Dye":   {"charge": -1, "radius": 1.50,  "class": "organic"}
}

df["Ads_Charge"] = df["Adsorbate"].map(
    lambda x: adsorbate_props.get(x, {}).get("charge", 0)
)

df["Ads_Radius"] = df["Adsorbate"].map(
    lambda x: adsorbate_props.get(x, {}).get("radius", 1.0)
)

df["Ads_Class"] = df["Adsorbate"].map(
    lambda x: adsorbate_props.get(x, {}).get("class", "unknown")
)

df["Ads_Charge_Density"] = df["Ads_Charge"] / df["Ads_Radius"]

# =====================================================
# 6. Feature set (FULL + PHYSICOCHEMICAL)
# =====================================================
features = [
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
    "Thermo_Capacity",
    "Ads_Charge",
    "Ads_Radius",
    "Ads_Charge_Density",
    "Ads_Class"
]

X = df[features].copy()
X["Ads_Class"] = X["Ads_Class"].astype("category").cat.codes
y = df["RE"]

# =====================================================
# 7. Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 8. Train Random Forest
# =====================================================
rf_physchem = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42
)

rf_physchem.fit(X_train, y_train)

# =====================================================
# 9. Predictions & metrics
# =====================================================
y_train_pred = rf_physchem.predict(X_train)
y_test_pred = rf_physchem.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

# =====================================================
# 10. Print results
# =====================================================
print("\n===== PHYSICOCHEMICAL RANDOM FOREST RESULTS =====\n")
print(f"Train R²  : {r2_train:.3f}")
print(f"Test  R²  : {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test  RMSE: {rmse_test:.3f}")

# =====================================================
# 11. Feature importance
# =====================================================
importances = rf_physchem.feature_importances_
fi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Physicochemical RF):\n")
print(fi_df)

# Plot importance
plt.figure(figsize=(8, 6))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Relative Importance")
plt.title("Feature Importance – Physicochemical Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
