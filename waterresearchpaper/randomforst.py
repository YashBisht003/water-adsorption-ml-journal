import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# 1. Load and preprocess data
# =====================================================
df = pd.read_excel("data_2023(1).xlsx")
df = df.rename(columns={"RE (%)": "RE"})

numeric_cols = [
    "Surface area(m2/g)",
    "Intial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)"
]

# Fix comma-decimal issue
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
# 2. Physics-informed features (NO capacity)
# =====================================================
df["Site_Density"] = (
    df["Dose(g/L)"] * df["Surface area(m2/g)"]
) / df["Intial Concentration(ppm)"]

df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]

# =====================================================
# 3. Feature set (capacity REMOVED)
# =====================================================
features_no_capacity = numeric_cols + [
    "Site_Density",
    "LogTime",
    "Mixing_Index",
    "Adsorbate"   # keep chemistry information
]

X = df[features_no_capacity]
y = df["RE"]

# Encode Adsorbate for Random Forest
X_rf = X.copy()
X_rf["Adsorbate"] = X_rf["Adsorbate"].astype("category").cat.codes

# =====================================================
# 4. Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42
)

# =====================================================
# 5. Train Random Forest (control model)
# =====================================================
rf_control = RandomForestRegressor(
    n_estimators=400,
    max_depth=8,
    random_state=42
)

rf_control.fit(X_train, y_train)

# =====================================================
# 6. Evaluation
# =====================================================
y_train_pred = rf_control.predict(X_train)
y_test_pred = rf_control.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

print("\nCapacity-Removed Random Forest Performance:\n")
print(f"Train R² : {r2_train:.3f}")
print(f"Test  R² : {r2_test:.3f}")
print(f"Train RMSE : {rmse_train:.3f}")
print(f"Test  RMSE : {rmse_test:.3f}")

# =====================================================
# 7. Feature importance (control model)
# =====================================================
importances = rf_control.feature_importances_

fi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Capacity Removed):\n")
print(fi_df)

# Plot
plt.figure(figsize=(8, 6))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Relative Importance")
plt.title("Random Forest Feature Importance (Capacity Removed)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
