import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

# =====================================================
# 1. Load data (file in SAME folder as this script)
# =====================================================
df = pd.read_excel("data_2023(1).xlsx")

# Rename target column
df = df.rename(columns={"RE (%)": "RE"})

# =====================================================
# 2. Columns definition
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

target_col = "RE"

# =====================================================
# 3. FIX: comma-as-decimal issue (CRITICAL)
# =====================================================
for col in numeric_cols + [target_col]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

# Convert to numeric
for col in numeric_cols + [target_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=numeric_cols + [target_col])

# =====================================================
# 4. Physics-informed feature engineering
# =====================================================
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

# =====================================================
# 5. Feature matrix & target
# =====================================================
features = numeric_cols + [
    "Site_Density",
    "Cap_Load_Ratio",
    "LogTime",
    "Mixing_Index",
    "Thermo_Capacity",
    "Adsorbate"  # categorical (IMPORTANT)
]

X = df[features]
y = df[target_col]

# Categorical feature index for CatBoost
cat_features = [X.columns.get_loc("Adsorbate")]

# =====================================================
# 6. Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 7. Linear Regression (baseline)
# =====================================================
lin_features = [f for f in features if f != "Adsorbate"]

lin_model = LinearRegression()
lin_model.fit(X_train[lin_features], y_train)
y_pred_lin = lin_model.predict(X_test[lin_features])

# =====================================================
# 8. Random Forest (baseline)
# =====================================================
X_rf = X.copy()
X_rf["Adsorbate"] = X_rf["Adsorbate"].astype("category").cat.codes

Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)
rf_model.fit(Xrf_train, yrf_train)
y_pred_rf = rf_model.predict(Xrf_test)

# ============================
# Random Forest Feature Importance
# ============================

import pandas as pd
import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
feature_names = Xrf_train.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:\n")
print(fi_df)

# Plot
plt.figure(figsize=(8, 6))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Relative Importance")
plt.title("Random Forest Feature Importance for RE (%) Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# =====================================================
# 9. CatBoost (FINAL, physics-informed model)
# =====================================================
cat_model = CatBoostRegressor(
    iterations=350,
    depth=6,
    learning_rate=0.08,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    early_stopping_rounds=40
)

y_pred_cat = cat_model.predict(X_test)

# =====================================================
# 10. Evaluation
# =====================================================
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return round(r2, 3), round(rmse, 3)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "CatBoost (Physics-informed)"
    ],
    "R2": [
        evaluate(y_test, y_pred_lin)[0],
        evaluate(yrf_test, y_pred_rf)[0],
        evaluate(y_test, y_pred_cat)[0]
    ],
    "RMSE": [
        evaluate(y_test, y_pred_lin)[1],
        evaluate(yrf_test, y_pred_rf)[1],
        evaluate(y_test, y_pred_cat)[1]
    ]
})

print("\nModel Performance Comparison:\n")
print(results)
