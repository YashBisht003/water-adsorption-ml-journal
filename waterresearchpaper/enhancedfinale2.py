# ==========================================================
# FINAL MODEL WITH:
# - Ionic descriptors
# - Adsorbent family
# - Feature ablation study
# ==========================================================

import pandas as pd
import numpy as np
import re
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================================
# 1. LOAD DATA (XLSX)
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
# 3. ADSORBENT FAMILY (MANUAL BUT SCIENTIFIC)
# ==========================================================
def classify_adsorbent(name):
    name = str(name).lower()
    if any(k in name for k in ["carbon", "graphene", "biochar", "activated"]):
        return "Carbon"
    if any(k in name for k in ["oxide", "fe3o4", "al2o3", "tio2", "mno2"]):
        return "Oxide"
    if any(k in name for k in ["polymer", "chitosan", "resin"]):
        return "Polymer"
    if any(k in name for k in ["biomass", "cellulose", "sawdust"]):
        return "Biomass"
    return "Composite"

df["Adsorbent_Family"] = df["Adsorbent"].apply(classify_adsorbent)
df = pd.get_dummies(df, columns=["Adsorbent_Family"], drop_first=True)

# ==========================================================
# 4. PHYSICS-INFORMED FEATURES
# ==========================================================
df["Site_Density"] = (df["Dose(g/L)"] * df["Surface area(m2/g)"]) / df["Intial Concentration(ppm)"]
df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / df["Intial Concentration(ppm)"]
df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]
df["Driving_Force"] = df["Intial Concentration(ppm)"] / (df["Dose(g/L)"] + 1e-6)
df["Acidity_Strength"] = np.abs(df["Initial pH"] - 7)

# ==========================================================
# 5. IONIC DESCRIPTORS (CORE CONTRIBUTION)
# ==========================================================
IONIC_TABLE = {
    "Pb2+": (2, 1.19, 4.01),
    "Cd2+": (2, 0.95, 4.26),
    "Cu2+": (2, 0.73, 4.19),
    "Zn2+": (2, 0.74, 4.30),
    "Ni2+": (2, 0.69, 4.04)
}

df["ionic_charge"] = np.nan
df["ionic_radius"] = np.nan
df["hydrated_radius"] = np.nan

for i, ion in df["Adsorbate"].items():
    ion = str(ion).replace("(II)", "2+").strip()
    if ion in IONIC_TABLE:
        df.loc[i, ["ionic_charge", "ionic_radius", "hydrated_radius"]] = IONIC_TABLE[ion]

df[["ionic_charge", "ionic_radius", "hydrated_radius"]] = \
    df[["ionic_charge", "ionic_radius", "hydrated_radius"]].fillna(
        df[["ionic_charge", "ionic_radius", "hydrated_radius"]].median()
    )

df["ionic_potential"] = df["ionic_charge"] / df["ionic_radius"]
df["hydration_ratio"] = df["hydrated_radius"] / df["ionic_radius"]

# ==========================================================
# 6. REGIME DEFINITION
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
# 7. FEATURE SETS (ABLATION)
# ==========================================================
BASE = [
    "Surface area(m2/g)", "Intial Concentration(ppm)",
    "Contact Time (min.)", "Dose(g/L)", "RPM", "Initial pH", "T(K)"
]

PHYSICS = BASE + [
    "Site_Density", "Cap_Load_Ratio", "LogTime",
    "Mixing_Index", "Thermo_Capacity",
    "Driving_Force", "Acidity_Strength"
]

IONIC = PHYSICS + [
    "ionic_charge", "ionic_radius",
    "hydrated_radius", "ionic_potential", "hydration_ratio"
]

FAMILY = IONIC + [c for c in df.columns if c.startswith("Adsorbent_Family_")]

FEATURE_SETS = {
    "M1_Base": BASE,
    "M2_Physics": PHYSICS,
    "M3_Ionic": IONIC,
    "M4_Final": FAMILY
}

# ==========================================================
# 8. TRAIN & EVALUATE
# ==========================================================
results = []

for name, feats in FEATURE_SETS.items():
    X = df[feats].fillna(0)
    y = df["RE"]
    regimes = df["Regime"]

    X_tr, X_te, y_tr, y_te, r_tr, r_te = train_test_split(
        X, y, regimes, test_size=0.2, random_state=42, stratify=regimes
    )

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )

    model.fit(X_tr, y_tr)
    y_pred = np.clip(model.predict(X_te), 0, 100)

    mask = r_te != "Neutral_LowC0"
    r2_restricted = r2_score(y_te[mask], y_pred[mask])

    results.append({
        "Model": name,
        "Global_R2": r2_score(y_te, y_pred),
        "Restricted_R2": r2_restricted,
        "RMSE": np.sqrt(mean_squared_error(y_te, y_pred))
    })

# ==========================================================
# 9. RESULTS TABLE
# ==========================================================
results_df = pd.DataFrame(results)
print("\n================ ABLATION RESULTS ================\n")
print(results_df.to_string(index=False))
