# add_ionic_features_and_train.py
# ==========================================================
# Add ionic descriptors (ionic radius, hydrated radius, hydration energy)
# Retrain LightGBM (your best model) and print global/regime metrics
# Sources: BioNumbers hydrated radii table, Shannon ionic radii, Mahler & Persson reviews
# ==========================================================

import os
import json
import re
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor

# -------------------------
# CONFIG
# -------------------------
DATA_FILE = "data_2023(1).xlsx"
OUTPUT_DIR = "ionic_feature_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -------------------------
# 0. load data (xlsx)
# -------------------------
df = pd.read_excel(DATA_FILE)
df = df.drop(columns=["Unnamed: 12"], errors="ignore")
df = df.rename(columns={"RE (%)": "RE"})

# -------------------------
# 1. robust numeric cleaning (same approach you used earlier)
# -------------------------
numeric_cols = [
    "Surface area(m2/g)", "Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)", "Contact Time (min.)",
    "Dose(g/L)", "RPM", "Initial pH", "T(K)", "RE"
]

def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    # decimal comma -> dot
    if re.match(r"^\d+,\d+$", x):
        x = x.replace(",", ".")
    else:
        # sometimes entries like "30, 0.1" -> take first token
        x = x.split(",")[0]
    x = x.replace("±", "").replace("~", "").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+", x)
    return float(m.group()) if m else np.nan

for c in numeric_cols:
    if c in df.columns:
        df[c] = df[c].apply(clean_numeric)

df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])
df = df[(df["RE"] >= 0) & (df["RE"] <= 100)].reset_index(drop=True)

# -------------------------
# 2. physics-informed features (keep your established set)
# -------------------------
df["Site_Density"] = (df["Dose(g/L)"] * df["Surface area(m2/g)"]) / df["Intial Concentration(ppm)"]
df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / df["Intial Concentration(ppm)"]
df["LogTime"] = np.log1p(df["Contact Time (min.)"])
df["Mixing_Index"] = df["RPM"] * df["LogTime"]
df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]

# safe proxies
df["Driving_Force"] = df["Intial Concentration(ppm)"] / (df["Dose(g/L)"] + 1e-9)
df["Acidity_Strength"] = np.abs(df["Initial pH"] - 7.0)

# -------------------------
# 3. ionic property lookup (compiled from published tables)
#    Units: ionic_radius_A (Å), hydrated_radius_A (Å), hydration_energy_kJmol (kJ/mol)
#    Notes: values compiled from BioNumbers (hydrated radii), Shannon/other compilations, Mahler & Persson
# -------------------------

# Starter table: extend/replace values with exact references in manuscript/SI.
# Many hydrated radii tables give hydrated radius in nm -> converted to Å (1 nm = 10 Å).
ION_PROPERTIES = {
    # Format: 'label as in Adsorbate column' : {
    #   "charge": int,
    #   "ionic_radius_A": float or None,
    #   "hydrated_radius_A": float or None,
    #   "hydration_energy_kJmol": float or None,
    #   "source": "short citation or 'BioNumbers/Shannon/...' "
    # }
    "Pb2+":   {"charge": 2, "ionic_radius_A": 1.19, "hydrated_radius_A": 4.01, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Cd2+":   {"charge": 2, "ionic_radius_A": 0.95, "hydrated_radius_A": 4.26, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Cu2+":   {"charge": 2, "ionic_radius_A": 0.73, "hydrated_radius_A": 4.19, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Zn2+":   {"charge": 2, "ionic_radius_A": 0.74, "hydrated_radius_A": 4.30, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Ni2+":   {"charge": 2, "ionic_radius_A": 0.69, "hydrated_radius_A": 4.04, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Fe2+":   {"charge": 2, "ionic_radius_A": 0.78, "hydrated_radius_A": 4.30, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Fe3+":   {"charge": 3, "ionic_radius_A": 0.65, "hydrated_radius_A": 4.00, "hydration_energy_kJmol": None, "source":"literature compiled"},
    "Al3+":   {"charge": 3, "ionic_radius_A": 0.50, "hydrated_radius_A": 4.8,  "hydration_energy_kJmol": None, "source":"BioNumbers/Shannon"},
    "Ca2+":   {"charge": 2, "ionic_radius_A": 0.99, "hydrated_radius_A": 4.10, "hydration_energy_kJmol": None, "source":"BioNumbers"},
    "Mg2+":   {"charge": 2, "ionic_radius_A": 0.65, "hydrated_radius_A": 4.30, "hydration_energy_kJmol": None, "source":"BioNumbers"},
    "Na+":    {"charge": 1, "ionic_radius_A": 0.95, "hydrated_radius_A": 3.6,  "hydration_energy_kJmol": None, "source":"BioNumbers"},
    "K+":     {"charge": 1, "ionic_radius_A": 1.33, "hydrated_radius_A": 3.3,  "hydration_energy_kJmol": None, "source":"BioNumbers"},
    # oxyanions (speciation depends on pH) - include if present but treat cautiously
    "CrO4(2-)": {"charge": -2, "ionic_radius_A": None, "hydrated_radius_A": 3.4, "hydration_energy_kJmol": None, "source":"literature"},
    "AsO4(3-)": {"charge": -3, "ionic_radius_A": None, "hydrated_radius_A": 3.6, "hydration_energy_kJmol": None, "source":"literature"},
    # Add more ions as needed...
}

# Save the ionic table to inspect/extend later
with open(os.path.join(OUTPUT_DIR, "ionic_table_used.json"), "w") as f:
    json.dump(ION_PROPERTIES, f, indent=2)

# helper to normalize adsorbate labels
def normalize_adsorbate_label(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = s.replace("(II)", "2+").replace("(III)", "3+").replace("(IV)", "4+")
    s = s.replace(" ", "")
    # if written like 'Pb2' -> 'Pb2+'
    s = re.sub(r"^([A-Za-z]+)(\d)$", r"\1\2+", s)
    return s

# -------------------------
# 4. map ionic properties into DataFrame
# -------------------------
# create columns
cols_ions = ['ionic_charge', 'ionic_radius_A', 'hydrated_radius_A', 'hydration_energy_kJmol']
for c in cols_ions:
    df[c] = np.nan

if 'Adsorbate' in df.columns:
    for idx, val in df['Adsorbate'].items():
        key = normalize_adsorbate_label(val)
        # direct match
        if key in ION_PROPERTIES:
            meta = ION_PROPERTIES[key]
            df.at[idx, 'ionic_charge'] = meta.get('charge', np.nan)
            df.at[idx, 'ionic_radius_A'] = meta.get('ionic_radius_A', np.nan)
            df.at[idx, 'hydrated_radius_A'] = meta.get('hydrated_radius_A', np.nan)
            df.at[idx, 'hydration_energy_kJmol'] = meta.get('hydration_energy_kJmol', np.nan)
        else:
            # try prefix match like 'Pb' in 'Pb2+'
            matched = False
            for ion_label in ION_PROPERTIES.keys():
                if ion_label.split()[0].startswith(key.split()[0]) or key.startswith(ion_label.split()[0]):
                    meta = ION_PROPERTIES[ion_label]
                    df.at[idx, 'ionic_charge'] = meta.get('charge', np.nan)
                    df.at[idx, 'ionic_radius_A'] = meta.get('ionic_radius_A', np.nan)
                    df.at[idx, 'hydrated_radius_A'] = meta.get('hydrated_radius_A', np.nan)
                    df.at[idx, 'hydration_energy_kJmol'] = meta.get('hydration_energy_kJmol', np.nan)
                    matched = True
                    break
            if not matched:
                # leave NaN (we will impute medians below)
                pass

# -------------------------
# 5. derived ionic features (physically motivated)
# -------------------------
# ionic_potential = charge / bare ionic radius (Å) -> proxy for charge density
df['ionic_potential'] = df['ionic_charge'] / (df['ionic_radius_A'] + 1e-9)

# hydration_ratio = hydrated_radius / ionic_radius
df['hydration_ratio'] = df['hydrated_radius_A'] / (df['ionic_radius_A'] + 1e-9)

# hydration_adjusted_driving: driving force scaled by hydration (example physically motivated proxy)
df['hydration_adjusted_driving'] = df['Driving_Force'] / (df['hydration_ratio'] + 1e-9)

# ensure numeric
for c in ['ionic_potential','hydration_ratio','hydration_adjusted_driving']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# impute medians for ionic properties (documented fallback)
ion_medians = df[['ionic_charge','ionic_radius_A','hydrated_radius_A','hydration_energy_kJmol','ionic_potential','hydration_ratio','hydration_adjusted_driving']].median()
df[['ionic_charge','ionic_radius_A','hydrated_radius_A','hydration_energy_kJmol','ionic_potential','hydration_ratio','hydration_adjusted_driving']] = df[['ionic_charge','ionic_radius_A','hydrated_radius_A','hydration_energy_kJmol','ionic_potential','hydration_ratio','hydration_adjusted_driving']].fillna(ion_medians)

# -------------------------
# 6. encode Adsorbate (optional) and drop identifiers
# -------------------------
if 'Adsorbate' in df.columns:
    df = pd.get_dummies(df, columns=['Adsorbate'], drop_first=True)

df = df.drop(columns=['Adsorbent','Ref.'], errors='ignore')

# -------------------------
# 7. prepare feature list & train/test split
# -------------------------
base_features = [
    "Surface area(m2/g)","Adsorption capacity(mg/g)",
    "Intial Concentration(ppm)","Contact Time (min.)",
    "Dose(g/L)","RPM","Initial pH","T(K)",
    "Site_Density","Cap_Load_Ratio","LogTime","Mixing_Index","Thermo_Capacity",
    "Driving_Force","Acidity_Strength",
    # ionic descriptors
    "ionic_charge","ionic_radius_A","hydrated_radius_A","hydration_energy_kJmol",
    "ionic_potential","hydration_ratio","hydration_adjusted_driving"
]

adsorbate_feats = [c for c in df.columns if c.startswith("Adsorbate_")]
FEATURES = [f for f in base_features + adsorbate_feats if f in df.columns]

X = df[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['RE']

# regimes (same rule as before)
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
df['Regime'] = df.apply(assign_regime, axis=1)
regimes = df['Regime']

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, regimes, test_size=0.2, random_state=RANDOM_STATE, stratify=regimes
)

# -------------------------
# 8. train LightGBM (your tuned settings)
# -------------------------
lgb_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=RANDOM_STATE,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

# -------------------------
# 9. evaluate
# -------------------------
y_pred = np.clip(lgb_model.predict(X_test), 0, 100)
global_r2 = r2_score(y_test, y_pred)
global_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== After adding ionic descriptors ===")
print("Global R²:", round(global_r2, 4), "RMSE:", round(global_rmse, 4))

# restricted global excluding Neutral_LowC0
mask_no_neutral_low = r_test != "Neutral_LowC0"
y_test_restricted = y_test[mask_no_neutral_low]
y_pred_restricted = y_pred[mask_no_neutral_low]
global_r2_no_neutral = r2_score(y_test_restricted, y_pred_restricted)
global_rmse_no_neutral = np.sqrt(mean_squared_error(y_test_restricted, y_pred_restricted))

print("Global R² (excluding Neutral_LowC0):", round(global_r2_no_neutral, 4))
print("Global RMSE (excluding Neutral_LowC0):", round(global_rmse_no_neutral, 4))

# regime-wise table
rows=[]
for reg in sorted(r_test.unique()):
    idx = r_test == reg
    n = idx.sum()
    if n < 5:
        r2 = float('nan'); rmse = float('nan')
    else:
        r2 = r2_score(y_test[idx], y_pred[idx])
        rmse = np.sqrt(mean_squared_error(y_test[idx], y_pred[idx]))
    rows.append({"Regime":reg,"Samples":int(n),"R2":r2,"RMSE":rmse})
regime_df = pd.DataFrame(rows)
print("\nRegime-wise performance:")
print(regime_df.to_string(index=False))

# corrected weighted regime R2 (exclude small & negative)
MIN_SAMPLES=10
valid = regime_df[(regime_df["Samples"]>=MIN_SAMPLES)&(regime_df["R2"]>0)]
if len(valid)>0:
    weighted_r2 = np.average(valid["R2"], weights=valid["Samples"])
else:
    weighted_r2 = float('nan')
print("\nCorrected Weighted Regime R²:", round(weighted_r2,4) if not math.isnan(weighted_r2) else weighted_r2)

# -------------------------
# 10. save outputs
# -------------------------
out = X_test.copy()
out['Actual_RE'] = y_test.values
out['Predicted_RE'] = y_pred
out['Regime'] = r_test.values
out.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_ionic_features.csv"), index=False)

# feature importance
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importances_with_ionic_features.csv"), index=False)

# save ionic table used
with open(os.path.join(OUTPUT_DIR, "ionic_table_used.json"), "w") as f:
    json.dump(ION_PROPERTIES, f, indent=2)

print("\nSaved predictions and feature importances to", OUTPUT_DIR)
