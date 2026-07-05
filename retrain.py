# retrain.py
# ─────────────────────────────────────────────────────────────────────────────
# Retrains soil_imputer.pkl and soil_model.pkl from your real CSV dataset.
# Run this ONCE inside your chatbot venv:
#     python retrain.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Load your CSV ──────────────────────────────────────────────────────────
CSV_PATH = "soil_crop_dataset.csv"   # put the CSV in the same folder as this file

print(f"Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, encoding="latin-1")
print(f"Raw rows loaded: {len(df)}")

# ── 2. Clean column names (strip whitespace) ──────────────────────────────────
df.columns = df.columns.str.strip()

# Fix the temperature column name (has a degree symbol that may vary)
df.rename(columns=lambda c: "Temperature" if "Temp" in c else c, inplace=True)
df.rename(columns={
    "pH":           "ph",
    "N":            "nitrogen",
    "P":            "phosphorus",
    "K":            "potassium",
    "Humidity (%)": "humidity",
    "Temperature":  "temperature",
    "Best Crops":   "crops",
    "Fertilizer Recommendation": "fertilizer",
    "Soil Type (Region)":        "soil_type",
}, inplace=True)

print("Columns after renaming:", list(df.columns))

# ── 3. Drop rows with missing or empty fertilizer label ───────────────────────
df["fertilizer"] = df["fertilizer"].str.strip()
df = df[df["fertilizer"].notna() & (df["fertilizer"] != "")]
print(f"Rows after dropping empty labels: {len(df)}")

# ── 4. Fix broken label characters (? instead of - or dash) ──────────────────
# Your CSV has 'NPK 15?15?15' instead of 'NPK 15-15-15' due to encoding
df["fertilizer"] = df["fertilizer"].str.replace("?", "-", regex=False)
df["fertilizer"] = df["fertilizer"].str.replace("\x96", "-", regex=False)

print("\nFertilizer label distribution after cleaning:")
for label, count in df["fertilizer"].value_counts().items():
    print(f"  {count:3d} rows — {label}")

# ── 5. Select features in the exact order the model expects ───────────────────
FEATURE_COLS = ["ph", "nitrogen", "phosphorus", "potassium", "humidity", "temperature"]

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").values
y = df["fertilizer"].values

print(f"\nFeature matrix shape : {X.shape}")
print(f"Labels shape         : {y.shape}")
print(f"Unique fertilizers   : {len(set(y))}")

# ── 6. Also save crop labels per fertilizer (for chatbot secondary advice) ────
# Build a mapping: fertilizer → most common crops in that soil type
crop_map = {}
for fert in df["fertilizer"].unique():
    subset = df[df["fertilizer"] == fert]["crops"].dropna()
    # Collect all individual crops mentioned
    all_crops = []
    for entry in subset:
        for crop in str(entry).split(","):
            c = crop.strip()
            if c and c.lower() != "nan":
                all_crops.append(c)
    from collections import Counter
    top = [c for c, _ in Counter(all_crops).most_common(3)]
    crop_map[fert] = top

print("\nCrop suggestions per fertilizer:")
for fert, crops in crop_map.items():
    print(f"  {fert}: {crops}")

# ── 7. Build and fit the imputer ──────────────────────────────────────────────
imputer = SimpleImputer(strategy="mean")
X_clean = imputer.fit_transform(X)
print(f"\nImputer fitted. Missing values filled: {np.isnan(X).sum()} total")

# ── 8. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 9. Train Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── 10. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"\nModel accuracy on test set: {acc*100:.1f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# ── 11. Save all pkl files ────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(imputer,  "models/soil_imputer.pkl")
joblib.dump(model,    "models/soil_model.pkl")
joblib.dump(crop_map, "models/crop_map.pkl")      # bonus: crop suggestions

import sklearn
print(f"sklearn version used : {sklearn.__version__}")
print("✅ models/soil_imputer.pkl saved")
print("✅ models/soil_model.pkl   saved")
print("✅ models/crop_map.pkl     saved  (crop suggestions per fertilizer)")

# ── 12. Quick live test ───────────────────────────────────────────────────────
print("\n── Live prediction test ─────────────────────────────────────────────")
tests = [
    {"ph": 5.8, "nitrogen": 80,  "phosphorus": 12, "potassium": 90,  "humidity": 62, "temperature": 28},
    {"ph": 6.7, "nitrogen": 55,  "phosphorus": 30, "potassium": 35,  "humidity": 65, "temperature": 32},
    {"ph": 5.2, "nitrogen": 85,  "phosphorus": 45, "potassium": 50,  "humidity": 82, "temperature": 22},
]
for t in tests:
    raw   = np.array([[t["ph"], t["nitrogen"], t["phosphorus"],
                        t["potassium"], t["humidity"], t["temperature"]]])
    clean = imputer.transform(raw)
    fert  = model.predict(clean)[0]
    crops = crop_map.get(fert, [])
    print(f"  pH={t['ph']} N={t['nitrogen']} → Fertilizer: {fert}")
    print(f"            Suggested crops : {', '.join(crops)}")

print("\nAll done! Your models are ready for the chatbot.")
