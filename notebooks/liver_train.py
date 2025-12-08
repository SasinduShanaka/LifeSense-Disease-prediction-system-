# notebooks/liver_train.py
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

DATA_PATH = Path("data/Indian Liver Patient Dataset (ILPD).csv")
if not DATA_PATH.exists():
    raise SystemExit("Place your liver CSV at data/Indian Liver Patient Dataset (ILPD).csv")

# 1. Load (treat common placeholders as NA)
df = pd.read_csv(DATA_PATH, na_values=["?", "", " "])
print("Loaded:", df.shape)
print(df.columns.tolist())

# 2. Normalize column names (strip)
df.columns = [c.strip() for c in df.columns]

# 3. Clean gender column if present
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map({'male':'Male','female':'Female'}).fillna(df['Gender'])

# 4. Auto-detect target column among common names
possible_targets = ['is_patient', 'Dataset', 'Result', 'Liver_disease', 'target', 'class']
target = None
for t in possible_targets:
    if t in df.columns:
        target = t
        break

# fallback: find binary 0/1 column
if target is None:
    for col in df.columns:
        vals = pd.Series(df[col].dropna().unique())
        if set(vals).issubset({0,1}) and col.lower() not in ('age','height','weight'):
            target = col
            print("Auto-detected target:", target)
            break

if target is None:
    raise SystemExit("Could not find target column. Edit script and set `target` variable manually.")

print("Using target column:", target)

# 5. If target uses labels like 1/2 or 1/0 where 1 = patient (check and normalize)
# Example: some ILPD versions use 1 = diseased, 2 = not diseased -> convert to 1/0
unique_vals = sorted([v for v in df[target].dropna().unique()])
if set(unique_vals) == {1, 2}:
    # convert 2 -> 0
    df[target] = df[target].map({1:1, 2:0})
    print("Mapped target 2->0, 1->1")

# 6. Drop any ID-like column
for idc in ('id', 'Id', 'ID'):
    if idc in df.columns:
        df = df.drop(columns=[idc])

# 7. Prepare X and y
X = df.drop(columns=[target])
y = df[target].astype(int)

# 8. Clean / convert columns: try numeric conversion for object types
for c in X.select_dtypes(include=['object']).columns:
    # strip whitespace
    X[c] = X[c].astype(str).str.strip()
    # try convert to numeric
    X[c] = pd.to_numeric(X[c], errors='ignore')

# 9. Identify numeric & categorical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# 10. Handle special column name 'Albumin_and_Globulin_Ratio' variants
# ensure column exists and numeric
for alt in ['Albumin_and_Globulin_Ratio', 'A/G Ratio', 'AG_Ratio', 'Albumin/Globulin_Ratio']:
    if alt in X.columns and alt not in num_cols:
        try:
            X[alt] = pd.to_numeric(X[alt], errors='coerce')
            if alt not in num_cols:
                num_cols.append(alt)
        except:
            pass

# 11. Build preprocessing pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], remainder="drop")

# 12. Model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

# 13. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe.fit(X_train, y_train)

# 14. Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n--- Performance (Liver) ---")
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# 15. CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print("CV ROC AUC:", scores.mean(), "±", scores.std())

# 16. Save model
Path("models").mkdir(exist_ok=True)
dump(pipe, "models/liver_model.pkl")
print("Saved model -> models/liver_model.pkl")
