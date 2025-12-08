# notebooks/kidney_train.py
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

DATA_PATH = Path("data/kidney_disease.csv")
if not DATA_PATH.exists():
    raise SystemExit("Place your kidney CSV at data/kidney.csv")

# 1. Load & quick clean
df = pd.read_csv(DATA_PATH, na_values=["?", "", " "])
print("Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Strip whitespace & lowercase column values (for object columns)
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()

# 3. Common string -> numeric conversions (yes/no, normal/abnormal)
map_yesno = {"yes":1, "no":0, "ckd":1, "notckd":0, "not ckd":0, "present":1, "absent":0}
for c in df.columns:
    if df[c].dtype == object:  # Only process string columns
        lower_vals = df[c].dropna().str.lower()
        if lower_vals.isin(map_yesno.keys()).any():
            df[c] = df[c].str.lower().map(map_yesno)

# 4. Try convert numeric-like columns to numeric
for c in df.columns:
    if df[c].dtype == object:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

# 5. Detect target column from common names
possible_targets = ["classification", "class", "ckd", "CKD", "label", "status"]
target = None
for t in possible_targets:
    if t in df.columns:
        target = t
        break

if target is None:
    raise SystemExit("Could not detect target column automatically. Edit script and set `target` variable.")

print("Using target column:", target)

# Ensure target is binary (handle NaN values first)
if df[target].dtype == object:
    df[target] = df[target].str.lower().map(map_yesno)
    
# Drop rows where target is NaN
df = df.dropna(subset=[target])
print(f"After dropping NaN targets: {df.shape}")

# 6. Drop any ID-like columns
for idcol in ("id", "Id", "ID"):
    if idcol in df.columns:
        df = df.drop(columns=[idcol])
        print("Dropped ID column:", idcol)

# 7. Map some specific known string values to numeric if present
if 'pcv' in df.columns:
    # sometimes pcv has trailing .0 or spaces; ensure numeric
    df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')

# 8. Prepare X/y
X = df.drop(columns=[target])
y = df[target].astype(int)

# 9. Column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# 10. Preprocessing
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

# 11. Model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

# 12. Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_train, y_train)

# 13. Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:,1]

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))

# 14. CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print("CV ROC AUC:", scores.mean(), "±", scores.std())

# 15. Save
Path("models").mkdir(exist_ok=True)
dump(pipe, "models/kidney_model.pkl")
print("Saved model -> models/kidney_model.pkl")
