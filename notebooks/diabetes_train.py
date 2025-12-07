# notebooks/diabetes_train.py

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# 1. Load data
DATA_PATH = Path("data/diabetes.csv")
df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)

# 2. Replace medically invalid zeros with NaN (Pima dataset)
replace_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in replace_zero_cols:
    if col in df.columns:
        df.loc[df[col] == 0, col] = np.nan

# 3. Split into X and y
target = "Outcome"
X = df.drop(columns=[target])
y = df[target]

# 4. Preprocessing
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols)
], remainder="drop")

# 5. Model — Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)

# 7. Evaluation
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# 8. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print("CV ROC AUC:", scores.mean(), "±", scores.std())

# 9. Save model
Path("models").mkdir(exist_ok=True)
dump(pipe, "models/diabetes_model.pkl")
print("Model saved to models/diabetes_model.pkl")
