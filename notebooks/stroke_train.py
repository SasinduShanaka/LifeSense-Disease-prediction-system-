# notebooks/stroke_train.py

import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# 1. Load dataset
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
print("Loaded stroke dataset:", df.shape)
print(df.head())

# 2. Drop ID column
if "id" in df.columns:
    df = df.drop(columns=["id"])

# 3. Target column
target = "stroke"  # 0 = no stroke, 1 = stroke

X = df.drop(columns=[target])
y = df[target]

# 4. Identify numeric & categorical features
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# 5. Preprocessing
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
])

# 6. Model (stroke dataset is imbalanced → use class_weight)
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)

# 8. Evaluation
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n--- Model Performance (Stroke) ---")
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# 9. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

print("\nCV ROC AUC:", scores.mean(), "±", scores.std())

# 10. Save model
Path("models").mkdir(exist_ok=True)
dump(pipe, "models/stroke_model.pkl")

print("\nModel saved → models/stroke_model.pkl")
