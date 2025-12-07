# notebooks/heart_train.py

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

# 1. Load data
DATA_PATH = Path("data/heart_disease_uci.csv")
df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)
print(df.head())

# 2. Set target column name (change here if needed)
target = "num"   # <-- if your column name is different, change this

# Convert multi-class (0-4) to binary: 0 = no disease, 1 = disease present
X = df.drop(columns=[target])
y = (df[target] > 0).astype(int)  # Convert any value > 0 to 1 (disease present)

# 3. Separate numeric & categorical columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

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

# 4. Model
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)

# 6. Evaluation
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

# 7. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print("CV ROC AUC:", scores.mean(), "±", scores.std())

# 8. Save model
Path("models").mkdir(exist_ok=True)
dump(pipe, "models/heart_model.pkl")
print("Model saved to models/heart_model.pkl")
