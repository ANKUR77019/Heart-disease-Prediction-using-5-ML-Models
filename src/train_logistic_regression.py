from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "processed.cleveland.data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    RAW_DATA,
    header=None,
    names=columns,
    na_values=["?"]
)

# Binary target: 0 = original target 0; 1 = original target > 0
df["target"] = (df["target"] > 0).astype(int)

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

binary_features = ["sex", "fbs", "exang"]

categorical_features = [
    "cp", "restecg", "slope", "ca", "thal"
]

features = numeric_features + binary_features + categorical_features

X = df[features]
y = df["target"]


# --------------------------------------------------
# 3. Split the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

binary_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("binary", binary_pipeline, binary_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 5. Combine preprocessing and model training
# --------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 7. Evaluate on the held-out test set
# --------------------------------------------------

y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== LOGISTIC REGRESSION BASELINE =====")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No disease recorded", "Disease recorded"],
        zero_division=0
    )
)


# --------------------------------------------------
# 8. Save the complete fitted pipeline
# --------------------------------------------------

model_path = MODEL_DIR / "logistic_regression_pipeline.joblib"
joblib.dump(model_pipeline, model_path)

print("\nSaved fitted pipeline to:", model_path)


# --------------------------------------------------
# 9. Save baseline metrics
# --------------------------------------------------

metrics = pd.DataFrame(
    [{
        "model": "Logistic Regression",
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }]
)

metrics.to_csv(
    RESULTS_DIR / "logistic_regression_baseline.csv",
    index=False
)

print("Saved metrics to:", RESULTS_DIR / "logistic_regression_baseline.csv")
