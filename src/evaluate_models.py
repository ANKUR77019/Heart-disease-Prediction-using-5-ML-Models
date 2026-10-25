import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from catboost import CatBoostClassifier

# Define the project directory
ROOT = Path(__file__).resolve().parent.parent

# Define the dataset columns
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Load the original dataset
data_path = ROOT / "data" / "raw" / "processed.cleveland.data"

df = pd.read_csv(
    data_path,
    header=None,
    names=COLUMNS,
    na_values=["?"]
)

# Convert the target into binary classes
df["target"] = (df["target"] > 0).astype(int)

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Recreate the same training/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully.")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Define feature groups
NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

BINARY_FEATURES = [
    "sex", "fbs", "exang"
]

CATEGORICAL_FEATURES = [
    "cp", "restecg", "slope", "ca", "thal"
]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Binary preprocessing
binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine the preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("binary", binary_pipeline, BINARY_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

# Define the five machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": CalibratedClassifierCV(
        estimator=SVC(),
        cv=3
    ),

    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=2000,
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        iterations=300,
        verbose=0,
        random_seed=42
    )
}

# Connect preprocessing with each model
pipelines = {
    name: Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    for name, model in models.items()
}

print("\nModels configured successfully:")
for name in pipelines:
    print(f"- {name}")

# Store the final test results
results = []

print("\n===== FINAL TEST EVALUATION =====")

for name, pipeline in pipelines.items():
    print(f"\nTraining and evaluating {name}...")

    # Train using the training data only
    pipeline.fit(X_train, y_train)

    # Predict classes and probabilities for test records
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    })

# Create a results table and rank by ROC-AUC
results_df = pd.DataFrame(results).sort_values(
    by="roc_auc",
    ascending=False
)

print("\n===== TEST SET RESULTS =====")
print(results_df.round(4).to_string(index=False))

# Save the results
output_path = ROOT / "results" / "metrics" / "test_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")


