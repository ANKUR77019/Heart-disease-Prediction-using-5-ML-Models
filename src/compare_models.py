import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from catboost import CatBoostClassifier

from pathlib import Path
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

# Separate input features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully.")
print(f"Total records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print("\nTarget distribution in training data:")
print(y_train.value_counts().sort_index())

# Group features according to their data types
NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

BINARY_FEATURES = [
    "sex", "fbs", "exang"
]

CATEGORICAL_FEATURES = [
    "cp", "restecg", "slope", "ca", "thal"
]

# Preprocessing for numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for binary features
binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Preprocessing for categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the appropriate preprocessing to each feature group
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

    "SVM": SVC(
        probability=True,
        random_state=42
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

# Connect preprocessing and each model
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

# Set up 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Metrics used to evaluate each model
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc"
}

# Store the cross-validation results
results = []

print("\nStarting 5-fold cross-validation...")

for name, pipeline in pipelines.items():
    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    row = {"model": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std()

    results.append(row)

# Convert results into a table
results_df = pd.DataFrame(results)

# Rank models by mean ROC-AUC
results_df = results_df.sort_values(
    by="roc_auc_mean",
    ascending=False
)

print("\n===== CROSS-VALIDATION RESULTS =====")
print(results_df.round(4).to_string(index=False))

# Save the comparison results
output_path = ROOT / "results" / "metrics" / "cross_validation_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")

