from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Define project paths and feature names
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

BINARY_FEATURES = [
    "sex", "fbs", "exang"
]

CATEGORICAL_FEATURES = [
    "cp", "restecg", "slope", "ca", "thal"
]

FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 2. Load the original dataset
# --------------------------------------------------

df = pd.read_csv(
    RAW_DATA,
    header=None,
    names=COLUMNS,
    na_values=["?"]
)

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 3. Create the binary target
# --------------------------------------------------

df["target"] = (df["target"] > 0).astype(int)

print("\nBinary target distribution:")
print(df["target"].value_counts().sort_index())


# --------------------------------------------------
# 4. Separate input features and target
# --------------------------------------------------

X = df[FEATURES]
y = df["target"]


# --------------------------------------------------
# 5. Split before fitting preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. Define preprocessing for each feature group
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
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("binary", binary_pipeline, BINARY_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)


# --------------------------------------------------
# 7. Fit on training data and transform both sets
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# --------------------------------------------------
# 8. Save processed datasets
# --------------------------------------------------

X_train_df.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test_df.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)


# --------------------------------------------------
# 9. Report and validate results
# --------------------------------------------------

print("\nProcessed training shape:", X_train_df.shape)
print("Processed testing shape:", X_test_df.shape)

print("\nMissing values after preprocessing:")
print("Training:", int(X_train_df.isna().sum().sum()))
print("Testing:", int(X_test_df.isna().sum().sum()))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nFirst five processed training rows:")
print(X_train_df.head())

print("\nPreprocessing completed successfully.")
