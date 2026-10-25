import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
FIGURES_PATH = ROOT / "results" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

# Load dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    DATA_PATH, header=None, names=columns, na_values=["?"]
)
df["target"] = (df["target"] > 0).astype(int)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Define preprocessing
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
binary_features = ["sex", "fbs", "exang"]
categorical_features = ["cp", "restecg", "slope", "ca", "thal"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("binary", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ]), binary_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Train CatBoost with the same settings as before
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", CatBoostClassifier(
        iterations=300, verbose=0, random_seed=42
    ))
])

model.fit(X_train, y_train)

import joblib

MODEL_PATH = ROOT / "models" / "catboost_pipeline.joblib"
joblib.dump(model, MODEL_PATH)

print(f"Trained model saved to: {MODEL_PATH}")


# Transform test data and calculate SHAP values
fitted_preprocessor = model.named_steps["preprocessor"]
fitted_catboost = model.named_steps["model"]

X_test_transformed = fitted_preprocessor.transform(X_test)
feature_names = fitted_preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(fitted_catboost)
shap_values = explainer.shap_values(X_test_transformed)

# Calculate mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": abs(shap_values).mean(axis=0)
}).sort_values("mean_absolute_shap", ascending=False)

print("\n===== TOP 10 FEATURES BY SHAP IMPORTANCE =====")
print(importance.head(10).to_string(index=False))

# Save importance table
importance.to_csv(
    ROOT / "results" / "metrics" / "catboost_shap_importance.csv",
    index=False
)

# Plot top 10 features
top_features = importance.head(10).sort_values("mean_absolute_shap")

plt.figure(figsize=(9, 6))
plt.barh(top_features["feature"], top_features["mean_absolute_shap"])
plt.xlabel("Mean absolute SHAP value")
plt.title("CatBoost: Top 10 Feature Importances")
plt.tight_layout()

plot_path = FIGURES_PATH / "catboost_shap_importance.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\nImportance table saved.")
print(f"Plot saved to: {plot_path}")
