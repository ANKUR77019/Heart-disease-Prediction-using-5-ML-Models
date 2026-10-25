import pandas as pd
from pathlib import Path

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    Path("data/raw/processed.cleveland.data"),
    header=None,
    names=columns,
    na_values=["?"]
)

print("===== NUMERICAL SUMMARY =====")
print(df.describe().round(2).T)

print("\n===== UNIQUE VALUES IN CATEGORICAL FEATURES =====")

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal", "target"
]

for feature in categorical_features:
    print(f"\n{feature}:")
    print(df[feature].value_counts(dropna=False).sort_index())

print("\n===== TARGET DISTRIBUTION =====")
print(df["target"].value_counts(normalize=True).sort_index().mul(100).round(2))
