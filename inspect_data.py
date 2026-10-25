import pandas as pd
from pathlib import Path

data_path = Path("data/raw/processed.cleveland.data")

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    data_path,
    header=None,
    names=columns,
    na_values=["?"]
)

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

print("\nTarget value counts:")
print(df["target"].value_counts(dropna=False).sort_index())

print("\nDuplicate rows:", df.duplicated().sum())
