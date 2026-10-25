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

# Calculate the IQR thresholds for cholesterol
q1 = df["chol"].quantile(0.25)
q3 = df["chol"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print("===== CHOLESTEROL OUTLIER THRESHOLDS =====")
print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Identify potential outliers without removing them
outliers = df[
    (df["chol"] < lower_bound) |
    (df["chol"] > upper_bound)
]

print("\n===== POTENTIAL CHOLESTEROL OUTLIERS =====")
print(outliers[["age", "sex", "chol", "target"]].to_string())

print("\nNumber of potential outliers:", len(outliers))

# Examine cholesterol by the original target categories
print("\n===== CHOLESTEROL BY ORIGINAL TARGET =====")
print(
    df.groupby("target")["chol"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)
