import pandas as pd
import matplotlib.pyplot as plt
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

# Figure 1: Age distribution
plt.figure()
plt.hist(df["age"], bins=15, edgecolor="black")
plt.xlabel("Age (years)")
plt.ylabel("Number of records")
plt.title("Age Distribution in the Cleveland Dataset")
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150)
plt.close()

# Figure 2: Cholesterol distribution
plt.figure()
plt.boxplot(df["chol"].dropna())
plt.ylabel("Serum cholesterol")
plt.title("Cholesterol Distribution")
plt.tight_layout()
plt.savefig("cholesterol_boxplot.png", dpi=150)
plt.close()

print("Visualizations saved:")
print(" - age_distribution.png")
print(" - cholesterol_boxplot.png")
