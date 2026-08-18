import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Remove duplicate rows
df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)

# -----------------------------
# 1. Placement Distribution
# -----------------------------
plt.figure(figsize=(6, 4))

sns.countplot(
    x="PlacementStatus",
    data=df
)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

plt.show()


# -----------------------------
# 2. CGPA vs Placement
# -----------------------------
plt.figure(figsize=(6, 4))

sns.boxplot(
    x="PlacementStatus",
    y="CGPA",
    data=df
)

plt.title("CGPA vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")

plt.show()


# -----------------------------
# 3. Internships vs Placement
# -----------------------------
plt.figure(figsize=(6, 4))

sns.countplot(
    x="Internships",
    hue="PlacementStatus",
    data=df
)

plt.title("Internships vs Placement Status")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")

plt.show()


# -----------------------------
# 4. Aptitude Score vs Placement
# -----------------------------
plt.figure(figsize=(6, 4))

sns.boxplot(
    x="PlacementStatus",
    y="AptitudeTestScore",
    data=df
)

plt.title("Aptitude Score vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Aptitude Score")

plt.show()


print("\n--- CATEGORICAL VALUES ---")

print("\nExtracurricularActivities:")
print(df["ExtracurricularActivities"].value_counts())

print("\nPlacementTraining:")
print(df["PlacementTraining"].value_counts())

print("\nPlacementStatus:")
print(df["PlacementStatus"].value_counts())