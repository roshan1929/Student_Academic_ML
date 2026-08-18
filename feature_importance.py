import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("train.csv")

df = df.drop_duplicates()


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("PlacementStatus", axis=1)

y = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})


# ==========================================
# 3. COLUMNS
# ==========================================

categorical_features = [
    "ExtracurricularActivities",
    "PlacementTraining"
]

numerical_features = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "SSC_Marks",
    "HSC_Marks"
]


# ==========================================
# 4. PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 5. RANDOM FOREST
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 6. PIPELINE
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. TRAIN
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 9. GET FEATURE NAMES
# ==========================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()


# ==========================================
# 10. GET FEATURE IMPORTANCE
# ==========================================

importances = model.named_steps[
    "classifier"
].feature_importances_


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 11. PRINT RESULTS
# ==========================================

print("\n================================")
print("FEATURE IMPORTANCE")
print("================================")

print(importance_df)


# ==========================================
# 12. PLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance - Random Forest")

plt.tight_layout()

plt.show()