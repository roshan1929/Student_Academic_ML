import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 5. MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. TRAIN
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. GET FEATURE NAMES
# ==========================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()


# ==========================================
# 9. GET COEFFICIENTS
# ==========================================

coefficients = model.named_steps[
    "classifier"
].coef_[0]


coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})


# ==========================================
# 10. SORT
# ==========================================

coefficient_df["AbsoluteCoefficient"] = (
    coefficient_df["Coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    by="AbsoluteCoefficient",
    ascending=False
)


# ==========================================
# 11. DISPLAY
# ==========================================

print("\n================================")
print("LOGISTIC REGRESSION COEFFICIENTS")
print("================================")

print(
    coefficient_df[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)


# ==========================================
# 12. PLOT
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=coefficient_df,
    x="Coefficient",
    y="Feature"
)

plt.title(
    "Feature Coefficients - Logistic Regression"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.tight_layout()

plt.show()