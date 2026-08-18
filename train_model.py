import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("train.csv")

print("Original dataset:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("PlacementStatus", axis=1)

y = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})


# ==========================================
# 4. DEFINE COLUMNS
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
# 5. PREPROCESSING
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
# 7. CREATE ML PIPELINE
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("LOGISTIC REGRESSION RESULTS")
print("================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NotPlaced", "Placed"]
    )
)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))