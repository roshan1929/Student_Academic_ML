import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("train.csv")

print("Original dataset shape:", df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("PlacementStatus", axis=1)

y = df["PlacementStatus"]


# Convert target into numerical values
# NotPlaced = 0
# Placed = 1

y = y.map({
    "NotPlaced": 0,
    "Placed": 1
})


print("\nTarget values:")
print(y.value_counts())


# ==========================================
# 4. IDENTIFY CATEGORICAL COLUMNS
# ==========================================

categorical_features = [
    "ExtracurricularActivities",
    "PlacementTraining"
]


# ==========================================
# 5. IDENTIFY NUMERICAL COLUMNS
# ==========================================

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
# 6. ENCODE CATEGORICAL FEATURES
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
# 8. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ==========================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)

print("\nPreprocessing completed successfully!")