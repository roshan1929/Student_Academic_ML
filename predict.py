import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. LOAD DATASET
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


# ==========================================
# 4. PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
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
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. STUDENT INPUT
# ==========================================

print("\n======================================")
print(" STUDENT PLACEMENT PREDICTOR")
print("======================================")

cgpa = float(input("Enter CGPA: "))

internships = int(
    input("Enter number of internships: ")
)

projects = int(
    input("Enter number of projects: ")
)

workshops = int(
    input("Enter number of workshops/certifications: ")
)

aptitude = int(
    input("Enter aptitude test score: ")
)

soft_skills = float(
    input("Enter soft skills rating: ")
)

extracurricular = input(
    "Extracurricular activities? (Yes/No): "
).strip().title()

training = input(
    "Placement training completed? (Yes/No): "
).strip().title()

ssc = int(
    input("Enter SSC marks: ")
)

hsc = int(
    input("Enter HSC marks: ")
)


# ==========================================
# 9. CREATE STUDENT DATAFRAME
# ==========================================

student = pd.DataFrame([{
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Workshops/Certifications": workshops,
    "AptitudeTestScore": aptitude,
    "SoftSkillsRating": soft_skills,
    "ExtracurricularActivities": extracurricular,
    "PlacementTraining": training,
    "SSC_Marks": ssc,
    "HSC_Marks": hsc
}])


# ==========================================
# 10. PREDICTION
# ==========================================

prediction = model.predict(student)

probabilities = model.predict_proba(student)

placement_probability = probabilities[0][1]


# ==========================================
# 11. DISPLAY RESULT
# ==========================================

print("\n======================================")
print(" RESULT")
print("======================================")

if prediction[0] == 1:
    print("Prediction: LIKELY TO BE PLACED")
else:
    print("Prediction: LIKELY NOT TO BE PLACED")


print(
    f"Placement Probability: "
    f"{placement_probability * 100:.2f}%"
)

print("======================================")