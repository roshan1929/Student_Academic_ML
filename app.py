import streamlit as st
import pandas as pd

from career_engine import recommend_career
from skill_gap import find_skill_gap
from employability import calculate_employability_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Placement Predictor")

st.write(
    "Enter a student's academic and skill information "
    "to predict their placement status."
)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("train.csv")

df = df.drop_duplicates()


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("PlacementStatus", axis=1)

y = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})


# ==========================================
# CATEGORICAL FEATURES
# ==========================================

categorical_features = [
    "ExtracurricularActivities",
    "PlacementTraining"
]


# ==========================================
# PREPROCESSOR
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
# MODEL
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
# TRAIN MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)


# ==========================================
# INPUT SECTION
# ==========================================

st.header("Student Information")


cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.1
)


internships = st.number_input(
    "Number of Internships",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)


projects = st.number_input(
    "Number of Projects",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)


workshops = st.number_input(
    "Workshops / Certifications",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)


aptitude = st.number_input(
    "Aptitude Test Score",
    min_value=0,
    max_value=100,
    value=65,
    step=1
)


soft_skills = st.number_input(
    "Soft Skills Rating",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.1
)


extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)


training = st.selectbox(
    "Placement Training",
    ["Yes", "No"]
)


ssc = st.number_input(
    "SSC Marks",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)


hsc = st.number_input(
    "HSC Marks",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)

# ==========================================
# CAREER SKILLS
# ==========================================

st.header("Skills")

all_skills = [
    "Python",
    "SQL",
    "Excel",
    "Power BI",
    "Statistics",
    "Machine Learning",
    "Scikit-learn",
    "Pandas",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Node.js",
    "MongoDB"
]

student_skills = st.multiselect(
    "Select the skills you currently have:",
    all_skills
)
# ==========================================
# PREDICT BUTTON
# ==========================================
if st.button(
    "🔮 Predict Placement",
    use_container_width=True
):

    # ======================================
    # CREATE STUDENT DATA
    # ======================================

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


    # ======================================
    # PLACEMENT PREDICTION
    # ======================================

    prediction = model.predict(student)

    probabilities = model.predict_proba(student)

    placement_probability = probabilities[0][1]


    # ======================================
    # EMPLOYABILITY SCORE
    # ======================================

    employability_score = calculate_employability_score(
        placement_probability,
        cgpa,
        aptitude,
        soft_skills,
        internships,
        projects
    )


    # ======================================
    # CAREER RECOMMENDATION
    # ======================================

    if student_skills:

        recommended_role, career_scores = recommend_career(
            student_skills
        )

    else:

        recommended_role = "Not enough skill information"

        career_scores = {}


    # ======================================
    # SKILL GAP
    # ======================================

    if student_skills and recommended_role != "Not enough skill information":

        matched_skills, missing_skills = find_skill_gap(
            student_skills,
            recommended_role
        )

    else:

        matched_skills = []
        missing_skills = []


    # ======================================
    # SALARY RANGE
    # ======================================

    salary_df = pd.read_csv("salary_ranges.csv")

    salary_row = salary_df[
        salary_df["Role"] == recommended_role
    ]

    if not salary_row.empty:

        min_salary = salary_row.iloc[0]["MinSalary"]

        max_salary = salary_row.iloc[0]["MaxSalary"]

        salary_range = (
            f"₹{min_salary:.1f} - ₹{max_salary:.1f} LPA"
        )

    else:

        salary_range = "Not available"


    # ======================================
    # DISPLAY RESULTS
    # ======================================

    st.divider()

    st.header("📊 Your Results")


    # Placement
    if prediction[0] == 1:

        st.success(
            "🟢 LIKELY TO BE PLACED"
        )

    else:

        st.error(
            "🔴 LIKELY NOT TO BE PLACED"
        )


    st.metric(
        "Placement Probability",
        f"{placement_probability * 100:.2f}%"
    )

    st.progress(
        float(placement_probability)
    )


    # ======================================
    # EMPLOYABILITY
    # ======================================

    st.subheader("🎯 Employability Score")

    st.metric(
        "Overall Score",
        f"{employability_score}/100"
    )


    # ======================================
    # CAREER
    # ======================================

    st.subheader("💼 Recommended Career")

    st.info(
        f"Recommended Role: **{recommended_role}**"
    )


    # ======================================
    # SALARY
    # ======================================

    st.subheader("💰 Expected Salary Range")

    st.write(salary_range)


    # ======================================
    # SKILL GAP
    # ======================================

    st.subheader("🧩 Skill Gap Analysis")


    if matched_skills:

        st.write("### ✅ Skills You Have")

        st.write(
            ", ".join(matched_skills)
        )


    if missing_skills:

        st.write("### ❌ Skills to Improve")

        for skill in missing_skills:

            st.write(
                f"- {skill}"
            )

    elif student_skills:

        st.success(
            "🎉 No major skill gaps found for this role!"
        )

    else:

        st.warning(
            "Select some skills to perform skill-gap analysis."
        )