import pandas as pd


def recommend_career(student_skills):

    df = pd.read_csv("career_skills.csv")

    roles = df["Role"].unique()

    scores = {}

    for role in roles:

        required_skills = set(
            df[df["Role"] == role]["Skill"]
        )

        student_skill_set = set(student_skills)

        matched_skills = (
            required_skills.intersection(student_skill_set)
        )

        scores[role] = len(matched_skills)


    recommended_role = max(
        scores,
        key=scores.get
    )

    return recommended_role, scores

if __name__ == "__main__":

    student_skills = [
        "Python",
        "SQL",
        "Excel",
        "Power BI"
    ]

    role, scores = recommend_career(student_skills)

    print("Recommended Career:", role)
    print("\nCareer Scores:")

    for r, score in scores.items():
        print(r, ":", score)