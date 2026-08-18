import pandas as pd


def find_skill_gap(student_skills, recommended_role):

    df = pd.read_csv("career_skills.csv")

    required_skills = set(
        df[df["Role"] == recommended_role]["Skill"]
    )

    student_skill_set = set(student_skills)

    matched_skills = required_skills.intersection(
        student_skill_set
    )

    missing_skills = required_skills.difference(
        student_skill_set
    )

    return list(matched_skills), list(missing_skills)


if __name__ == "__main__":

    student_skills = [
        "Python",
        "SQL",
        "Excel"
    ]

    role = "Data Analyst"

    matched, missing = find_skill_gap(
        student_skills,
        role
    )

    print("Recommended Role:", role)

    print("\nSkills Student Has:")
    print(matched)

    print("\nMissing Skills:")
    print(missing)