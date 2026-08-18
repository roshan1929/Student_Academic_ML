def calculate_employability_score(
    placement_probability,
    cgpa,
    aptitude,
    soft_skills,
    internships,
    projects
):

    score = (
        placement_probability * 50
        + (cgpa / 10) * 15
        + (aptitude / 100) * 15
        + (soft_skills / 10) * 10
        + min(internships / 2, 1) * 5
        + min(projects / 4, 1) * 5
    )

    return round(min(score, 100), 2)


if __name__ == "__main__":

    score = calculate_employability_score(
        placement_probability=0.85,
        cgpa=8.2,
        aptitude=78,
        soft_skills=8,
        internships=1,
        projects=4
    )

    print("Employability Score:", score, "/ 100")