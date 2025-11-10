import pandas as pd

# SETTINGS
# Zet hier de bestandsnamen van je CSV's
MENTOREN_FILE = "mentoren.csv"
STUDENTEN_FILE = "studenten.csv"
OUTPUT_FILE = "matches.csv"

# DATA INLADEN
mentoren_df = pd.read_csv(MENTOREN_FILE)
studenten_df = pd.read_csv(STUDENTEN_FILE)

# Zorg dat onderwerpen correct als lijst worden gelezen
mentoren_df["Onderwerpen"] = mentoren_df["Onderwerpen"].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
)

# Opleidingsniveaus ordenen
level_order = {"Associate": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

# MATCHING FUNCTIE
def find_best_mentor(student, mentoren_df):
    student_level = level_order.get(student["Opleidingsniveau"], -1)
    student_subject = student["Onderwerp"]

    best_match = None
    best_score = -1

    for _, mentor in mentoren_df.iterrows():
        mentor_level = level_order.get(mentor["Opleidingsniveau"], -1)

        # Mentor moet hoger niveau hebben
        if mentor_level <= student_level:
            continue

        # Onderwerp match
        subject_match = 1 if student_subject in mentor["Onderwerpen"] else 0

        # Score = niveauverschil + extra punten voor onderwerp
        score = (mentor_level - student_level) + (subject_match * 2)

        if score > best_score:
            best_score = score
            best_match = mentor

    if best_match is not None:
        return {
            "Student": f"{student['Voornaam']} {student['Achternaam']}",
            "Student_niveau": student["Opleidingsniveau"],
            "Onderwerp": student_subject,
            "Mentor": f"{best_match['Voornaam']} {best_match['Achternaam']}",
            "Mentor_niveau": best_match["Opleidingsniveau"],
            "Mentor_onderwerpen": best_match["Onderwerpen"],
            "Match_score": best_score,
        }
    else:
        return {
            "Student": f"{student['Voornaam']} {student['Achternaam']}",
            "Student_niveau": student["Opleidingsniveau"],
            "Onderwerp": student_subject,
            "Mentor": None,
            "Mentor_niveau": None,
            "Mentor_onderwerpen": None,
            "Match_score": 0,
        }

# MATCHES MAKEN
matches = [find_best_mentor(student, mentoren_df) for _, student in studenten_df.iterrows()]
matches_df = pd.DataFrame(matches)

# RESULTAAT OPSLAAN
matches_df.to_csv(OUTPUT_FILE, index=False)
print(f"Matching voltooid! Resultaat opgeslagen in {OUTPUT_FILE}")