education_mapping = {
    'Associate': 1,
    'Bachelor': 2, 
    'Master': 3,
    'PhD': 4
}

def is_valid_match(match):
    student, mentor = match
    
    student_edu_level = education_mapping[student['opleidingsniveau']]
    mentor_edu_level = education_mapping[mentor['opleidingsniveau']]
    valid_education = mentor_edu_level > student_edu_level
    
    student_subject = student['onderwerp']
    mentor_subjects = mentor['onderwerpen']
    valid_expertise = student_subject in mentor_subjects

    student_availability = student['beschikbaarheid']
    mentor_availability = mentor['beschikbaarheid']
    valid_availability = any(day in mentor_availability for day in student_availability)
    
    return valid_education and valid_expertise and valid_availability

def get_invalid_matches(matches):
    invalid_matches = []
    for match in matches:
        if not is_valid_match(match):
            invalid_matches.append(match)
    return invalid_matches

def check_mentor_capacity(matches):
    mentor_counts = {}
    
    for student, mentor in matches:
        mentor_name = f"{mentor['voornaam']} {mentor['achternaam']}"
        mentor_counts[mentor_name] = mentor_counts.get(mentor_name, 0) + 1
    
    over_capacity = []
    
    for student, mentor in matches:
        mentor_name = f"{mentor['voornaam']} {mentor['achternaam']}"
        max_students = mentor['max_studenten']
        
        if mentor_counts[mentor_name] > max_students and mentor_name not in over_capacity:
            over_capacity.append(mentor_name)
    
    return over_capacity