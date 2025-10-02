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
    
    return valid_education and valid_expertise

def get_invalid_matches(matches):
    invalid_matches = []
    for match in matches:
        if not is_valid_match(match):
            invalid_matches.append(match)
    return invalid_matches