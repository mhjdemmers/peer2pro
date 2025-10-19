education_mapping = {
    'Associate': 1,
    'Bachelor': 2, 
    'Master': 3,
    'PhD': 4
}

def is_valid_match(match):
    """
    Validate a match tuple (student, mentor, day)
    """
    student, mentor, day = match
    
    # Check education level
    student_edu_level = education_mapping[student['opleidingsniveau']]
    mentor_edu_level = education_mapping[mentor['opleidingsniveau']]
    valid_education = mentor_edu_level > student_edu_level
    
    # Check expertise match
    student_subject = student['onderwerp']
    mentor_subjects = mentor['onderwerpen']
    valid_expertise = student_subject in mentor_subjects

    # Check that both are available on the matched day
    student_availability = [d.lower() for d in student['beschikbaarheid']]
    mentor_availability = [d.lower() for d in mentor['beschikbaarheid']]
    valid_availability = day.lower() in student_availability and day.lower() in mentor_availability
    
    return valid_education and valid_expertise and valid_availability

def get_invalid_matches(matches):
    """
    Get all invalid matches from the match list
    """
    invalid_matches = []
    for match in matches:
        if not is_valid_match(match):
            invalid_matches.append(match)
    return invalid_matches

def check_mentor_capacity(matches):
    """
    Check if any mentors exceed their maximum student capacity
    """
    mentor_counts = {}
    mentor_max = {}
    
    for student, mentor, day in matches:
        mentor_name = f"{mentor['voornaam']} {mentor['achternaam']}"
        mentor_counts[mentor_name] = mentor_counts.get(mentor_name, 0) + 1
        mentor_max[mentor_name] = mentor['max_studenten']
    
    over_capacity = []
    
    for mentor_name, count in mentor_counts.items():
        if count > mentor_max[mentor_name]:
            over_capacity.append({
                'mentor': mentor_name,
                'matched': count,
                'max': mentor_max[mentor_name]
            })
    
    return over_capacity

def check_day_conflicts(matches):
    """
    Check if any person (student or mentor) has multiple matches on the same day
    """
    student_day_matches = {}
    mentor_day_matches = {}
    conflicts = []
    
    for student, mentor, day in matches:
        student_name = f"{student['voornaam']} {student['achternaam']}"
        mentor_name = f"{mentor['voornaam']} {mentor['achternaam']}"
        
        # Track student matches per day
        key = (student_name, day)
        if key in student_day_matches:
            conflicts.append({
                'type': 'student',
                'name': student_name,
                'day': day,
                'count': student_day_matches[key] + 1
            })
        student_day_matches[key] = student_day_matches.get(key, 0) + 1
        
        # Track mentor matches per day
        key = (mentor_name, day)
        if key in mentor_day_matches:
            conflicts.append({
                'type': 'mentor',
                'name': mentor_name,
                'day': day,
                'count': mentor_day_matches[key] + 1
            })
        mentor_day_matches[key] = mentor_day_matches.get(key, 0) + 1
    
    return conflicts

def check_multiple_days(matches):
    """
    Check if any student is matched on multiple days
    """
    student_days = {}
    multiple_day_students = []
    
    for student, mentor, day in matches:
        student_name = f"{student['voornaam']} {student['achternaam']}"
        
        if student_name not in student_days:
            student_days[student_name] = []
        
        if day not in student_days[student_name]:
            student_days[student_name].append(day)
        
        if len(student_days[student_name]) > 1:
            if student_name not in [s['student'] for s in multiple_day_students]:
                multiple_day_students.append({
                    'student': student_name,
                    'days': student_days[student_name]
                })
    
    return multiple_day_students