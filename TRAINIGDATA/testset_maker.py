import pandas as pd
import random

# Minor onderwerpen
subjects = [
    "Creative Digital Innovation",
    "Design Science Research",
    "Cyber Security",
    "Software Architecture",
    "Ethical Hacking",
    "Artificial Intelligence",
    "Data Science",
    "Business Process Analytics",
    "Data Visualisation",
    "Business Change and Innovation"
]

# Voor elk onderwerp een set kernwoorden / zinsdelen
templates = {
    "Creative Digital Innovation": [
        "De student ontwikkelt een {app_type} die {doel} door {technologie}.",
        "Tijdens dit project ontwerpt de student een {product} gericht op {doelgroep} met {innovatie}.",
        "Het doel is een {digitale oplossing} te creëren die {impact} heeft op {gebied}."
    ],
    "Design Science Research": [
        "De student onderzoekt {probleem} en ontwikkelt een {model} volgens de Design Science Research methodologie.",
        "Tijdens dit project wordt een {systeem} getest en gevalideerd op basis van {criteria}.",
        "De student past een ontwerpmethode toe om {uitdaging} op te lossen en documenteert de resultaten."
    ],
    # Voeg hier vergelijkbare templates toe voor de andere 8 minoren...
}

# Mogelijke variabelen voor invulling (voor diversiteit)
variables = {
    "app_type": ["mobiele app", "webapplicatie", "interactieve installatie", "dashboard"],
    "doel": ["duurzame gewoontes te stimuleren", "gebruikerservaring te verbeteren", "data inzichtelijk te maken", "beslissingen te ondersteunen"],
    "technologie": ["AR-technologie", "Python", "machine learning", "data-analyse"],
    "product": ["prototype", "tool", "platform", "simulatie"],
    "doelgroep": ["studenten", "medewerkers", "managers", "consumenten"],
    "innovatie": ["gamification", "interactief design", "AI-algoritmes", "procesautomatisering"],
    "digitale oplossing": ["app", "dashboard", "tool", "systeem"],
    "impact": ["ecologische voetafdruk te verminderen", "productiviteit te verhogen", "beslissingen te ondersteunen", "veiligheid te verbeteren"],
    "gebied": ["onderwijs", "zorg", "bedrijf", "publieke sector"],
    "probleem": ["praktijkprobleem", "complexe workflow", "informatiesysteem", "organisatieproces"],
    "model": ["prototype", "ontwerpmodel", "simulatiemodel", "analysemethode"],
    "systeem": ["workflow", "applicatie", "dashboard", "informatiesysteem"],
    "criteria": ["efficiëntie", "gebruiksvriendelijkheid", "betrouwbaarheid", "veiligheid"],
    "uitdaging": ["procesoptimalisatie", "datakwaliteit", "gebruikersinteractie", "veiligheidsrisico's"]
}

# Functie om een zin te genereren uit templates
def generate_sentence(template, vars_dict):
    sentence = template
    for var in vars_dict:
        sentence = sentence.replace("{" + var + "}", random.choice(vars_dict[var]))
    return sentence

# Genereren van 500+ zinnen per minor
records = []
for subject in subjects:
    count = 0
    while count < 500:
        if subject in templates:
            template = random.choice(templates[subject])
            sentence = generate_sentence(template, variables)
            records.append({"omschrijving": sentence, "onderwerp": subject})
            count += 1

# DataFrame maken en opslaan
df = pd.DataFrame(records)
df.to_csv("/mnt/data/dummy_opdrachten_dataset_500.csv", index=False)
print("Dataset met 500+ unieke zinnen per minor opgeslagen als 'dummy_opdrachten_dataset_500.csv'")