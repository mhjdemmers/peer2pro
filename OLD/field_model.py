import joblib
import numpy as np

# 1. Laad het opgeslagen model
nlp_model = joblib.load("nlp_model.pkl")

print("Welkom bij de opdracht-classificatie!")
print("Typ een opdrachtomschrijving om de juiste minor te vinden (of typ 'exit' om te stoppen).")

while True:
    # 2. Vraag input in de terminal
    opdracht = input("\nGeef een opdrachtomschrijving: ")

    # Stoppen
    if opdracht.lower() == "exit":
        print("Programma afgesloten.")
        break

    # 3. Voorspel met NLP-model
    minor = nlp_model.predict([opdracht])[0]

    # 4. Bereken de top 3 voorspellingen en zekerheid
    try:
        probabilities = nlp_model.predict_proba([opdracht])[0]
        labels = nlp_model.classes_

        # Top 3 indices
        top3_idx = np.argsort(probabilities)[::-1][:3]
        print("➝ Top 3 voorspellingen:")
        for i in top3_idx:
            label = labels[i]
            confidence_percent = round(probabilities[i]*100, 2)
            print(f"   {label} ({confidence_percent}%)")
    except AttributeError:
        # Als predict_proba niet beschikbaar is
        print(f"➝ Voorspelling: {minor} (zekerheid niet beschikbaar)")