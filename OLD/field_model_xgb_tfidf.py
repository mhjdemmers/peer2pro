import joblib

# === 1. Model laden ===
nlp_model = joblib.load("nlp_model_xgb_tfidf.pkl")

print("Welkom bij de opdracht-classificatie met XGBoost + TF-IDF!")
print("Typ een opdrachtomschrijving om de juiste minor te vinden (of typ 'exit' om te stoppen).")

while True:
    opdracht = input("\nGeef een opdrachtomschrijving: ")

    if opdracht.lower() == "exit":
        print("Programma afgesloten.")
        break

    # === 2. Voorspel ===
    minor = nlp_model.predict([opdracht])[0]

    # === 3. Zekerheidsscore ===
    proba = nlp_model.predict_proba([opdracht])[0]
    max_score = max(proba)
    score_pct = round(max_score * 100, 2)

    # === 4. Output ===
    print(f"➝ Deze opdracht hoort het meest bij de minor: {minor} (zekerheid: {score_pct}%)")