import joblib

# === 1. Model en labelencoder laden ===
nlp_model = joblib.load("nlp_model_xgb_tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Welkom bij de opdracht-classificatie met XGBoost + TF-IDF!")
print("Typ een opdrachtomschrijving om de juiste minor te vinden (of typ 'exit' om te stoppen).")

while True:
    opdracht = input("\nGeef een opdrachtomschrijving: ")

    if opdracht.lower() == "exit":
        print("Programma afgesloten.")
        break

    # === 2. Voorspel ===
    pred_num = nlp_model.predict([opdracht])[0]               # getal (0–9)
    pred_minor = label_encoder.inverse_transform([pred_num])[0]  # terug naar naam

    # === 3. Zekerheidsscore ===
    proba = nlp_model.predict_proba([opdracht])[0]
    max_score = max(proba)
    score_pct = round(max_score * 100, 2)

    # === 4. Output ===
    print(f"➝ Deze opdracht hoort het meest bij de minor: {pred_minor} (zekerheid: {score_pct}%)")