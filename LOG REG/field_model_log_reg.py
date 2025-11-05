import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# === 1. Laden van model en encoder ===
model = joblib.load("nlp_model_logreg_embeddings.pkl")
label_encoder = joblib.load("label_encoder_log_reg.pkl")

# === 2. Embedding model laden ===
embed_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("✅ Model succesvol geladen!")
print("Kies een modus:")
print("1 → Handmatige invoer")
print("2 → CSV-bestand uploaden\n")

mode = input("Maak een keuze (1 of 2): ").strip()

# === 3. Handmatige modus ===
if mode == "1":
    while True:
        opdracht = input("\nGeef een opdrachtomschrijving (of 'exit' om te stoppen): ")
        if opdracht.lower() == "exit":
            break

        # Embed nieuwe opdracht
        emb = embed_model.encode([opdracht])

        # Voorspel categorie
        pred_num = model.predict(emb)[0]
        pred_minor = label_encoder.inverse_transform([pred_num])[0]

        # Zekerheid berekenen
        proba = model.predict_proba(emb)[0]
        score_pct = round(max(proba) * 100, 2)

        print(f"➝ Deze opdracht hoort het meest bij: {pred_minor} (zekerheid: {score_pct}%)")

# === 4. CSV-modus ===
elif mode == "2":
    csv_path = input("\nGeef het pad naar het CSV-bestand (met kolom 'omschrijving'): ").strip()

    try:
        df = pd.read_csv(csv_path)
        if "omschrijving" not in df.columns:
            raise ValueError("CSV mist verplichte kolom 'omschrijving'.")

        print(f"✅ {len(df)} opdrachten gevonden. Genereren van embeddings...")
        emb = embed_model.encode(df["omschrijving"].astype(str).tolist(), show_progress_bar=True)

        # Voorspel categorieën
        preds = model.predict(emb)
        pred_labels = label_encoder.inverse_transform(preds)

        # Zekerheidsscores
        proba = model.predict_proba(emb)
        scores = proba.max(axis=1)

        # Voeg resultaten toe aan dataframe
        df["voorspeld_onderwerp"] = pred_labels
        df["zekerheid_%"] = (scores * 100).round(2)

        # Opslaan
        output_path = "voorspellingen_output_log_reg.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Resultaten opgeslagen in: {output_path}")

        # Voorbeeldoutput
        print("\n📊 Voorbeeldresultaten:")
        print(df[["omschrijving", "voorspeld_onderwerp", "zekerheid_%"]].head(10))

    except Exception as e:
        print(f"❌ Fout bij verwerken van CSV: {e}")

else:
    print("❌ Ongeldige keuze. Kies 1 of 2 aub.")