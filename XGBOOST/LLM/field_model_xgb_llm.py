import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# === 1. Laden van model en encoder ===
xgb_model = joblib.load("nlp_model_xgb_embeddings.pkl")
label_encoder = joblib.load("label_encoder_xgb_llm.pkl")

# === 2. Embedding model laden ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# === 3. Functie om een opdracht te voorspellen ===
def voorspellen(opdracht):
    if not isinstance(opdracht, str):
        opdracht = str(opdracht)
    emb = embed_model.encode([opdracht])
    pred_num = xgb_model.predict(emb)[0]
    pred_minor = label_encoder.inverse_transform([pred_num])[0]
    proba = xgb_model.predict_proba(emb)[0]
    score_pct = round(max(proba) * 100, 2)
    return pred_minor, score_pct

# === 4. Kies inputmethode ===
print("Kies inputmethode:")
print("1: Interactief (typ een opdracht)")
print("2: CSV uploaden (via pad invoer)")

keuze = input("Voer 1 of 2 in: ")

if keuze == "1":
    while True:
        opdracht = input("\nGeef een opdrachtomschrijving (of 'exit'): ")
        if opdracht.lower() == "exit":
            break
        pred_minor, score_pct = voorspellen(opdracht)
        print(f"➝ Deze opdracht hoort het meest bij de minor: {pred_minor} (zekerheid: {score_pct}%)")

elif keuze == "2":
    pad_csv = input("Voer het volledige pad in naar je CSV-bestand (bijv. ./input.csv): ").strip()
    
    try:
        df = pd.read_csv(pad_csv)
    except FileNotFoundError:
        print(f"❌ Bestand niet gevonden: {pad_csv}")
    except Exception as e:
        print(f"❌ Fout bij het lezen van het bestand: {e}")
    else:
        if 'omschrijving' not in df.columns:
            print("❌ Kolom 'omschrijving' niet gevonden in het CSV-bestand.")
        else:
            df['omschrijving'] = df['omschrijving'].fillna('').astype(str)
            resultaten = []

            for opdracht in df['omschrijving']:
                pred_minor, score_pct = voorspellen(opdracht)
                resultaten.append({"omschrijving": opdracht, "minor": pred_minor, "zekerheid": score_pct})

            output_csv = "voorspelling_output_xgb_llm.csv"
            df_result = pd.DataFrame(resultaten)
            df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"✅ Voorspellingen opgeslagen in '{output_csv}'")

else:
    print("Ongeldige keuze. Voer 1 of 2 in.")
