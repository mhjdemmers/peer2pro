import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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
print("2: CSV uploaden (via Finder)")

keuze = input("Voer 1 of 2 in: ")

if keuze == "1":
    while True:
        opdracht = input("\nGeef een opdrachtomschrijving (of 'exit'): ")
        if opdracht.lower() == "exit":
            break
        pred_minor, score_pct = voorspellen(opdracht)
        print(f"➝ Deze opdracht hoort het meest bij de minor: {pred_minor} (zekerheid: {score_pct}%)")

elif keuze == "2":
    # Open bestandskiezer via Finder
    Tk().withdraw()  # verberg het hoofdvenster van Tkinter
    pad_csv = askopenfilename(title="Selecteer je CSV bestand", filetypes=[("CSV files", "*.csv")])
    
    if pad_csv:
        df = pd.read_csv(pad_csv)
        df['opdracht'] = df['opdracht'].fillna('').astype(str)  # veilig naar string
        resultaten = []
        
        for opdracht in df['opdracht']:
            pred_minor, score_pct = voorspellen(opdracht)
            resultaten.append({"opdracht": opdracht, "minor": pred_minor, "zekerheid": score_pct})
        
        # Opslaan
        output_csv = "voorspelde_onderwerpen.csv"
        df_result = pd.DataFrame(resultaten)
        df_result.to_csv(output_csv, index=False)
        print(f"Voorspellingen opgeslagen in '{output_csv}'")
    else:
        print("Geen CSV geselecteerd.")

else:
    print("Ongeldige keuze. Voer 1 of 2 in.")