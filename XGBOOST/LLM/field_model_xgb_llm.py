import joblib
from sentence_transformers import SentenceTransformer

# === 1. Laden van model en encoder ===
xgb_model = joblib.load("nlp_model_xgb_embeddings.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === 2. Embedding model laden ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# === 3. Interactief veldmodel ===
while True:
    opdracht = input("\nGeef een opdrachtomschrijving (of 'exit'): ")
    if opdracht.lower() == "exit":
        break

    # Embed nieuwe zin
    emb = embed_model.encode([opdracht])

    # Voorspel
    pred_num = xgb_model.predict(emb)[0]
    pred_minor = label_encoder.inverse_transform([pred_num])[0]

    # Zekerheidsscore
    proba = xgb_model.predict_proba(emb)[0]
    score_pct = round(max(proba) * 100, 2)

    print(f"➝ Deze opdracht hoort het meest bij de minor: {pred_minor} (zekerheid: {score_pct}%)")