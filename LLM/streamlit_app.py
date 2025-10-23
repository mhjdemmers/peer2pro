import streamlit as st
import joblib

# === 1. Model laden ===
nlp_model = joblib.load("nlp_model.pkl")

# === 2. UI ===
st.set_page_config(page_title="Opdracht Classificatie", page_icon="🎓", layout="centered")

st.title("🎓 Opdracht Classificatie App")
st.write("Typ hieronder een opdrachtomschrijving en zie welke **minor** erbij past.")

# Input tekstveld
opdracht = st.text_area("Opdrachtomschrijving invoeren:")

if st.button("Classificeer"):
    if opdracht.strip():
        # === 3. Voorspellen ===
        minor = nlp_model.predict([opdracht])[0]
        proba = nlp_model.predict_proba([opdracht])[0]
        score_pct = round(max(proba) * 100, 2)

        # === 4. Output ===
        st.success(f"Deze opdracht hoort het meest bij de minor: **{minor}**")
        st.info(f"Zekerheid van model: **{score_pct}%**")
    else:
        st.warning("Voer eerst een opdrachtomschrijving in.")

# Footer
st.caption("Prototype – Classificatie met embeddings + Logistic Regression")

# streamlit run streamlit_app.py