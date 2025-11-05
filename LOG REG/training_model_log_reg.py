import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# === 1. Dataset laden ===
data = pd.read_csv("dummy_opdrachten_dataset.csv")  # Kolommen: "omschrijving", "onderwerp"
X = data["omschrijving"].astype(str)
y = data["onderwerp"]

# === 2. Label encoder voor de onderwerpen ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 4. Nieuwe, meertalige embedding model (beter voor NL) ===
print("🔹 Genereren van embeddings met paraphrase-multilingual-mpnet-base-v2...")
embed_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

X_train_emb = embed_model.encode(X_train.tolist(), show_progress_bar=True, convert_to_numpy=True)
X_test_emb = embed_model.encode(X_test.tolist(), show_progress_bar=True, convert_to_numpy=True)

# === 5. Logistic Regression classifier ===
print("🔹 Model trainen (Logistic Regression)...")
clf = LogisticRegression(
    max_iter=2000,
    solver='lbfgs',
    multi_class='multinomial',
    n_jobs=-1
)
clf.fit(X_train_emb, y_train)

# === 6. Evaluatie ===
y_pred = clf.predict(X_test_emb)
print("\n=== Evaluatie ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === 7. Opslaan van model en encoder ===
joblib.dump(clf, "nlp_model_logreg_embeddings.pkl")
joblib.dump(label_encoder, "label_encoder_log_reg.pkl")
print("\n✅ Model en encoder opgeslagen voor later gebruik")