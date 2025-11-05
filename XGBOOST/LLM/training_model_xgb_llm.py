import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# === 1. Dataset laden ===
data = pd.read_csv("dummy_opdrachten_dataset.csv")  # Kolommen: "omschrijving", "onderwerp"
X = data["omschrijving"].astype(str)
y = data["onderwerp"]

# === 2. Label encoder voor de minoren ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 4. Embeddings genereren ===
# Model: small en snel voor prototyping, semantische betekenis goed
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

X_train_emb = embed_model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = embed_model.encode(X_test.tolist(), show_progress_bar=True)

# === 5. XGBoost classifier trainen ===
xgb_model = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_emb, y_train)

# === 6. Evaluatie ===
y_pred = xgb_model.predict(X_test_emb)
print("=== Evaluatie ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === 7. Opslaan van model en encoder ===
joblib.dump(xgb_model, "nlp_model_xgb_embeddings.pkl")
joblib.dump(label_encoder, "label_encoder_xgb_llm.pkl")
print("✅ Model en encoder opgeslagen voor later gebruik")