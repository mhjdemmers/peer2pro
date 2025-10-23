import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from embedding_transformer import EmbeddingTransformer

# === 1. Dataset laden ===
data = pd.read_csv("trainingsdata3.csv")  # Kolommen: "omschrijving", "label"

X = data["omschrijving"].astype(str)
y = data["label"]

# === 2. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Pipeline met EmbeddingTransformer ===
pipeline = Pipeline([
    ("embeddings", EmbeddingTransformer(model_name="all-MiniLM-L6-v2")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# === 4. Trainen ===
pipeline.fit(X_train, y_train)

# === 5. Evalueren ===
y_pred = pipeline.predict(X_test)
print("=== Evaluatie ===")
print(classification_report(y_test, y_pred))

# === 6. Opslaan ===
joblib.dump(pipeline, "nlp_model.pkl")
print("Model opgeslagen als 'nlp_model.pkl'")