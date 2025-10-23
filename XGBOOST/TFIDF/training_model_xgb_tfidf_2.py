import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === 1. Dataset laden ===
data = pd.read_csv("dummy_opdrachten_dataset.csv")  # Kolommen: "omschrijving", "onderwerp"

X = data["omschrijving"].astype(str)
y = data["onderwerp"]

# === 2. LabelEncoder toepassen (strings → numerieke waarden) ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 3. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 4. Pipeline met TF-IDF + XGBoost ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# === 5. Trainen ===
pipeline.fit(X_train, y_train)

# === 6. Evalueren ===
y_pred = pipeline.predict(X_test)

print("=== Evaluatie ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === 7. Opslaan van model én labelencoder ===
joblib.dump(pipeline, "nlp_model_xgb_tfidf.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model opgeslagen als 'nlp_model_xgb_tfidf.pkl'")
print("✅ LabelEncoder opgeslagen als 'label_encoder.pkl'")