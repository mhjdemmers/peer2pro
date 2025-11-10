import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Dataset laden 
data = pd.read_csv("dummy_opdrachten_dataset.csv")  # Kolommen: "omschrijving", "onderwerp"

X = data["omschrijving"].astype(str)
y = data["onderwerp"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline met TF-IDF + XGBoost 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams + bigrams
        max_features=5000    # aantal features beperken
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

# Trainen 
pipeline.fit(X_train, y_train)

# Evalueren 
y_pred = pipeline.predict(X_test)
print("=== Evaluatie ===")
print(classification_report(y_test, y_pred))

# Opslaan
joblib.dump(pipeline, "nlp_model_xgb_tfidf.pkl")
print("Model opgeslagen als 'nlp_model_xgb_tfidf.pkl'")