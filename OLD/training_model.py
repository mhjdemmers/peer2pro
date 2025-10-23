import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
# conda activate casusnlp

# 1. Data inladen
data = pd.read_csv("trainingsdata3.csv")

X = data["omschrijving"]   
y = data["label"]                  

# 2. Train/test split (optioneel voor evaluatie)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. NLP pipeline: TF-IDF + Logistic Regression classifier
nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Trainen
nlp_pipeline.fit(X_train, y_train)

# 5. Optioneel: Accuracy check
print("Train accuracy:", nlp_pipeline.score(X_train, y_train))
print("Test accuracy:", nlp_pipeline.score(X_test, y_test))

# 6. Opslaan van het model voor later gebruik
joblib.dump(nlp_pipeline, "nlp_model.pkl")
print("✅ Model opgeslagen als 'nlp_model.pkl'")