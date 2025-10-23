from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self  # geen training nodig

    def transform(self, X):
        # Zorg dat X altijd een lijst van strings wordt
        if hasattr(X, "tolist"):
            X = X.tolist()
        return self.model.encode(X, convert_to_numpy=True)