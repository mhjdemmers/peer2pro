from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

print(model.encode(["Dit is een test"]))