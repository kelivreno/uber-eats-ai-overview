from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["cheesy pizza", "Extra Large Meat Lovers - Whole Pie - $15.99"]
embeddings = model.encode(sentences)
print(embeddings.shape)