from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

sentence = "RAG is a powerful technique for AI."
embedding = model.encode(sentence)

print(embedding.shape)

docs = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Photosynthesis is how plants make food."
]

doc_embeddings = model.encode(docs)
print(doc_embeddings.shape)

query = "Tell me about Paris."
query_embedding = model.encode(query)

hits = util.cos_sim(query_embedding, doc_embeddings)
print(hits)

best_index = hits.argmax()
print('Best Index: ', best_index)
print(docs[best_index])
