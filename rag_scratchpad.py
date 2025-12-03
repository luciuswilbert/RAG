import numpy as np

# sentence_A = np.array([0.1, 0.5, 0.9])
# sentence_B = np.array([0.2, 0.5, 0.8])

# dot_product = np.dot(sentence_A, sentence_B)
# print("np.dot():", dot_product)
# mul = (0.1 * 0.2) + (0.5 * 0.5) + (0.9 * 0.8)
# print("Manual Multiplication:", mul)

# norm_A = np.linalg.norm(sentence_A)
# norm_B = np.linalg.norm(sentence_B)
# print("A np.linalg.norm(): ", norm_A)
# print("B np.linalg.norm(): ", norm_B)

# cosine_similarity = dot_product / (norm_A * norm_B)
# print("Cosine_Similarity: ", cosine_similarity)

database = np.array([[0.2, 0.5, 0.8],
                    [0.1, 0.9, 0.1],
                    [0.9, 0.1, 0.1]])

# 1. Define the query vector (matching our previous Sentence A)
query = np.array([0.1, 0.5, 0.9])

# 2. Calculate the query's norm just once (for efficiency)
norm_query = np.linalg.norm(query)

# 3. Loop through the database
for doc_vector in database:
    # A. Calculate Dot Product (query vs doc_vector)
    current_dot = np.dot(query, doc_vector)
    
    # B. Calculate Norm of the current doc_vector
    norm_doc = np.linalg.norm(doc_vector)
    
    # C. Calculate Cosine Similarity
    similarity = current_dot / (norm_query * norm_doc)
    
    print(similarity)