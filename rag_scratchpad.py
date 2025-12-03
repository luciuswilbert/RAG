import numpy as np

sentence_A = np.array([0.1, 0.5, 0.9])
sentence_B = np.array([0.2, 0.5, 0.8])

dot_product = np.dot(sentence_A, sentence_B)
print(dot_product)
mul = (0.1 * 0.2) + (0.5 * 0.5) + (0.9 * 0.8)
print(mul)