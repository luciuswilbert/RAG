import ollama

# # Generate the embedding
# response = ollama.embeddings(
#   model='nomic-embed-text',
#   prompt='The sky is blue because of rayleigh scattering'
# )

# # Access the vector
# vector = response['embedding']

# print(f"Vector length: {len(vector)}")
# print(vector[:5]) # Print first 5 dimensions

import fitz  # PyMuPDF

# Open the PDF
doc = fitz.open("Internship Report Part 2 - Lucius Wilbert Tjoa - TP072404.pdf") 
full_text = ""
for page in doc:
    full_text += page.get_text()

# The 'Clean' Count (Removing all spaces and newlines)
clean_text = full_text.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

with open("Database/report.txt", 'w', encoding='utf-8') as w:
    w.write(clean_text)
