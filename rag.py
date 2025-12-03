import ollama
import fitz  
import re

# # Generate the embedding
# response = ollama.embeddings(
#   model='nomic-embed-text',
#   prompt='The sky is blue because of rayleigh scattering'
# )

# # Access the vector
# vector = response['embedding']

# print(f"Vector length: {len(vector)}")
# print(vector[:5]) # Print first 5 dimensions


# # Open the PDF
# doc = fitz.open("Internship Report Part 2 - Lucius Wilbert Tjoa - TP072404.pdf") 
# full_text = ""
# for page in doc:
#     full_text += page.get_text()

# # The 'Clean' Count (Removing all spaces and newlines)
# # clean_text = full_text.replace("\n", "").replace("\r", "").replace("\t", "")
# clean_text = re.sub(r'\s+', ' ', full_text)

# with open("Database/report.txt", 'w', encoding='utf-8') as w:
#     w.write(clean_text)
  
def read_pdf(filepath):
  doc = fitz.open(filepath) 
  full_text = ""
  for page in doc:
      full_text += page.get_text()
  return full_text

def clean_text(full_text):
   clean_texts = re.sub(r'\s+', ' ', full_text)
   return clean_texts

def write_text(database_path, clean_texts):
   with open(database_path, 'w', encoding='utf-8') as x:
      x.write(clean_texts)