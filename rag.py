from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import ollama
import fitz  
import re


  
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

if __name__ == "__main__":
  filepath = "Internship Report Part 2 - Lucius Wilbert Tjoa - TP072404.pdf"
  database_path ="Database/report.txt"

  full_text = read_pdf(filepath)
  clean_text = clean_text(full_text)

  word_count = len(clean_text.split())
  print(f"Total Word Count: {word_count}")

  write_text(database_path, clean_text)
  print("Done")













# # Generate the embedding
# response = ollama.embeddings(
#   model='nomic-embed-text',
#   prompt='The sky is blue because of rayleigh scattering'
# )

# # Access the vector
# vector = response['embedding']

# print(f"Vector length: {len(vector)}")
# print(vector[:5]) # Print first 5 dimensions