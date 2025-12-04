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

'''-----------------------------------------------------------------------------------------------------------------------------------'''

def load_report_txt(database_path):
   loader = TextLoader(database_path, encoding="utf-8")
   docs = loader.load()
   return docs

def split_text_into_chunks(chunk_size, chunk_overlap, docs):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   splits = text_splitter.split_documents(docs)
   return splits

def ollama_embed_to_chroma_db(splits, embed_model, chroma_db_directory):
   embedding_function = OllamaEmbeddings(model=embed_model)

   vectorstore = Chroma.from_documents(
      documents = splits,
      embedding = embedding_function,
      persist_directory = chroma_db_directory
   )

   return vectorstore

def query_similarity_search(embed_model, chroma_db_directory, query, k):
  embedding_function = OllamaEmbeddings(model=embed_model)

  vectorstore = Chroma(
     persist_directory=chroma_db_directory,
     embedding_function=embedding_function
    )
  
  query = query

  # Langchain handles the similarity_search function
  results = vectorstore.similarity_search(query, k=k)

  for doc in results:
    print(f"\Found Content: {doc.page_content}")
  
  return results

if __name__ == "__main__":
  filepath = "Internship Report Part 2 - Lucius Wilbert Tjoa - TP072404.pdf"
  database_path ="Database/report.txt"
  embed_model = "nomic-embed-text"
  chroma_db_directory = "Database/chroma_db"

  query = "What does S.A.G.E stands for?"
  k = 5
  chunk_size = 1000
  chunk_overlap = 200

  docs = load_report_txt(database_path)

  splits = split_text_into_chunks(chunk_size, chunk_overlap, docs)

  vectorstore = ollama_embed_to_chroma_db(splits, embed_model, chroma_db_directory)

  results = query_similarity_search(embed_model, chroma_db_directory, query, k)





  # full_text = read_pdf(filepath)
  # clean_text = clean_text(full_text)

  # word_count = len(clean_text.split())
  # print(f"Total Word Count: {word_count}")

  # write_text(database_path, clean_text)
  # print("Done")














# # Generate the embedding
# response = ollama.embeddings(
#   model='nomic-embed-text',
#   prompt='The sky is blue because of rayleigh scattering'
# )

# # Access the vector
# vector = response['embedding']

# print(f"Vector length: {len(vector)}")
# print(vector[:5]) # Print first 5 dimensions