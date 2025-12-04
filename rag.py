from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import ollama
import fitz  
import re
import os

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
    print(f"\nFound Content: {doc.page_content}")
  
  return results

# Generates answer from the LLM
def generate_answer(results, query, answer_model):
  context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

  PROMPT_TEMPLATE = '''
  Answer the question based on the following context:

  {context}

  -------


  Question: {question}
  '''
  # ChatPromptTemplate is Langchain helping developers not to crash the code when passing in the data to the context and question there inside the prompt
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context = context_text, question = query)

  model = ChatOllama(model=answer_model)

  answer_text = model.invoke(prompt)

  return answer_text.content

if __name__ == "__main__":
  filepath = "Internship Report Part 2 - Lucius Wilbert Tjoa - TP072404.pdf"
  database_path ="Database/report.txt"
  embed_model = "nomic-embed-text"
  chroma_db_directory = "Database/chroma_db"

  query = "What does S.A.G.E stands for?"
  k = 2
  chunk_size = 1000
  chunk_overlap = 200

  if os.path.exists(chroma_db_directory):
    print("Database Found, no need to embed to database anymore")

  else:
    print("Database not found, creating new one")

    docs = load_report_txt(database_path)

    splits = split_text_into_chunks(chunk_size, chunk_overlap, docs)

    vectorstore = ollama_embed_to_chroma_db(splits, embed_model, chroma_db_directory)
    
    print("Done creating new Database")

  results = query_similarity_search(embed_model, chroma_db_directory, query, k)
  # print(results)

  answer = generate_answer(results, query, "deepseek-r1:1.5b")
  print(answer)





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