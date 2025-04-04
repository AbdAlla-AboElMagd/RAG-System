import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

print("\n=== Loading PDF Documents ===")
try:
    loaders = [
        PyPDFLoader("docs/pdf/Django - Getting Python in The Web - lecture 1.pdf"),
        PyPDFLoader("docs/pdf/Python - Lecture 1.pdf"),
        PyPDFLoader("docs/pdf/Python - Lecture 2.pdf"),
        PyPDFLoader("docs/pdf/Python - Lecture 3.pdf")
    ]
    docs = []
    for i, loader in enumerate(loaders, 1):
        print(f"Loading document {i}/{len(loaders)}...")
        docs.extend(loader.load())
    print(f"Successfully loaded {len(docs)} documents")

except Exception as e:
    print(f"Error loading documents: {str(e)}")
    sys.exit(1)

print("\n=== Splitting Documents ===")
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} splits")

print("\n=== Testing Embeddings ===")
embedding = OpenAIEmbeddings()

# Test sentences
sentences = {
    "sentence1": "i like dogs",
    "sentence2": "i like canines",
    "sentence3": "the weather is ugly outside"
}

# Generate embeddings
embeddings = {}
for name, sentence in sentences.items():
    print(f"Generating embedding for {name}...")
    embeddings[name] = embedding.embed_query(sentence)

# Compare similarities
print("\nComparing similarities:")
import numpy as np
for i in range(1, 4):
    for j in range(i+1, 4):
        similarity = np.dot(embeddings[f"sentence{i}"], embeddings[f"sentence{j}"])
        print(f"Similarity between sentence{i} and sentence{j}: {similarity:.4f}")

print("\n=== Creating Vector Database ===")
persist_directory = 'docs/chroma/'
print(f"Using persist directory: {persist_directory}")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

print(f"Vector database created with {vectordb._collection.count()} documents")

print("\n=== Testing Similarity Search ===")
questions = [
    "is there an email i can ask for help",
    "what did they say about matlab?",
    "what did they say about regression in the third lecture?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    docs = vectordb.similarity_search(question, k=3)
    print(f"Found {len(docs)} relevant documents")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")

# Persist the database
print("\nPersisting vector database...")
vectordb.persist()
print("Database persisted successfully")