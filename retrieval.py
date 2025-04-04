import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.retrievers import SVMRetriever, TFIDFRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("\n=== Loading Vector Database ===")
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(f"Loaded vector database with {vectordb._collection.count()} documents")

print("\n=== Testing Small Database ===")
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]
smalldb = Chroma.from_texts(texts, embedding=embedding)
question = "Tell me about all-white mushrooms with large fruiting bodies"
print(f"\nQuestion: {question}")
print("\nSimilarity Search Results:")
for doc in smalldb.similarity_search(question, k=2):
    print(f"- {doc.page_content}")

print("\nMMR Search Results:")
for doc in smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3):
    print(f"- {doc.page_content}")

print("\n=== Testing Main Database Queries ===")
for query in ["what did they say about matlab?", 
              "what did they say about regression in the third lecture?"]:
    print(f"\nQuery: {query}")
    
    print("\nSimilarity Search Results:")
    docs_ss = vectordb.similarity_search(query, k=3)
    for i, doc in enumerate(docs_ss, 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    print("\nMMR Search Results:")
    docs_mmr = vectordb.max_marginal_relevance_search(query, k=3)
    for i, doc in enumerate(docs_mmr, 1):
        print(f"\nDocument {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")

print("\n=== Testing Retrieval QA ===")
# Use basic chat configuration
llm = ChatOpenAI(
    model='o1-mini',
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_kwargs={
        'messages': [
            {'role': 'user', 'content': 'You are a helpful assistant.'}
        ]
    }
)

# Create the retrieval chain with simpler configuration
try:
    print("Testing LLM connection with o1-mini...")
    test_response = llm.invoke("test", messages=[{'role': 'user', 'content': 'test'}])
    print("LLM connection successful")
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Changed to simpler chain type
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    question = "what did they say about regression in the third lecture?"
    print(f"\nProcessing question: {question}")
    
    result = qa_chain.invoke({"query": question})
    print("\nAnswer:", result.get("result", "No answer found"))
    print("\nSource Documents:")
    for i, doc in enumerate(result.get("source_documents", []), 1):
        print(f"\nDocument {i}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")

except Exception as e:
    print(f"Error with LLM: {str(e)}")
    print("Falling back to simple retrieval without LLM")
    # Continue with basic retrieval functionality
    for query in ["what did they say about matlab?", 
                  "what did they say about regression in the third lecture?"]:
        print(f"\nQuery: {query}")
        try:
            docs = vectordb.similarity_search(query, k=3)
            if docs:
                print(f"Found {len(docs)} relevant documents")
                for i, doc in enumerate(docs, 1):
                    print(f"\nDocument {i}:")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content: {doc.page_content[:200]}...")
            else:
                print("No relevant documents found")
        except Exception as search_error:
            print(f"Error in similarity search: {str(search_error)}")
            print("Trying alternative search method...")
            try:
                docs = vectordb.max_marginal_relevance_search(query, k=3)
                print(f"Found {len(docs)} documents using MMR search")
                for i, doc in enumerate(docs, 1):
                    print(f"\nDocument {i}:")
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"Content: {doc.page_content[:200]}...")
            except Exception as mmr_error:
                print(f"Error in MMR search: {str(mmr_error)}")

print("\n=== Testing Contextual Compression ===")
def pretty_print_docs(docs):
    if not docs:
        print("No documents found")
        return
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}/{len(docs)}:")
        print("-" * 40)
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        print("-" * 40)

# Initialize LLM for compression
compression_llm = ChatOpenAI(
    model='o1-mini',
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Initialize compressor before the loop
print("Initializing document compressor...")
try:
    compressor = LLMChainExtractor.from_llm(compression_llm)
    print("Compressor initialized successfully")
except Exception as e:
    print(f"Error initializing compressor: {str(e)}")
    print("Skipping compression tests")
    compressor = None

# Test with supported search types only if compressor is available
if compressor:
    for search_type in ["similarity", "mmr"]:
        print(f"\nTesting {search_type.upper()} compression retrieval...")
        print(f"Step 1/3: Initializing {search_type} retriever")
        try:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vectordb.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": 3}
                )
            )
            print(f"Step 2/3: Processing query with {search_type}")
            compressed_docs = compression_retriever.invoke("what did they say about matlab?")
            print(f"Step 3/3: Displaying {search_type} results")
            pretty_print_docs(compressed_docs)
            print(f"{search_type.upper()} retrieval completed successfully")
        except Exception as e:
            print(f"Error in {search_type} compression: {str(e)}")

print("\n=== Testing Alternative Retrievers ===")
# Load and prepare text
print("Step 1/4: Loading PDF document...")
loader = PyPDFLoader("docs/pdf/Python - Lecture 1.pdf")
pages = loader.load()

print("Step 2/4: Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(" ".join([p.page_content for p in pages]))
print(f"Created {len(splits)} text splits")

print("Step 3/4: Initializing retrievers...")
svm_retriever = SVMRetriever.from_texts(splits, embedding)
tfidf_retriever = TFIDFRetriever.from_texts(splits)

print("Step 4/4: Testing retrievers with queries...")
for question in ["What are major topics for this class?", "what did they say about matlab?"]:
    print(f"\nProcessing Question: {question}")
    
    print("\nSVM Retriever Results:")
    docs = svm_retriever.invoke(question)
    print(f"Found {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}/{len(docs)}:")
        print(f"- {doc.page_content[:200]}...")
    
    print("\nTFIDF Retriever Results:")
    docs = tfidf_retriever.invoke(question)
    print(f"Found {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}/{len(docs)}:")
        print(f"- {doc.page_content[:200]}...")

print("\nRetrieval testing completed")

