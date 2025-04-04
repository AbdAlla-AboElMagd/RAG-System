import os
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA  # Added missing import

print("\n=== Initializing LLM Configuration ===")
llm_name = 'o1-mini'
print(f"Using model: {llm_name}")

print("\n=== Loading Vector Database ===")
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(f"Loaded vector database with {vectordb._collection.count()} documents")

# Create LLM instance with basic configuration
llm = ChatOpenAI(
    model=llm_name,
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_kwargs={
        'messages': [
            {'role': 'user', 'content': 'You are a helpful assistant.'}
        ]
    }
)

print("\n=== Setting up QA Chain ===")

# Update template to use basic roles
template = """Answer the question based on the context below. Be concise.
Context: {context}
Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Update chain configurations with correct prompt variables
chain_configs = {
    "stuff": {
        "prompt": QA_CHAIN_PROMPT,
        "verbose": True
    },
    "map_reduce": {
        "combine_prompt": PromptTemplate.from_template(
            """Given the following extracted parts of a long document and a question, combine them to give a final answer.
            Question: {question}
            Extracts: {summaries}
            Answer:"""
        ),
        "question_prompt": PromptTemplate.from_template(
            """Given this section of a document:
            {context}
            Answer this question: {question}
            Answer:"""
        ),
        "verbose": True
    },
    "refine": {
        "initial_response_name": "existing_answer",
        "document_variable_name": "context_str",
        "question_prompt": PromptTemplate.from_template(
            """Given this context:
            {context_str}
            Answer this question: {question}
            Answer:"""
        ),
        "refine_prompt": PromptTemplate.from_template(
            """Original answer: {existing_answer}
            New context: {context_str}
            Question: {question}
            Updated answer:"""
        ),
        "verbose": True
    }
}

print("\nTesting with different chain types...")
for chain_type, config in chain_configs.items():
    print(f"\n--- Testing {chain_type} chain ---")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type=chain_type,
            return_source_documents=True,
            chain_type_kwargs=config
        )
        
        question = "Is probability a class topic?"
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer ({chain_type}):", result["result"])
        
        if "source_documents" in result:
            print("\nSource Documents:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\nDocument {i}:")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"Error in {chain_type} chain: {str(e)}")

print("\nQA Testing completed")