from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma  # Updated import
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever  # Add this import
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Add constants at the top after imports
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'pdf')
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'chroma')
ALLOWED_EXTENSIONS = {'pdf'}  # Add this line

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Create static directory
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'js'), exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load environment variables
load_dotenv()

# Initialize chat history
chat_history = []

# Add new global variable for fallback preference
allow_global_fallback = False

class CustomRetriever(BaseRetriever, BaseModel):
    chroma_store: Any = Field(default=None)
    search_kwargs: dict = Field(default_factory=dict)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.chroma_store.similarity_search(
            query, 
            k=self.search_kwargs.get('k', 4)
        )
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

class ChromaWithPersistence:
    def __init__(self, persist_directory, embedding_function):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("documents")
        self.embedding_function = embedding_function

    def add_documents(self, documents: List[Document]):
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = [str(j) for j in range(i, i + len(batch))]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            embeddings = self.embedding_function.embed_documents(texts)
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            )
            documents.append(doc)
        return documents

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Return a retriever instance."""
        return CustomRetriever(chroma_store=self, search_kwargs=search_kwargs or {})

# Initialize the database and QA system
def init_qa():
    embeddings = OpenAIEmbeddings()
    db = ChromaWithPersistence(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    llm = ChatOpenAI(
        model='o1-mini'
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(
                """Using the following context, answer the question. If no relevant information is found in the context, indicate that clearly.

                Context: {context}
                
                Question: {question}
                
                Answer based on the context only:"""
            )
        }
    )
    
    return qa, llm  # Return both qa chain and llm

qa, llm = init_qa()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/toggle-fallback', methods=['POST'])
def toggle_fallback():
    global allow_global_fallback
    allow_global_fallback = request.json.get('enabled', False)
    return jsonify({'enabled': allow_global_fallback})

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history, allow_global_fallback
    
    query = request.json.get('message', '')
    if not query:
        return jsonify({'error': 'No message provided'}), 400

    try:
        answer = None
        sources = []
        is_local = True

        if not allow_global_fallback:
            # Try local documents first
            result = qa.invoke({
                "question": query,
                "chat_history": chat_history
            })
            answer = result['answer']
            
            if 'source_documents' in result:
                sources = [
                    {
                        'source': doc.metadata.get('source', 'Unknown'),
                        'content': doc.page_content[:200] + '...'
                    }
                    for doc in result['source_documents']
                ]
        else:
            # Direct to global response when fallback is enabled
            fallback_prompt = f"""Please provide a comprehensive answer to the following question, 
            based on your general knowledge: {query}"""
            direct_response = llm.invoke(fallback_prompt)
            answer = direct_response.content
            is_local = False

        # Update chat history
        chat_history.extend([(query, answer)])
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'is_local': is_local
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({'message': 'Chat history cleared'})

@app.route('/sources', methods=['GET'])
def get_sources():
    try:
        # List all PDF files in the docs/pdf directory
        pdf_files = []
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.endswith('.pdf'):
                pdf_files.append({
                    'name': file,
                    'path': os.path.join(app.config['UPLOAD_FOLDER'], file)
                })
        return jsonify({'sources': pdf_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create directories if they don't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the new PDF
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            
            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )
            splits = text_splitter.split_documents(pages)
            
            # Add to existing vectorstore
            embeddings = OpenAIEmbeddings()
            vectordb = ChromaWithPersistence(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            vectordb.add_documents(splits)
            
            return jsonify({'message': f'File {filename} uploaded and processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
