import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pickle
from datetime import datetime
import hashlib
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='foodimetric_ai.log'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the models
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create a prompt template - modified for Foodimetric focus
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are Foodimetric AI, Foodimetric's AI nutrition assistant. Foodimetric is using technology to help Africans eat healthier by bridging the gap between nutrition knowledge and better health outcomes.

Your role is to:
1. Provide helpful nutrition information and advice
2. Naturally incorporate relevant Foodimetric features in your responses
3. Guide users to Foodimetric tools that can help them achieve their nutrition goals

Key Foodimetric features to recommend when relevant:
- Food Search: Look up nutritional content of local foods
- Multi-food Search: Compare nutrients across different foods
- Nutrient Search: Find foods rich in specific nutrients
- Food Diary: Track daily dietary intake
- Nutritional Assessment Calculators: Check nutritional status

When answering questions:
- Provide clear, direct answers first
- Then suggest relevant Foodimetric features that could help
- Use a friendly, helpful tone
- Keep responses concise and practical
- If unsure, recommend consulting a nutritionist

Use the following context to inform your nutrition knowledge, but respond naturally without directly quoting it:

Context: {context}

Question: {input}

Remember to:
- Be conversational and engaging
- Focus on practical advice
- Naturally integrate Foodimetric features
- Encourage users to try relevant tools on www.foodimetric.com
- Mention they can contact foodimetric@gmail.com for more information when appropriate""")
])

def get_document_hash(file_path):
    """Calculate hash of a file to detect changes"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def save_vector_store(vector_store, index_path, hash_path):
    """Save the FAISS index and document hashes"""
    # Save the FAISS index
    vector_store.save_local(index_path)
    
    # Save document hashes
    if os.path.exists(hash_path):
        with open(hash_path, 'rb') as f:
            doc_hashes = pickle.load(f)
    else:
        doc_hashes = {}
    
    # Update hashes for current documents
    data_dir = "data"
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        doc_hashes[file] = get_document_hash(file_path)
    
    with open(hash_path, 'wb') as f:
        pickle.dump(doc_hashes, f)

def should_update_embeddings(hash_path):
    """Check if documents have been modified since last embedding"""
    if not os.path.exists(hash_path):
        return True
        
    try:
        with open(hash_path, 'rb') as f:
            stored_hashes = pickle.load(f)
    except:
        return True
    
    data_dir = "data"
    current_files = set(os.listdir(data_dir))
    stored_files = set(stored_hashes.keys())
    
    # Check if files were added or removed
    if current_files != stored_files:
        return True
    
    # Check if any files were modified
    for file in current_files:
        file_path = os.path.join(data_dir, file)
        current_hash = get_document_hash(file_path)
        if file not in stored_hashes or stored_hashes[file] != current_hash:
            return True
    
    return False

def initialize_vector_store():
    """Initialize and return the vector store before starting the app"""
    storage_dir = "vector_store"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    index_path = os.path.join(storage_dir, "faiss_index")
    hash_path = os.path.join(storage_dir, "document_hashes.pkl")
    
    # Check if we need to update embeddings
    update_needed = should_update_embeddings(hash_path)
    
    if not update_needed and os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(
                index_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully!")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            update_needed = True
    
    # Load and process documents
    logger.info("Processing documents and creating new embeddings...")
    
    data_dir = "data"
    documents = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.error("No documents found in data directory")
        print("Error: Please add nutrition documents to the 'data' folder before starting the app.")
        sys.exit(1)
    
    # Process all files in the data directory
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                logger.info(f"Loaded PDF file: {file}")
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
                logger.info(f"Loaded text file: {file}")
        except Exception as e:
            logger.error(f"Error loading file {file}: {str(e)}")
            continue
    
    if not documents:
        logger.error("No valid documents found to process")
        print("Error: No valid documents found in the 'data' folder. Please add PDF or TXT files.")
        sys.exit(1)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(document_chunks)} chunks")
    
    # Create vector store
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    
    # Save the vector store and document hashes
    try:
        save_vector_store(vector_store, index_path, hash_path)
        logger.info("Vector store created and saved successfully!")
    except Exception as e:
        logger.error(f"Warning: Could not save vector store: {str(e)}")
    
    return vector_store

# Initialize vector store before starting the app
print("Initializing Foodimetric AI...")
vector_store = initialize_vector_store()
print("Initialization complete! Starting the chat interface...")

@cl.on_chat_start
async def start():
    # Create retriever
    retriever = vector_store.as_retriever()
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Store chain in user session
    cl.user_session.set("chain", rag_chain)
    
    await cl.Message(content="👋 Hi! I'm ready to help you with your nutrition questions. What would you like to know?").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Get the chain from user session
        chain = cl.user_session.get("chain")
        
        if chain is None:
            await cl.Message(content="Please wait for document processing to complete.").send()
            return
            
        # Get response using RAG
        response = await chain.ainvoke(message.content)
        
        # Send response back to user
        await cl.Message(content=response).send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()