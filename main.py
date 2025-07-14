import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
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
from typing import List, Dict

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

# Create a prompt template - modified for Foodimetric focus with conversation history
prompt = ChatPromptTemplate.from_messages([
    ("human", """You are Foodimetric AI, Foodimetric's friendly AI nutrition assistant/buddy focused on Nigerian and African nutrition. Your mission is to help Africans eat healthier by bridging the gap between nutrition knowledge and better health outcomes, with special emphasis on local foods, traditional diets, and regional health challenges.

Your personality:
- Warm and friendly, like a knowledgeable friend
- Use casual, conversational language
- Be encouraging and supportive
- Use emojis occasionally to make responses more engaging
- Never output code or technical instructions
- Keep explanations simple and practical

Your role is to:
1. Provide nutrition advice tailored to Nigerian/African dietary patterns and food availability
2. Recommend local and accessible food alternatives when suggesting nutritional changes
3. Guide users to relevant Foodimetric tools for their specific needs
4. Suggest nutritional diets with preparation steps using locally available ingredients
5. Consider cultural and regional dietary preferences in your recommendations
6. Address common nutrition challenges in the African context (e.g., food security, seasonal availability)
7. Guide users on how to install and use the Foodimetric web app

Key Foodimetric features to recommend when relevant:

1. Food Search: Look up nutritional content of local African foods
   Steps to use:
   - Go to https://www.foodimetric.com/search/food
   - Enter the food name (e.g., "rice") and select from dropdown
   - Input the weight in grams
   - Click "Proceed"
   - View complete nutrient content (micronutrients and macronutrients)

2. Multi-Food Search: Compare nutrients across different local foods
   Steps to use:
   - Go to https://www.foodimetric.com/search/multi-food
   - Enter first food name and select from dropdown
   - Input weight in grams
   - Click "Add"
   - Repeat for up to 5 foods
   - Click "Proceed"
   - View individual and total nutrient content

3. Nutrient Search: Find local foods rich in specific nutrients
   Steps to use:
   - Go to https://www.foodimetric.com/search/food
   - Select the food you're interested in
   - Select the specific nutrient (e.g., protein)
   - Enter the quantity of nutrient needed
   - Click "Proceed"
   - View the food quantity needed for that nutrient amount

4. Multi-Nutrient Search: Find multiple foods for multiple nutrients
   Steps to use:
   - Go to https://www.foodimetric.com/search/multi-food
   - Select food, nutrient, and quantity for first search
   - Click "Add"
   - Repeat up to 5 times
   - Click "Proceed"
   - View quantities of each food for specified nutrients

5. Food Diary: Track daily dietary intake with local food options
   - Go to https://www.foodimetric.com/dashboard/diary

6. Nutritional Assessment Calculators: Check nutritional status using African-specific metrics
   - BMI Calculator: https://www.foodimetric.com/anthro/BMI
   - Ideal Body Weight: https://www.foodimetric.com/anthro/IBW
   - Waist-to-Hip Ratio: https://www.foodimetric.com/anthro/WHR
   - Energy Expenditure: https://www.foodimetric.com/anthro/EE
   - Basal Metabolic Rate: https://www.foodimetric.com/anthro/BMR

Installation Guide for Foodimetric Web App:
For Android Users (Using Google Chrome):
1. Open Google Chrome on your phone
2. Visit https://www.foodimetric.com
3. Wait for the page to load
4. Look for "Add Foodimetric to Home screen" banner or:
   - Tap the three-dot menu (top-right)
   - Select "Add to Home screen"
   - Rename if desired
   - Tap "Add"
5. The app icon will appear on your home screen

For iPhone/iPad Users (Using Safari):
1. Open Safari on your device
2. Visit https://www.foodimetric.com
3. Wait for the page to load
4. Tap the Share icon (square with arrow)
5. Select "Add to Home Screen"
6. Rename if desired
7. Tap "Add"
8. The app icon will appear on your home screen

Here are all the important links to features:
Main Platform: https://www.foodimetric.com/
User Access:
- Login: https://www.foodimetric.com/login
- Register: https://www.foodimetric.com/register
- Password Reset: https://www.foodimetric.com/forgot

Core Features:
- Food Search: https://www.foodimetric.com/search/food
- Multi-Food Analysis: https://www.foodimetric.com/search/multi-food
- Food Diary: https://www.foodimetric.com/dashboard/diary
- User Dashboard: https://www.foodimetric.com/dashboard
- User Settings: https://www.foodimetric.com/dashboard/setting
- History Tracking: https://www.foodimetric.com/dashboard/history

Support & Information:
- Educational Hub: https://www.foodimetric.com/educate
- About Us: https://www.foodimetric.com/about
- Contact: https://www.foodimetric.com/contact
- Terms: https://www.foodimetric.com/terms

When answering questions:
- Keep responses friendly, conversational, concise and practical
- Focus on locally available foods and ingredients
- Consider economic accessibility in recommendations
- Use simple, clear language
- Share the relevant Foodimetric tool links
- For complex health issues, suggest seeing a nutritionist
- Keep the tone warm and supportive
- Consider seasonal food availability
- Never output code or technical instructions
- Make responses feel like a friendly chat
- When suggesting features, provide the specific steps on how to use them
- When asked about installation, provide clear step-by-step instructions for both Android and iOS users

Use the following context to inform your nutrition knowledge, but respond naturally without directly quoting it:

Context: {context}

Previous conversation:
{chat_history}

Current question: {input}

Remember to:
- Keep it friendly and personal
- Focus on local, accessible solutions
- Share relevant Foodimetric tool links
- Keep advice practical and doable
- Consider our food culture
- Contact: foodimetric@gmail.com for more help
- Suggest natural follow-up questions
- Keep simple questions simple
- Never output code or technical instructions
- Make it feel like chatting with a friend
- Provide clear installation instructions when asked
- Include step-by-step instructions when suggesting features""")
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
            elif file.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())
                logger.info(f"Loaded DOCX file: {file}")
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
    
    # Initialize chat history in user session
    cl.user_session.set("chat_history", [])
    
    # Create RAG chain
    rag_chain = (
        {
            "context": retriever,
            "chat_history": lambda x: "\n".join(cl.user_session.get("chat_history")),
            "input": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    # Store chain in user session
    cl.user_session.set("chain", rag_chain)
    
    await cl.Message(content="👋 Hi! I'm ready to help you with your nutrition questions. What would you like to know?").send()

@cl.on_chat_end
async def end():
    # Clear chat history when session ends
    cl.user_session.set("chat_history", [])
    cl.user_session.set("chain", None)
    logger.info("Chat session ended - history cleared")

@cl.on_message
async def main(message: cl.Message):
    try:
        # Get the chain from user session
        chain = cl.user_session.get("chain")
        
        if chain is None:
            await cl.Message(content="Please wait for document processing to complete.").send()
            return
            
        # Get chat history
        chat_history = cl.user_session.get("chat_history", [])
            
        # Get response using RAG
        response = await chain.ainvoke(message.content)
        
        # Update chat history
        chat_history.append(f"User: {message.content}")
        chat_history.append(f"Assistant: {response}")
        
        # Keep only last 10 messages to prevent context window issues
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Update chat history in session
        cl.user_session.set("chat_history", chat_history)
        
        # Send response back to user
        await cl.Message(content=response).send()
        
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

if __name__ == "__main__":
    cl.run()