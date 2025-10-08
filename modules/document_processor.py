import os
import hashlib
import pickle
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import logger

# Get the root directory (parent of modules folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def get_document_hash(file_path):
    """Calculate hash of a file to detect changes"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

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

def load_documents(data_dir=None):
    """Load and process all documents from the data directory"""
    if data_dir is None:
        data_dir = DATA_DIR
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.error("No documents found in data directory")
        return None
    
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
        return None
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(document_chunks)} chunks")
    
    return document_chunks

def save_document_hashes(hash_path, data_dir=None):
    """Save document hashes for change detection"""
    if data_dir is None:
        data_dir = DATA_DIR
    if os.path.exists(hash_path):
        with open(hash_path, 'rb') as f:
            doc_hashes = pickle.load(f)
    else:
        doc_hashes = {}
    
    # Update hashes for current documents
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        doc_hashes[file] = get_document_hash(file_path)
    
    with open(hash_path, 'wb') as f:
        pickle.dump(doc_hashes, f)