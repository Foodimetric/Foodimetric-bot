import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import get_embeddings
from .document_processor import (
    should_update_embeddings,
    load_documents,
    save_document_hashes
)
from .config import logger

def save_vector_store(vector_store, index_path, hash_path):
    """Save the FAISS index and document hashes"""
    vector_store.save_local(index_path)
    save_document_hashes(hash_path)

def load_existing_vector_store(index_path, embeddings):
    """Try to load existing vector store"""
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
        return None

def create_minimal_vector_store(embeddings):
    """Create a minimal vector store as fallback"""
    logger.warning("Creating minimal vector store with sample data...")
    print("Creating minimal vector store. Some features may be limited.")
    
    sample_docs = [Document(
        page_content="Nutrition information for Nigerian foods. Contact foodimetric@gmail.com for assistance.",
        metadata={"source": "fallback"}
    )]
    
    try:
        vector_store = FAISS.from_documents(sample_docs, embeddings)
        logger.info("Minimal vector store created successfully!")
        return vector_store
    except Exception as fallback_error:
        logger.error(f"Failed to create even minimal vector store: {str(fallback_error)}")
        print("Unable to create vector store. Please check your API key and try again later.")
        sys.exit(1)

def initialize_vector_store():
    """Initialize and return the vector store before starting the app"""
    storage_dir = "vector_store"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    index_path = os.path.join(storage_dir, "faiss_index")
    hash_path = os.path.join(storage_dir, "document_hashes.pkl")
    
    embeddings = get_embeddings()
    update_needed = should_update_embeddings(hash_path)
    
    if not update_needed and os.path.exists(index_path):
        vector_store = load_existing_vector_store(index_path, embeddings)
        if vector_store:
            return vector_store
        update_needed = True
    
    # Load and process documents
    logger.info("Processing documents and creating new embeddings...")
    document_chunks = load_documents()
    
    if not document_chunks:
        print("Error: Please add nutrition documents to the 'data' folder before starting the app.")
        sys.exit(1)
    
    # Create vector store
    try:
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        logger.info("Vector store created successfully!")
        
        # Save the vector store and document hashes
        try:
            save_vector_store(vector_store, index_path, hash_path)
            logger.info("Vector store saved successfully!")
        except Exception as e:
            logger.error(f"Warning: Could not save vector store: {str(e)}")
        
        return vector_store
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            logger.error(f"Google API quota exceeded: {error_msg}")
            print("Google API quota exceeded. Attempting to use cached embeddings...")
            
            # Try to load existing vector store even if it's outdated
            if os.path.exists(index_path):
                vector_store = load_existing_vector_store(index_path, embeddings)
                if vector_store:
                    print("Successfully loaded cached embeddings. App will work with existing data.")
                    return vector_store
            
            # If all else fails, create a minimal vector store
            return create_minimal_vector_store(embeddings)
        else:
            logger.error(f"Unexpected error creating vector store: {error_msg}")
            print(f"Error creating vector store: {error_msg}")
            sys.exit(1)