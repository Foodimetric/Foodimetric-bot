import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import logger

def get_embeddings():
    """Get embeddings with fallback to HuggingFace if Google fails"""
    try:
        google_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        # Test the embeddings with a small sample
        test_embedding = google_embeddings.embed_query("test")
        logger.info(f"Using Google Gemini embeddings (dimension: {len(test_embedding)})")
        return google_embeddings, "google_gemini", len(test_embedding)
    except Exception as e:
        logger.warning(f"Google Gemini embeddings failed: {str(e)}")
        logger.info("Falling back to local HuggingFace embeddings")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Test to get dimension
        test_embedding = hf_embeddings.embed_query("test")
        logger.info(f"Using HuggingFace embeddings (dimension: {len(test_embedding)})")
        return hf_embeddings, "huggingface_minilm", len(test_embedding)

def save_embedding_metadata(metadata_path, model_type, dimension):
    """Save embedding model metadata"""
    metadata = {
        'model_type': model_type,
        'dimension': dimension
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved embedding metadata: {metadata}")

def load_embedding_metadata(metadata_path):
    """Load embedding model metadata"""
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except Exception as e:
        logger.warning(f"Could not load embedding metadata: {str(e)}")
        return None

def check_embedding_compatibility(metadata_path, current_model_type, current_dimension):
    """Check if current embeddings are compatible with stored vector store"""
    stored_metadata = load_embedding_metadata(metadata_path)
    
    if not stored_metadata:
        return False
    
    # Check both model type and dimension
    is_compatible = (stored_metadata['model_type'] == current_model_type and 
                    stored_metadata['dimension'] == current_dimension)
    
    if not is_compatible:
        logger.warning("Embedding model mismatch detected!")
        logger.warning(f"Stored: {stored_metadata['model_type']} (dim: {stored_metadata['dimension']})")
        logger.warning(f"Current: {current_model_type} (dim: {current_dimension})")
    
    return is_compatible