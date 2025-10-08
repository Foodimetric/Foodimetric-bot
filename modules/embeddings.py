import os
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
        google_embeddings.embed_query("test")
        logger.info("Using Google Gemini embeddings")
        return google_embeddings
    except Exception as e:
        logger.warning(f"Google Gemini embeddings failed: {str(e)}")
        logger.info("Falling back to local HuggingFace embeddings")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )