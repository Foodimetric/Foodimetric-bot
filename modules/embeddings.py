import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import logger


def get_embeddings():
    """Get Google Gemini embeddings. Raises clearly if API key is missing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    # Quick smoke-test so failures surface early
    test = embeddings.embed_query("test")
    logger.info(f"Google Gemini embeddings ready (dim: {len(test)})")
    return embeddings, "google_gemini", len(test)


def save_embedding_metadata(metadata_path, model_type, dimension):
    """Save embedding model metadata."""
    metadata = {"model_type": model_type, "dimension": dimension}
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved embedding metadata: {metadata}")


def load_embedding_metadata(metadata_path):
    """Load embedding model metadata."""
    try:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load embedding metadata: {e}")
        return None


def check_embedding_compatibility(metadata_path, current_model_type, current_dimension):
    """Check if current embeddings are compatible with stored vector store."""
    stored = load_embedding_metadata(metadata_path)
    if not stored:
        return False
    compatible = (
        stored["model_type"] == current_model_type
        and stored["dimension"] == current_dimension
    )
    if not compatible:
        logger.warning(
            f"Embedding mismatch — stored: {stored}, current: "
            f"{current_model_type} dim={current_dimension}"
        )
    return compatible
