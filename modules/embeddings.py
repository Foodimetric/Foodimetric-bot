import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import logger

# langchain-google-genai >= 2.x uses the new Google GenAI SDK.
# Model name format changed — no 'models/' prefix, new model name.
# gemini-embedding-001 is the current production embedding model.
EMBEDDING_MODEL = "gemini-embedding-001"


def get_embeddings():
    """
    Return Google Gemini embeddings using the current model.
    Raises early with a clear message if the API key is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it in your Render service → Environment settings."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        task_type="retrieval_document",
    )

    # Smoke-test so a bad key or quota issue surfaces immediately at startup
    try:
        test = embeddings.embed_query("test")
    except Exception as e:
        raise RuntimeError(
            f"Embedding smoke-test failed (model='{EMBEDDING_MODEL}'). "
            f"Check GEMINI_API_KEY and quota. Error: {e}"
        ) from e

    logger.info(f"Embeddings ready — model={EMBEDDING_MODEL}, dim={len(test)}")
    return embeddings, EMBEDDING_MODEL, len(test)


def save_embedding_metadata(metadata_path, model_type, dimension):
    """Persist embedding model metadata alongside the FAISS index."""
    metadata = {"model_type": model_type, "dimension": dimension}
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved embedding metadata: {metadata}")


def load_embedding_metadata(metadata_path):
    """Load persisted embedding metadata; returns None on any failure."""
    try:
        with open(metadata_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Could not load embedding metadata: {e}")
        return None


def check_embedding_compatibility(metadata_path, current_model_type, current_dimension):
    """Return True only if stored model type AND dimension match current values."""
    stored = load_embedding_metadata(metadata_path)
    if not stored:
        return False
    compatible = (
        stored["model_type"] == current_model_type
        and stored["dimension"] == current_dimension
    )
    if not compatible:
        logger.warning(
            f"Embedding mismatch — stored: {stored}, "
            f"current: {current_model_type} dim={current_dimension}. "
            "Vector store will be rebuilt."
        )
    return compatible
