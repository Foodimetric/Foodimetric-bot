import os
import sys
import time
import shutil
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import (
    get_embeddings,
    save_embedding_metadata,
    check_embedding_compatibility,
)
from .document_processor import (
    should_update_embeddings,
    load_documents,
    save_document_hashes,
)
from .config import logger

# Batch size for embedding requests — smaller = fewer 429s
EMBED_BATCH_SIZE = 50
# Retry settings for 429 / transient errors
MAX_RETRIES = 6
RETRY_BASE_DELAY = 10  # seconds, doubles each attempt


def _embed_with_retry(embeddings_obj, chunks):
    """
    Call FAISS.from_documents with exponential backoff on rate-limit errors.
    Splits chunks into smaller batches to avoid hitting quota in one burst.
    """
    from langchain_community.vectorstores import FAISS as _FAISS

    vector_store = None
    total = len(chunks)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)…")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if vector_store is None:
                    vector_store = _FAISS.from_documents(batch, embeddings_obj)
                else:
                    batch_store = _FAISS.from_documents(batch, embeddings_obj)
                    vector_store.merge_from(batch_store)
                break  # success — move to next batch
            except Exception as e:
                msg = str(e)
                is_quota = "429" in msg or "quota" in msg.lower() or "resource_exhausted" in msg.lower()
                if is_quota and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Rate limit hit on batch {batch_num} "
                        f"(attempt {attempt}/{MAX_RETRIES}). "
                        f"Waiting {delay}s before retry…"
                    )
                    time.sleep(delay)
                else:
                    raise  # non-quota error or out of retries

        # Small pause between batches to stay under quota
        if i + EMBED_BATCH_SIZE < total:
            time.sleep(1)

    return vector_store


def save_vector_store(vector_store, index_path, hash_path):
    vector_store.save_local(index_path)
    save_document_hashes(hash_path)


def load_existing_vector_store(index_path, embeddings):
    try:
        vs = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded from disk successfully.")
        return vs
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


def _wipe_index(index_path):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        logger.warning(f"Removed stale index at {index_path} — will rebuild.")


def initialize_vector_store():
    storage_dir = "vector_store"
    os.makedirs(storage_dir, exist_ok=True)

    index_path    = os.path.join(storage_dir, "faiss_index")
    hash_path     = os.path.join(storage_dir, "document_hashes.pkl")
    metadata_path = os.path.join(storage_dir, "embedding_metadata.pkl")

    embeddings, model_type, dimension = get_embeddings()

    embeddings_compatible = check_embedding_compatibility(
        metadata_path, model_type, dimension
    )

    # Wipe index built with old/unknown model (no metadata file)
    if not os.path.exists(metadata_path) and os.path.exists(index_path):
        logger.warning("FAISS index found with no metadata — built with deprecated model. Rebuilding.")
        _wipe_index(index_path)
        embeddings_compatible = False

    docs_changed  = should_update_embeddings(hash_path)
    update_needed = docs_changed or not embeddings_compatible

    # Try loading existing index
    if not update_needed and os.path.exists(index_path):
        vs = load_existing_vector_store(index_path, embeddings)
        if vs:
            logger.info(f"Using cached vector store ({model_type}, dim={dimension}).")
            return vs
        update_needed = True

    # Rebuild
    if not embeddings_compatible:
        logger.warning("Embedding model changed — rebuilding vector store.")
    if docs_changed:
        logger.info("Documents changed — rebuilding vector store.")

    logger.info(f"Building vector store with {model_type} (dim={dimension})…")
    chunks = load_documents()

    if not chunks:
        logger.error("No documents found in data/.")
        sys.exit(1)

    try:
        vector_store = _embed_with_retry(embeddings, chunks)
        logger.info("Vector store built successfully.")
        try:
            save_vector_store(vector_store, index_path, hash_path)
            save_embedding_metadata(metadata_path, model_type, dimension)
            logger.info("Vector store saved to disk.")
        except Exception as e:
            logger.warning(f"Could not save vector store (non-fatal): {e}")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        # Last resort — load whatever is on disk even if stale
        if os.path.exists(index_path):
            logger.warning("Attempting emergency load of existing index.")
            vs = load_existing_vector_store(index_path, embeddings)
            if vs:
                return vs
        logger.error("No vector store available. Check API quota and redeploy.")
        sys.exit(1)
