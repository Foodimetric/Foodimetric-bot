#!/usr/bin/env python3
"""
Run this script LOCALLY (not on Render) whenever you update files in data/.
It builds the FAISS vector store and saves it to vector_store/ so Render
can load it instantly on startup without any API calls.

Usage:
    GEMINI_API_KEY=your_key python3 build_vector_store.py
"""
import os
import sys

# Ensure we can import modules from this directory
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: Set GEMINI_API_KEY in your .env file or environment before running.")
    sys.exit(1)

print("Building vector store — this may take a few minutes for large document sets...")
from modules.vector_store import initialize_vector_store

# Force a fresh rebuild by deleting stale metadata
metadata_path = os.path.join("vector_store", "embedding_metadata.pkl")
hash_path     = os.path.join("vector_store", "document_hashes.pkl")

for path in [metadata_path, hash_path]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed {path} to force fresh build.")

vs = initialize_vector_store()
print("\n✅ Vector store built and saved to vector_store/")
print("   Now commit the vector_store/ folder to git and push to Render.")
print("   Render will load it instantly — no embedding API calls on startup.")
