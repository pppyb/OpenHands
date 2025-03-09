#!/usr/bin/env python3
"""
Simple RAG test that doesn't require a git repository
"""

import os
import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def create_simple_index(directory, extensions=None):
    """Create a simple index for a directory"""
    if extensions is None:
        extensions = [".py", ".md"]
        
    # Initialize the model
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Collect documents
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    rel_path = os.path.relpath(file_path, directory)
                    documents.append({
                        'id': rel_path,
                        'path': rel_path,
                        'content': content
                    })
                except (UnicodeDecodeError, IOError):
                    continue
    
    if not documents:
        return "No documents found"
    
    # Create embeddings
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=False)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index and documents
    index_dir = os.path.join(directory, ".simple_index")
    os.makedirs(index_dir, exist_ok=True)
    
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "documents.pkl"), 'wb') as f:
        pickle.dump(documents, f)
        
    return f"Created index with {len(documents)} documents"


def search_index(directory, query, k=3):
    """Search the index"""
    index_dir = os.path.join(directory, ".simple_index")
    
    # Check if index exists
    if not (os.path.exists(os.path.join(index_dir, "index.faiss")) and 
            os.path.exists(os.path.join(index_dir, "documents.pkl"))):
        return "Index not found. Please create it first."
    
    # Load index and documents
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "documents.pkl"), 'rb') as f:
        documents = pickle.load(f)
    
    # Initialize the model
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    
    # Encode query
    query_embedding = model.encode([query], convert_to_tensor=False)
    
    # Search
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(documents):
            doc = documents[idx]
            results.append({
                'file': doc['path'],
                'score': float(score),
                'content': doc['content']
            })
    
    return results


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python simple_rag_test.py [create|search] [query]")
        sys.exit(1)
    
    directory = "/workspace/OpenHands/examples"
    command = sys.argv[1]
    
    if command == "create":
        result = create_simple_index(directory)
        print(result)
    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide a search query")
            sys.exit(1)
        query = sys.argv[2]
        results = search_index(directory, query)
        
        if isinstance(results, str):
            print(results)
        else:
            print(f"Found {len(results)} results:")
            for i, res in enumerate(results, 1):
                print(f"\nResult {i}: {res['file']} (Similarity: {res['score']:.3f})")
                print("-" * 80)
                print(res["content"][:500] + "..." if len(res["content"]) > 500 else res["content"])
                print("-" * 80)
    else:
        print("Unknown command. Use 'create' or 'search'")


if __name__ == "__main__":
    main()