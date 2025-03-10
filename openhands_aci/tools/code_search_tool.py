"""
Code search tool for OpenHands.

This module provides functionality to search for relevant code in a repository
using semantic search. It is used by the CodeSearchAction to find code snippets
that match a natural language query.
"""

import os
import glob
from typing import Dict, List, Any, Optional


def code_search_tool(
    query: str,
    repo_path: str,
    save_dir: Optional[str] = None,
    extensions: Optional[List[str]] = None,
    k: int = 5,
    remove_duplicates: bool = True,
    min_score: float = 0.5
) -> Dict[str, Any]:
    """
    Search for relevant code in a repository using semantic search.
    
    This is a mock implementation for testing purposes. In a real implementation,
    this would use a vector database or similar technology to perform semantic search.
    
    Args:
        query: Natural language query.
        repo_path: Path to the Git repository to search.
        save_dir: Directory to save/load the search index.
        extensions: List of file extensions to include (e.g. [".py", ".js"]).
        k: Number of results to return.
        remove_duplicates: Whether to remove duplicate file results.
        min_score: Minimum score threshold to filter out low-quality matches.
        
    Returns:
        Dictionary with status and results.
    """
    # Default extensions if none provided
    if extensions is None:
        extensions = [".py"]
    
    # Default save directory if none provided
    if save_dir is None:
        save_dir = os.path.join(repo_path, '.code_search_index')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Find all files with the specified extensions
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(f"{repo_path}/**/*{ext}", recursive=True))
        
        # Mock results for testing
        results = []
        for i, file_path in enumerate(all_files[:k]):
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Truncate content if too long
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                # Calculate mock score (in a real implementation, this would be based on semantic similarity)
                score = 0.9 - (i * 0.05)  # Mock scores from 0.9 down
                
                # Skip if score is below threshold
                if score < min_score:
                    continue
                
                # Add to results
                results.append({
                    "file": os.path.relpath(file_path, repo_path),
                    "score": score,
                    "content": content
                })
            except Exception as e:
                # Skip files that can't be read
                continue
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during code search: {str(e)}"
        }