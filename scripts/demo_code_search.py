#!/usr/bin/env python3
"""
Demo script for the code search functionality in OpenHands.

This script demonstrates how to use the code search functionality to search
for code in a repository using natural language queries.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from openhands.core.config.app_config import AppConfig
from openhands.core.config.code_search_config import CodeSearchConfig
from openhands.events.action.code_search import CodeSearchAction
from openhands.runtime.search_engine.code_search import code_search


def main():
    # Get the repository path (current directory by default)
    repo_path = os.getcwd()
    
    # Set up the save directory for the index
    save_dir = os.path.join(repo_path, ".code_search_index")
    
    # Enable code search in the configuration
    config = AppConfig()
    config.code_search = CodeSearchConfig(
        enable_code_search=True,
        embedding_model="BAAI/bge-base-en-v1.5",
        default_save_dir=".code_search_index",
        default_extensions=[".py"],
        default_results_count=5,
    )
    
    # Initialize the search index (only needed once)
    print(f"Indexing repository: {repo_path}")
    action = CodeSearchAction(
        query="initialize",  # Dummy query for initialization
        repo_path=repo_path,
        save_dir=save_dir,
        extensions=[".py"],  # Only index Python files
    )
    
    result = code_search(action)
    
    if isinstance(result, Exception):
        print(f"Error initializing code search: {result}")
        return
    
    print(f"Successfully indexed repository: {repo_path}")
    
    # Example searches
    queries = [
        "code that handles file editing",
        "function that runs shell commands",
        "code for browser automation",
        "utility functions for generating diffs",
    ]
    
    for query in queries:
        print(f"\n\nSearching for: '{query}'")
        action = CodeSearchAction(
            query=query,
            save_dir=save_dir,
            k=3  # Return top 3 results
        )
        
        result = code_search(action)
        
        if isinstance(result, Exception):
            print(f"Error searching code: {result}")
            continue
        
        print(result.content)


if __name__ == "__main__":
    main()