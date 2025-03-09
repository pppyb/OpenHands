#!/usr/bin/env python3
"""
Small test for RAG Code Search in OpenHands
"""

import os
import sys

from openhands_aci.tools.code_search_tool import code_search_tool


def main():
    """Main function"""
    # Use a smaller directory for testing
    repo_path = "/workspace/OpenHands/examples"
    query = "code search example"

    print(f"Searching for: '{query}' in directory: {repo_path}")
    
    result = code_search_tool(
        query=query,
        repo_path=repo_path,
        k=3,
        remove_duplicates=True,
        min_score=0.5
    )
    
    if result["status"] == "error":
        print(f"Error: {result['message']}")
        sys.exit(1)
        
    print(f"Found {len(result['results'])} results")
    
    for i, res in enumerate(result["results"], 1):
        print(f"\nResult {i}: {res['file']} (Similarity: {res['score']:.3f})")
        print("-" * 80)
        print(res["content"])
        print("-" * 80)


if __name__ == "__main__":
    main()