#!/usr/bin/env python3
"""
Test script for RAG Code Search in OpenHands
"""

import os
import sys

from openhands.runtime.plugins.agent_skills.code_search.tool import code_search_tool


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_rag_code_search.py <repo_path> <query>")
        sys.exit(1)

    repo_path = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "how to handle API requests"

    print(f"Searching for: '{query}' in repository: {repo_path}")
    
    result = code_search_tool(
        query=query,
        repo_path=repo_path,
        k=5,
        remove_duplicates=True,
        min_score=0.5
    )
    
    if result["status"] == "error":
        print(f"Error: {result['message']}")
        sys.exit(1)
        
    print(f"Found {len(result['results'])} results")
    
    if "formatted_output" in result:
        print(result["formatted_output"])
    else:
        for i, res in enumerate(result["results"], 1):
            print(f"\nResult {i}: {res['file']} (Similarity: {res['score']:.3f})")
            print("-" * 80)
            print(res["content"])
            print("-" * 80)


if __name__ == "__main__":
    main()