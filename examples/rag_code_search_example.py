#!/usr/bin/env python3
"""
RAG Code Search Example for OpenHands

This example demonstrates how to use the RAG code search functionality
from openhands-aci to search code in a repository.
"""

import os
import sys
from pathlib import Path

from openhands_aci.tools.code_search_tool import code_search_tool
from openhands_aci.editor import OHEditor


class CodeAssistant:
    """Example Code Assistant using RAG for code search"""

    def __init__(self, repo_path: str):
        """Initialize the code assistant.
        
        Args:
            repo_path: Path to the repository to search
        """
        self.repo_path = repo_path
        self.editor = OHEditor()
        self.index_dir = os.path.join(repo_path, ".code_search_index")
        self._ensure_index()

    def _ensure_index(self):
        """Ensure that the index exists"""
        index_exists = (
            os.path.exists(os.path.join(self.index_dir, "index.faiss"))
            and os.path.exists(os.path.join(self.index_dir, "documents.pkl"))
        )

        if not index_exists:
            print(f"Indexing repository: {self.repo_path}")
            code_search_tool(
                query="Initialize index",
                repo_path=self.repo_path,
                save_dir=self.index_dir,
                extensions=[".py", ".js", ".html", ".tsx", ".jsx"]
            )

    def search_code(self, query: str, k: int = 5):
        """Search code in the repository.
        
        Args:
            query: The search query
            k: Number of results to return
        """
        print(f"Searching: '{query}'")
        result = code_search_tool(
            query=query,
            save_dir=self.index_dir,
            k=k,
            remove_duplicates=True,
            min_score=0.5
        )

        if result["status"] == "error":
            print(f"Search error: {result['message']}")
            return

        print(f"\nFound {len(result['results'])} result(s):")
        for i, res in enumerate(result["results"], 1):
            print(f"\nResult {i}: {res['file']} (Similarity: {res['score']:.3f})")
            print("-" * 80)
            print(res["content"])
            print("-" * 80)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python rag_code_search_example.py <repo_path> [query]")
        sys.exit(1)

    repo_path = sys.argv[1]
    assistant = CodeAssistant(repo_path)

    if len(sys.argv) > 2:
        query = sys.argv[2]
        assistant.search_code(query)
    else:
        # Interactive mode
        while True:
            print("\nOptions:")
            print("1. Search code")
            print("2. Exit")
            choice = input("Please select (1-2): ")

            if choice == "1":
                query = input("Enter search query: ")
                k = input("Number of results (default 5): ")
                try:
                    k = int(k) if k else 5
                except ValueError:
                    k = 5
                assistant.search_code(query, k)
            elif choice == "2":
                break
            else:
                print("Invalid selection")


if __name__ == "__main__":
    main()