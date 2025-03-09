"""
Code Search Plugin for OpenHands

This plugin provides a RAG-based code search functionality for OpenHands agents.
"""

import os
from typing import Any, Dict, List, Optional

from openhands_aci.tools.code_search_tool import code_search_tool


def search_code(
    query: str,
    repo_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    extensions: Optional[List[str]] = None,
    k: int = 5,
    remove_duplicates: bool = True,
    min_score: float = 0.5,
) -> Dict[str, Any]:
    """Search code in a repository using semantic search.

    This tool uses Retrieval Augmented Generation (RAG) to find relevant code
    based on natural language queries. It first indexes the repository (if needed)
    and then performs a semantic search.

    Args:
        query: The search query in natural language.
        repo_path: Path to the git repository to search (optional if save_dir exists).
        save_dir: Directory to save/load the search index (defaults to .code_search_index).
        extensions: List of file extensions to include (e.g. [".py", ".js"]).
        k: Number of results to return.
        remove_duplicates: Whether to remove duplicate file results.
        min_score: Minimum score threshold to filter out low-quality matches.

    Returns:
        Dictionary with status and search results.
    """
    # Set default save_dir if not provided
    if save_dir is None and repo_path is not None:
        save_dir = os.path.join(repo_path, ".code_search_index")
    elif save_dir is None:
        save_dir = os.path.join(os.getcwd(), ".code_search_index")

    # Set default extensions if not provided
    if extensions is None:
        extensions = [".py", ".js", ".html", ".tsx", ".jsx", ".ts", ".css", ".md"]

    # Call the code search tool
    return code_search_tool(
        query=query,
        repo_path=repo_path,
        save_dir=save_dir,
        extensions=extensions,
        k=k,
        remove_duplicates=remove_duplicates,
        min_score=min_score,
    )