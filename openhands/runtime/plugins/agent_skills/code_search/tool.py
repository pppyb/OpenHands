"""
Code Search Tool Definition for OpenHands
"""

from typing import Any, Dict, List, Optional

from openhands.runtime.plugins.agent_skills.code_search import search_code


def code_search_tool(
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
    result = search_code(
        query=query,
        repo_path=repo_path,
        save_dir=save_dir,
        extensions=extensions,
        k=k,
        remove_duplicates=remove_duplicates,
        min_score=min_score,
    )
    
    # Format the results for better display in the agent
    if result["status"] == "success" and result["results"]:
        formatted_output = f"Found {len(result['results'])} relevant code snippets:\n\n"
        
        for i, res in enumerate(result["results"], 1):
            formatted_output += f"Result {i}: {res['file']} (Similarity: {res['score']:.3f})\n"
            formatted_output += "-" * 80 + "\n"
            formatted_output += res["content"] + "\n"
            formatted_output += "-" * 80 + "\n\n"
            
        result["formatted_output"] = formatted_output
    
    return result