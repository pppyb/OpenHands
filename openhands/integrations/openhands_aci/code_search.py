"""
Integration with openhands-aci code search functionality.
"""

import os
from typing import Dict, List, Optional, Any

from openhands.core.logger import openhands_logger as logger
from openhands_aci.code_search.tools import initialize_code_search as aci_initialize_code_search
from openhands_aci.code_search.tools import search_code as aci_search_code


def initialize_code_search(
    repo_path: str,
    save_dir: str,
    extensions: Optional[List[str]] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize code search for a repository.

    Args:
        repo_path: Path to the git repository.
        save_dir: Directory to save the index.
        extensions: List of file extensions to include (e.g. ['.py', '.js']).
                    If None, include all files.
        embedding_model: Name or path of the sentence transformer model to use.
                         If None, will use the model specified in the EMBEDDING_MODEL environment variable.

    Returns:
        Dictionary with status and message.
    """
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable is not set. Code search may not work properly.")
    
    try:
        result = aci_initialize_code_search(
            repo_path=repo_path,
            save_dir=save_dir,
            extensions=extensions,
            embedding_model=embedding_model,
        )
        logger.info(f"Code search initialized: {result}")
        return result
    except Exception as e:
        logger.error(f"Error initializing code search: {e}")
        return {"status": "error", "message": f"Error initializing code search: {e}"}


def search_code(
    save_dir: str,
    query: str,
    k: int = 5,
) -> Dict[str, Any]:
    """Search code in an indexed repository.

    Args:
        save_dir: Directory where the index is saved.
        query: Search query.
        k: Number of results to return.

    Returns:
        Dictionary with status, message, and results.
    """
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable is not set. Code search may not work properly.")
    
    try:
        result = aci_search_code(
            save_dir=save_dir,
            query=query,
            k=k,
        )
        logger.info(f"Code search results: {len(result.get('results', []))} matches")
        return result
    except Exception as e:
        logger.error(f"Error searching code: {e}")
        return {"status": "error", "message": f"Error searching code: {e}"}