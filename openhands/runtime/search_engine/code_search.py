"""
Code search implementation for OpenHands.

This module provides functionality to search code in a repository using semantic search.
It uses the code search functionality from openhands-aci.
"""

import os
from typing import Dict, List, Optional, Any

from openhands.core.config.app_config import AppConfig
from openhands.events.action import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events.observation.error import ErrorObservation
from openhands.utils.tenacity_stop import stop_if_should_exit

import tenacity
from openhands_aci.code_search import initialize_code_search, search_code
from openhands_aci.code_search.tools import update_code_search


def return_error(retry_state: tenacity.RetryCallState):
    """Return an error observation when retries are exhausted."""
    return ErrorObservation('Failed to execute code search operation.')


@tenacity.retry(
    wait=tenacity.wait_exponential(min=2, max=10),
    stop=tenacity.stop_after_attempt(3) | stop_if_should_exit(),
    retry_error_callback=return_error,
)
def code_search(action: CodeSearchAction):
    """Execute a code search action.

    Args:
        action: The code search action to execute.

    Returns:
        A CodeSearchObservation or ErrorObservation.
    """
    # Get configuration
    config = AppConfig().code_search
    
    # Set default save_dir if not provided
    save_dir = action.save_dir or os.path.join(os.getcwd(), config.default_save_dir)
    
    # Set default extensions if not provided
    extensions = action.extensions or config.default_extensions
    
    # Check if we need to initialize or update the index
    if action.repo_path:
        repo_path = os.path.abspath(os.path.expanduser(action.repo_path))
        
        # Check if repository exists
        if not os.path.exists(repo_path):
            return ErrorObservation(
                content=f"Repository path does not exist: {repo_path}. Please provide a valid path."
            )
        
        # Check if index exists
        index_exists = (
            os.path.exists(os.path.join(save_dir, "index.faiss"))
            and os.path.exists(os.path.join(save_dir, "documents.pkl"))
        )
        
        if not index_exists:
            # Initialize the index
            result = initialize_code_search(
                repo_path=repo_path,
                save_dir=save_dir,
                extensions=extensions,
                embedding_model=config.embedding_model,
            )
            
            if result["status"] == "error":
                return ErrorObservation(content=result["message"])
            
            # Return initialization result
            return CodeSearchObservation(
                query=action.query,
                results=[],
                status="success",
                message=f"Successfully indexed {result.get('num_documents', 0)} files from {repo_path}",
                num_documents=result.get("num_documents", 0),
                repo_path=repo_path,
            )
        else:
            # Update the index
            result = update_code_search(
                repo_path=repo_path,
                save_dir=save_dir,
                extensions=extensions,
            )
            
            if result["status"] == "error":
                return ErrorObservation(content=result["message"])
    
    # Perform search
    search_result = search_code(
        save_dir=save_dir,
        query=action.query,
        k=action.k or config.default_results_count,
    )
    
    if search_result["status"] == "error":
        return ErrorObservation(content=search_result["message"])
    
    # Format results for better readability
    formatted_results = []
    for result in search_result["results"]:
        formatted_results.append({
            "file": result["path"],
            "score": round(result["score"], 3),
            "content": result["content"],
        })
    
    # Return search results
    return CodeSearchObservation(
        query=action.query,
        results=formatted_results,
        status="success",
        repo_path=action.repo_path,
    )