#!/usr/bin/env python3
"""Script for testing code search functionality."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import os
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openhands_aci.tools.code_search_tool import code_search_tool

from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events.observation.error import ErrorObservation


async def test_code_search(repo_path, query, extensions=None, k=5, min_score=0.5):
    """Test code search functionality.
    
    Args:
        repo_path: Path to the repository to search.
        query: Search query.
        extensions: List of file extensions to search.
        k: Number of results to return.
        min_score: Minimum score threshold.
    """
    # Import ActionExecutor
    from openhands.runtime.action_execution_server import ActionExecutor
    
    # Create ActionExecutor instance with browsergym_eval_env parameter
    executor = ActionExecutor(
        plugins_to_load=[],
        work_dir=repo_path,
        username="openhands",
        user_id=1000,
        browsergym_eval_env=None  # Add the missing parameter
    )
    
    # Initialize ActionExecutor
    await executor.initialize()
    
    # Create code search action
    action = CodeSearchAction(
        query=query,
        repo_path=repo_path,
        extensions=extensions or [".py"],
        k=k,
        min_score=min_score
    )
    
    print(f"Search query: {action.query}")
    print(f"Repository: {action.repo_path}")
    print(f"Extensions: {', '.join(action.extensions)}")
    print(f"Max results: {action.k}")
    print(f"Min score: {action.min_score}")
    print("-" * 80)
    
    # Execute action
    observation = await executor.code_search(action)
    
    # Print results
    if isinstance(observation, CodeSearchObservation):
        print(f"Found {len(observation.results)} results:")
        for i, result in enumerate(observation.results, 1):
            print(f"\nResult {i}: {result['file']} (Score: {result['score']})")
            print("-" * 40)
            print(result['content'])
    elif isinstance(observation, ErrorObservation):
        print(f"Error: {observation.error}")
    else:
        print(f"Unknown observation type: {type(observation)}")
    
    # Close ActionExecutor
    executor.close()

def main():
    """Test integration of code search functionality."""
    # Use current directory as test repository
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define search parameters
    query = "code search functionality"
    extensions = ['.py']
    k = 3
    min_score = 0.5
    
    print(f"Search query: {query}")
    print(f"Repository: {repo_path}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Max results: {k}")
    print(f"Min score: {min_score}")
    print("-" * 80)
    
    # Execute code search
    result = code_search_tool(
        query=query,
        repo_path=repo_path,
        extensions=extensions,
        k=k,
        remove_duplicates=True,
        min_score=min_score
    )
    
    # Print results
    print(f"\nSearch status: {result['status']}")
    if result['status'] == 'success':
        print(f"Found {len(result['results'])} results:")
        for i, res in enumerate(result['results'], 1):
            print(f"\nResult {i}: {res['file']} (Score: {res['score']})")
            print("-" * 40)
            print(res['content'])
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()