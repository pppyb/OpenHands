#!/usr/bin/env python3
"""Script for testing code search functionality."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    """Main function."""
    parser = argparse.ArgumentParser(description='Test code search functionality')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='File extensions to search')
    parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum score threshold')
    
    args = parser.parse_args()
    
    # Run test
    asyncio.run(test_code_search(
        repo_path=args.repo,
        query=args.query,
        extensions=args.extensions,
        k=args.results,
        min_score=args.min_score
    ))


if __name__ == "__main__":
    main()