#!/usr/bin/env python3
"""
Simple direct test for code search functionality in OpenHands.

This script tests the code search functionality directly using the code_search_tool,
bypassing the ActionExecutor to avoid permission issues.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components
from openhands_aci.tools.code_search_tool import code_search_tool

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_direct_code_search(repo_path, query, extensions=None, k=5, min_score=0.5):
    """
    Execute a direct code search using the code_search_tool function.
    
    This function bypasses the ActionExecutor to avoid permission issues.
    
    Args:
        repo_path: Path to the repository to search
        query: Search query
        extensions: List of file extensions to search
        k: Number of results to return
        min_score: Minimum score threshold
    """
    logger.info(f"Executing direct code search in repository: {repo_path}")
    logger.info(f"Query: {query}")
    
    try:
        # Execute code search directly
        result = code_search_tool(
            query=query,
            repo_path=repo_path,
            extensions=extensions or [".py"],
            k=k,
            remove_duplicates=True,
            min_score=min_score
        )
        
        # Process the result
        if result["status"] == "success":
            logger.info(f"Search successful with {len(result['results'])} results")
            
            # Print the results
            print("\n" + "="*80)
            print(f"CODE SEARCH RESULTS FOR: '{query}'")
            print("="*80)
            
            for i, res in enumerate(result["results"], 1):
                print(f"\nResult {i}: {res['file']} (Score: {res['score']:.3f})")
                print("-" * 60)
                
                # Truncate content if too long
                content = res['content']
                if len(content) > 500:
                    content = content[:500] + "...\n[content truncated]"
                print(content)
            
            return result
        else:
            logger.error(f"Search failed: {result.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logger.exception(f"Error executing code search: {e}")
        return None


def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description='Test direct code search functionality')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
    parser.add_argument('--query', default="code search functionality", help='Search query')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='File extensions to search')
    parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum score threshold')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the test
    execute_direct_code_search(
        repo_path=args.repo,
        query=args.query,
        extensions=args.extensions,
        k=args.results,
        min_score=args.min_score
    )


if __name__ == "__main__":
    main()