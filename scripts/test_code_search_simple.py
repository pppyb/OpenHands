#!/usr/bin/env python3
"""Simple script for testing code search functionality."""

import argparse
import os
import sys

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import code_search_tool function
from openhands_aci.tools.code_search_tool import code_search_tool


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple test for code search functionality')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='File extensions to search')
    parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum score threshold')
    
    args = parser.parse_args()
    
    print(f"Search query: {args.query}")
    print(f"Repository: {args.repo}")
    print(f"Extensions: {', '.join(args.extensions)}")
    print(f"Max results: {args.results}")
    print(f"Min score: {args.min_score}")
    print("-" * 80)
    
    # Execute code search
    result = code_search_tool(
        query=args.query,
        repo_path=args.repo,
        extensions=args.extensions,
        k=args.results,
        remove_duplicates=True,
        min_score=args.min_score
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