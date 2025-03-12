#!/usr/bin/env python3
"""
Simple real-world test for code search functionality in OpenHands.

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

# #!/usr/bin/env python3
# """
# Simple real-world test for code search functionality in OpenHands.

# This script tests the code search functionality directly using the ActionExecutor,
# which is what a real agent would use to execute code search actions.
# """

# import os
# import sys
# import argparse
# import logging
# import asyncio
# from pathlib import Path

# # Add project root directory to Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Import necessary components
# from openhands.events.action.code_search import CodeSearchAction
# from openhands.events.observation.code_search import CodeSearchObservation
# from openhands.runtime.action_execution_server import ActionExecutor

# # Configure logging
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# async def execute_real_code_search(repo_path, query, extensions=None, k=5):
#     """
#     Execute a real code search using the ActionExecutor.
    
#     This function demonstrates how a real agent would execute a code search action:
#     1. Create an ActionExecutor
#     2. Create a CodeSearchAction
#     3. Execute the action
#     4. Process the observation
    
#     Args:
#         repo_path: Path to the repository to search
#         query: Search query
#         extensions: List of file extensions to search
#         k: Number of results to return
#     """
#     logger.info(f"Executing real code search in repository: {repo_path}")
#     logger.info(f"Query: {query}")
    
#     try:
#         # Initialize ActionExecutor
#         executor = ActionExecutor(
#             plugins_to_load=[],
#             work_dir=repo_path,
#             username="openhands",
#             user_id=os.getuid(),  # Use current user ID to avoid permission issues
#             browsergym_eval_env=None
#         )
        
#         # Initialize ActionExecutor
#         await executor.initialize()
        
#         # Create a code search action
#         action = CodeSearchAction(
#             query=query,
#             repo_path=repo_path,
#             extensions=extensions or [".py"],
#             k=k,
#             thought="I need to understand the codebase better to complete this task."
#         )
        
#         logger.info("Created code search action")
#         logger.info(f"Action details: {action}")
        
#         # Execute the action
#         logger.info("Executing code search action...")
#         observation = await executor.code_search(action)
        
#         # Process the observation
#         if isinstance(observation, CodeSearchObservation):
#             logger.info(f"Received code search observation with {len(observation.results)} results")
            
#             # Print the results
#             print("\n" + "="*80)
#             print(f"CODE SEARCH RESULTS FOR: '{query}'")
#             print("="*80)
            
#             for i, result in enumerate(observation.results, 1):
#                 print(f"\nResult {i}: {result['file']} (Score: {result['score']:.3f})")
#                 print("-" * 60)
                
#                 # Truncate content if too long
#                 content = result['content']
#                 if len(content) > 500:
#                     content = content[:500] + "...\n[content truncated]"
#                 print(content)
            
#             return observation
#         else:
#             logger.error(f"Unexpected observation type: {type(observation)}")
#             return None
#     except Exception as e:
#         logger.exception(f"Error executing code search: {e}")
#         return None
#     finally:
#         # Clean up
#         if 'executor' in locals():
#             await executor.close()


# async def main():
#     """Main function to run the test."""
#     parser = argparse.ArgumentParser(description='Test real code search functionality')
#     parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
#     parser.add_argument('--query', default="code search functionality", help='Search query')
#     parser.add_argument('--extensions', nargs='+', default=['.py'], help='File extensions to search')
#     parser.add_argument('--results', type=int, default=5, help='Number of results to return')
#     parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
#     args = parser.parse_args()
    
#     # Set logging level
#     if args.verbose:
#         logging.getLogger().setLevel(logging.DEBUG)
    
#     # Run the test
#     await execute_real_code_search(
#         repo_path=args.repo,
#         query=args.query,
#         extensions=args.extensions,
#         k=args.results
#     )


# if __name__ == "__main__":
#     asyncio.run(main())