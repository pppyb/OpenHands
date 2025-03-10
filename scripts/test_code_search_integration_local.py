# # File: /workspace/OpenHands/scripts/test_code_search_integration.py

# import os
# import sys
# import asyncio
# from pathlib import Path

# # Add project root directory to Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from openhands.events.action.code_search import CodeSearchAction
# from openhands.events.observation.code_search import CodeSearchObservation
# from openhands.events.observation.error import ErrorObservation
# from openhands.runtime.action_execution_server import ActionExecutor
#!/usr/bin/env python3
"""Direct test for code search functionality without using ActionExecutor."""

import os
import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openhands_aci.tools.code_search_tool import code_search_tool
from openhands.events.action.code_search import CodeSearchAction



async def execute_code_search(action):
    """Execute code search action using ActionExecutor."""
    # Create ActionExecutor instance
    executor = ActionExecutor(
        plugins_to_load=[],
        work_dir=action.repo_path,
        username="openhands",
        user_id=1000,
        browsergym_eval_env=None
    )
    
    # Initialize ActionExecutor
    await executor.initialize()
    
    try:
        # Execute code search action
        observation = await executor.code_search(action)
        return observation
    finally:
        # Close ActionExecutor
        executor.close()


# def main():
#     """Test integration of code search functionality."""
#     # Use current directory as test repository
#     repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
#     # Create action
#     action = CodeSearchAction(
#         query="code search functionality",
#         repo_path=repo_path,
#         extensions=['.py'],
#         k=3
#     )
    
#     print(f"Search query: {action.query}")
#     print(f"Repository: {action.repo_path}")
#     print(f"Extensions: {', '.join(action.extensions)}")
#     print("-" * 80)
    
#     # Execute action
#     observation = asyncio.run(execute_code_search(action))
    
#     # Print results
#     if isinstance(observation, CodeSearchObservation):
#         print(f"Found {len(observation.results)} results:")
#         for i, result in enumerate(observation.results, 1):
#             print(f"\nResult {i}: {result['file']} (Score: {result['score']})")
#             print("-" * 40)
#             print(result['content'])
#     elif isinstance(observation, ErrorObservation):
#         print(f"Error: {observation.error}")
#     else:
#         print(f"Unknown observation type: {type(observation)}")


# if __name__ == "__main__":
#     main()
def main():
    """Test code search functionality directly."""
    # Use current directory as test repository
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create action (just for parameter organization)
    action = CodeSearchAction(
        query="code search functionality",
        repo_path=repo_path,
        extensions=['.py'],
        k=3
    )
    
    print(f"Search query: {action.query}")
    print(f"Repository: {action.repo_path}")
    print(f"Extensions: {', '.join(action.extensions)}")
    print("-" * 80)
    
    # Execute code search directly
    result = code_search_tool(
        query=action.query,
        repo_path=action.repo_path,
        extensions=action.extensions,
        k=action.k,
        remove_duplicates=action.remove_duplicates,
        min_score=action.min_score
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