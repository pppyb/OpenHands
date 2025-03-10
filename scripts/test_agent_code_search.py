#!/usr/bin/env python3
"""
Simple test for code search functionality in an OpenHands agent.

This script demonstrates how an OpenHands agent can use the RAG code search
functionality to understand and work with a codebase.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components
from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.runtime.action_execution_server import ActionExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def simulate_agent_code_search(repo_path, query, extensions=None, k=5):
    """
    Simulate how an agent would use code search functionality.
    
    This function demonstrates the process an agent would follow to:
    1. Create a code search action
    2. Execute the action
    3. Process the observation
    4. Use the results
    
    Args:
        repo_path: Path to the repository to search
        query: Search query
        extensions: List of file extensions to search
        k: Number of results to return
    """
    logger.info(f"Simulating agent code search in repository: {repo_path}")
    logger.info(f"Query: {query}")
    
    # Step 1: Agent creates a code search action
    action = CodeSearchAction(
        query=query,
        repo_path=repo_path,
        extensions=extensions or [".py"],
        k=k,
        thought="I need to understand the codebase better to complete this task."
    )
    
    logger.info("Agent created code search action")
    logger.info(f"Action details: {action}")
    
    # Step 2: Agent executes the action
    try:
        # Initialize ActionExecutor (in a real agent, this would be done once at startup)
        executor = ActionExecutor(
            plugins_to_load=[],
            work_dir=repo_path,
            username="openhands",
            user_id=1000,
            browsergym_eval_env=None
        )
        
        # Initialize ActionExecutor
        await executor.initialize()
        
        logger.info("Executing code search action...")
        
        # Execute the action
        observation = await executor.code_search(action)
        
        # Step 3: Agent processes the observation
        if isinstance(observation, CodeSearchObservation):
            logger.info(f"Received code search observation with {len(observation.results)} results")
            
            # Step 4: Agent uses the results
            logger.info("Agent analyzing search results...")
            
            # Print the results (in a real agent, this would be used for reasoning)
            print("\n" + "="*80)
            print(f"AGENT SEARCH RESULTS FOR: '{query}'")
            print("="*80)
            
            for i, result in enumerate(observation.results, 1):
                print(f"\nResult {i}: {result['file']} (Score: {result['score']:.3f})")
                print("-" * 60)
                
                # Truncate content if too long
                content = result['content']
                if len(content) > 500:
                    content = content[:500] + "...\n[content truncated]"
                print(content)
            
            # Simulate agent reasoning based on results
            print("\n" + "="*80)
            print("AGENT REASONING")
            print("="*80)
            print("Based on the search results, I can understand:")
            
            if observation.results:
                print(f"1. Found {len(observation.results)} relevant code snippets")
                print(f"2. The most relevant file is {observation.results[0]['file']}")
                print(f"3. The code appears to be related to {query}")
                print("4. I can use this information to complete the task")
            else:
                print("No relevant code found. I need to try a different approach.")
            
            return observation
        else:
            logger.error(f"Unexpected observation type: {type(observation)}")
            return None
    except Exception as e:
        logger.exception(f"Error executing code search: {e}")
        return None
    finally:
        # Clean up
        if 'executor' in locals():
            executor.close()


def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description='Test agent code search functionality')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
    parser.add_argument('--query', default="code search functionality", help='Search query')
    parser.add_argument('--extensions', nargs='+', default=['.py'], help='File extensions to search')
    parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    
    args = parser.parse_args()
    
    # Run the test
    import asyncio
    asyncio.run(simulate_agent_code_search(
        repo_path=args.repo,
        query=args.query,
        extensions=args.extensions,
        k=args.results
    ))


if __name__ == "__main__":
    main()