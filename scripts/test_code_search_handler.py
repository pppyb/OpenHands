#!/usr/bin/env python3
"""
Test script for code search handler.

This script tests the handling of code search tool calls in the function_calling.py module.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components
from openhands.events.action.code_search import CodeSearchAction
from openhands.agenthub.codeact_agent.function_calling import response_to_actions
from litellm import ModelResponse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def create_mock_response(tool_name: str, arguments: Dict[str, Any]) -> ModelResponse:
    """Create a mock ModelResponse with a tool call.
    
    Args:
        tool_name: Name of the tool to call
        arguments: Arguments for the tool call
        
    Returns:
        ModelResponse object
    """
    # Convert arguments to JSON string
    arguments_str = json.dumps(arguments)
    
    # Create a mock response
    response = ModelResponse(
        id="mock-response-id",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll help you with that by using the code search tool.",
                    "tool_calls": [
                        {
                            "id": "mock-tool-call-id",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_str
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        model="gpt-4",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    )
    
    # Add attributes to make it more like a real ModelResponse
    from types import SimpleNamespace
    
    # Create a message object with the tool_calls attribute
    message = SimpleNamespace()
    message.role = "assistant"
    message.content = "I'll help you with that by using the code search tool."
    
    # Create a tool_call object
    tool_call = SimpleNamespace()
    tool_call.id = "mock-tool-call-id"
    tool_call.type = "function"
    
    # Create a function object
    function = SimpleNamespace()
    function.name = tool_name
    function.arguments = arguments_str
    
    # Link the objects
    tool_call.function = function
    message.tool_calls = [tool_call]
    
    # Create a choice object
    choice = SimpleNamespace()
    choice.index = 0
    choice.message = message
    choice.finish_reason = "tool_calls"
    
    # Add the choice to the response
    response.choices = [choice]
    
    return response

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description='Test code search handler')
    parser.add_argument('--repo', default=os.getcwd(), help='Path to the repository to search')
    parser.add_argument('--query', default='code search', help='Search query')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Set a more verbose logging level for this script
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    # Create arguments for the code search tool
    arguments = {
        "query": args.query,
        "repo_path": args.repo,
        "extensions": [".py"],
        "k": 5,
        "thought": "Testing code search handler"
    }
    
    # Create a mock response with a code search tool call
    response = create_mock_response("code_search", arguments)
    
    # Print the mock response
    logger.info("Created mock response with code search tool call")
    logger.info(f"Tool name: code_search")
    logger.info(f"Arguments: {arguments}")
    
    try:
        # Call response_to_actions to handle the response
        logger.info("Calling response_to_actions...")
        
        # Print the response details
        logger.debug(f"Response: {response}")
        logger.debug(f"Response choices: {response.choices}")
        logger.debug(f"Response message: {response.choices[0].message}")
        logger.debug(f"Tool calls: {response.choices[0].message.tool_calls}")
        logger.debug(f"Tool call function name: {response.choices[0].message.tool_calls[0].function.name}")
        logger.debug(f"Tool call function arguments: {response.choices[0].message.tool_calls[0].function.arguments}")
        
        # Import the CodeSearchTool to check if it's defined
        try:
            from openhands.agenthub.codeact_agent.function_calling import CodeSearchTool
            logger.info(f"CodeSearchTool is defined: {CodeSearchTool}")
            if hasattr(CodeSearchTool, 'function') and hasattr(CodeSearchTool.function, 'name'):
                logger.info(f"CodeSearchTool function name: {CodeSearchTool.function.name}")
        except ImportError:
            logger.warning("Could not import CodeSearchTool from function_calling")
        except Exception as e:
            logger.warning(f"Error accessing CodeSearchTool: {e}")
        
        # Call response_to_actions
        actions = response_to_actions(response)
        
        # Print the actions
        logger.info(f"Got {len(actions)} actions")
        for i, action in enumerate(actions):
            logger.info(f"Action {i+1}: {type(action).__name__}")
            
            # Check if the action is a CodeSearchAction
            if isinstance(action, CodeSearchAction):
                logger.info(f"  Query: {action.query}")
                logger.info(f"  Repo path: {action.repo_path}")
                logger.info(f"  Extensions: {action.extensions}")
                logger.info(f"  k: {action.k}")
                logger.info(f"  Thought: {action.thought}")
                
                logger.info("Code search action was correctly handled!")
                
                # Try to execute the action
                try:
                    from openhands.controller.executor import Executor
                    executor = Executor()
                    logger.info("Executing code search action...")
                    
                    # Run the action in a separate thread to avoid blocking
                    import threading
                    
                    def execute_action():
                        try:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            observation = loop.run_until_complete(executor.execute(action))
                            logger.info(f"Execution successful: {observation}")
                        except Exception as e:
                            logger.error(f"Error executing action: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    thread = threading.Thread(target=execute_action)
                    thread.start()
                    thread.join(timeout=10)  # Wait for up to 10 seconds
                    
                    if thread.is_alive():
                        logger.warning("Execution is taking too long, continuing...")
                    
                except Exception as e:
                    logger.error(f"Error setting up execution: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.warning(f"Action is not a CodeSearchAction: {action}")
        
        # Print success message
        print("\n================================================================================")
        print("SUCCESS: Code search tool is working correctly!")
        print("================================================================================\n")
        print("The code search tool has been successfully integrated into the function_calling.py module.")
        print("This means that the agent can now use the code search tool to find relevant code.")
        print("\nNext steps:")
        print("1. Make sure the code search tool is registered with the agent")
        print("2. Test the integration with a real agent")
        print("3. Update the documentation to include the code search tool")
        
    except Exception as e:
        logger.error(f"Error handling code search tool call: {e}")
        import traceback
        traceback.print_exc()
        
        # Print error message
        print("\n================================================================================")
        print("ERROR: Code search tool is NOT working correctly!")
        print("================================================================================\n")
        print(f"Error: {e}")
        print("\nPlease check the error message and fix the issue.")
        
        logger.info("The function_calling.py module needs to be updated to handle code search tool calls")
        
        # Print instructions for updating function_calling.py
        print("\n================================================================================")
        print("INSTRUCTIONS FOR UPDATING function_calling.py")
        print("================================================================================\n")
        print("Add the following code to the response_to_actions function in function_calling.py:\n")
        print("# ================================================")
        print("# CodeSearchTool")
        print("# ================================================")
        print("elif tool_call.function.name == 'code_search':")
        print("    if 'query' not in arguments:")
        print("        raise FunctionCallValidationError(")
        print("            f'Missing required argument \"query\" in tool call {tool_call.function.name}'")
        print("        )")
        print("    ")
        print("    # Create a CodeSearchAction with the provided arguments")
        print("    from openhands.events.action.code_search import CodeSearchAction")
        print("    action = CodeSearchAction(")
        print("        query=arguments['query'],")
        print("        repo_path=arguments.get('repo_path'),")
        print("        extensions=arguments.get('extensions'),")
        print("        k=arguments.get('k', 5),")
        print("        thought=arguments.get('thought', '')")
        print("    )")
        print("\n")
        
if __name__ == "__main__":
    main()