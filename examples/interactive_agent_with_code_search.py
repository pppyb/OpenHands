"""
Interactive example demonstrating the CodeActAgent with code search functionality.

This script creates a CodeActAgent with code search enabled and allows users
to interact with it through a command-line interface.
"""

import argparse
import os
import sys
from typing import Optional

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.controller.state.state import State
from openhands.events.action import Action, MessageAction
from openhands.llm.llm import LLM


def setup_agent() -> CodeActAgent:
    """Set up the CodeActAgent with code search enabled."""
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to use the code search functionality.")
        print("Example: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a LLM with default config
    from openhands.core.config.llm_config import LLMConfig
    llm = LLM(config=LLMConfig())
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    return agent


def run_interactive_session(agent: CodeActAgent, initial_message: Optional[str] = None):
    """Run an interactive session with the agent."""
    # Create a state
    state = State()
    
    # Add initial message if provided
    if initial_message:
        state.add_message(
            Message(
                role="user",
                content=[TextContent(text=initial_message)],
            )
        )
        print(f"User: {initial_message}")
    
    # Main interaction loop
    while True:
        # Get the next action from the agent
        action = agent.step(state)
        
        # Handle the action
        if action is None:
            print("Agent returned no action. Exiting.")
            break
        
        # Add the action to the state
        state.add_action(action)
        
        # Handle different types of actions
        if isinstance(action, MessageAction):
            # Display the agent's message
            print(f"Agent: {action.content}")
            
            # Get user input
            user_input = input("User: ")
            
            # Check if the user wants to exit
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            
            # Add the user's message to the state
            state.add_message(
                Message(
                    role="user",
                    content=[TextContent(text=user_input)],
                )
            )
        else:
            # For other actions, execute them and add the result to the state
            print(f"Agent is executing: {action.__class__.__name__}")
            try:
                result = action.execute()
                state.add_observation(result)
                
                # Print a summary of the result
                if hasattr(result, "content"):
                    print(f"Result: {result.content[:100]}...")
                elif isinstance(result, dict) and "status" in result:
                    if result["status"] == "success":
                        print(f"Success: {result.get('message', 'Operation completed successfully')}")
                        
                        # If this is a code search result, show a summary
                        if "results" in result:
                            print(f"Found {len(result['results'])} results:")
                            for i, doc in enumerate(result["results"][:3]):  # Show top 3 results
                                print(f"  {i+1}. {doc.get('path', 'Unknown')} (score: {doc.get('score', 0):.2f})")
                                content = doc.get("content", "")
                                print(f"     {content[:100]}..." if len(content) > 100 else f"     {content}")
                    else:
                        print(f"Error: {result.get('message', 'Unknown error')}")
                else:
                    print("Action completed.")
            except Exception as e:
                print(f"Error executing action: {e}")
                state.add_observation({"status": "error", "message": str(e)})


def main():
    """Run the interactive agent example."""
    parser = argparse.ArgumentParser(description="Interactive agent with code search functionality")
    parser.add_argument(
        "--message", 
        type=str, 
        help="Initial message to send to the agent",
        default=None
    )
    args = parser.parse_args()
    
    # Set up the agent
    agent = setup_agent()
    
    # Run the interactive session
    print("Welcome to the interactive agent with code search functionality!")
    print("You can ask the agent to search for code in a repository.")
    print("For example: 'Search for code that handles API keys in the repository at /path/to/repo'")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("-" * 80)
    
    run_interactive_session(agent, args.message)


if __name__ == "__main__":
    main()