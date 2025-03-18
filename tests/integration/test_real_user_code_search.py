"""
End-to-end integration test for code search functionality with real user input and repository.
"""

import os
import pytest

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.controller.state.state import State
from openhands.events.action import CodeSearchAction, MessageAction
from openhands.llm.openai import OpenAILLM


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_real_user_input_and_repo():
    """
    Test code search functionality with real user input and the OpenHands repository itself.
    
    This test simulates a real user asking to search for code in the OpenHands repository.
    It verifies that the agent can understand the natural language query, initialize the
    code search, and return relevant results.
    """
    # Use the OpenHands repository itself as the test repository
    repo_path = "/workspace/OpenHands"
    
    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a real OpenAI LLM
    llm = OpenAILLM()
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a state with a user message asking to search for code
    # This simulates a real user input
    state = State()
    state.add_message(
        Message(
            role="user",
            content=[
                TextContent(
                    text=f"I need to find code in the OpenHands repository that handles API keys. "
                    f"Can you search the repository at {repo_path} and find files that deal with "
                    f"API key management or authentication?"
                )
            ],
        )
    )
    
    # Execute the agent step
    action = agent.step(state)
    
    # Verify that the action is not None
    assert action is not None
    
    # Process the action and continue the conversation
    max_steps = 5  # Limit the number of steps to avoid infinite loops
    step_count = 0
    found_code_search = False
    search_results = []
    
    while step_count < max_steps:
        step_count += 1
        
        # If the action is a CodeSearchAction, execute it and collect results
        if isinstance(action, CodeSearchAction):
            found_code_search = True
            result = action.execute()
            
            # Store the search results
            if result['status'] == 'success' and 'results' in result:
                search_results = result['results']
            
            # Add the action and its result to the state
            state.add_action(action)
            state.add_observation(result)
            
            # Get the next action
            action = agent.step(state)
        
        # If the action is a message, check if it contains search results
        elif isinstance(action, MessageAction):
            # Add the message to the state
            state.add_action(action)
            
            # If we've already found code search results, we can stop
            if found_code_search and search_results:
                break
            
            # Otherwise, continue the conversation
            action = agent.step(state)
        
        # For any other action, add it to the state and continue
        else:
            state.add_action(action)
            action = agent.step(state)
    
    # Verify that code search was performed
    assert found_code_search, "The agent should have performed a code search"
    
    # Verify that we found some results
    assert len(search_results) > 0, "The search should have returned some results"
    
    # Verify that the results are relevant to API keys
    relevant_results = False
    for result in search_results:
        content = result.get('content', '').lower()
        if 'api' in content and ('key' in content or 'token' in content or 'auth' in content):
            relevant_results = True
            break
    
    assert relevant_results, "The search results should be relevant to API keys"