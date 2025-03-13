"""
End-to-end integration test for code search functionality with real LLM API calls.

This test verifies that the CodeActAgent can correctly understand user queries
related to code search and choose the appropriate tool.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest
from git import Repo

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.controller.state.state import State
from openhands.events.action import CodeSearchAction, Action, MessageAction
from openhands.llm.llm import LLM


@pytest.fixture
def test_repo():
    """Create a temporary git repository with some test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize git repo
        repo = Repo.init(temp_dir)

        # Create test files
        files = {
            'main.py': 'def hello():\n    print("Hello, World!")',
            'utils/helper.py': 'def add(a, b):\n    """Add two numbers and return the result."""\n    return a + b',
            'auth/api_keys.py': 'def get_api_key():\n    """Get API key from environment."""\n    return os.environ.get("API_KEY")',
            'README.md': '# Test Repository\n This is a test.',
        }

        for path, content in files.items():
            file_path = Path(temp_dir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

        # Add and commit files
        repo.git.add('*')
        repo.git.commit('-m', 'Initial commit')

        yield temp_dir


def get_all_actions_from_state(state: State) -> List[Action]:
    """Extract all actions from a state."""
    actions = []
    for event in state.events:
        if hasattr(event, 'action') and event.action is not None:
            actions.append(event.action)
    return actions


def contains_code_search_action(actions: List[Action]) -> bool:
    """Check if the list of actions contains a CodeSearchAction."""
    return any(isinstance(action, CodeSearchAction) for action in actions)


def get_code_search_results(actions: List[Action]) -> Dict[str, Any]:
    """Get the results of the first CodeSearchAction in the list."""
    for action in actions:
        if isinstance(action, CodeSearchAction):
            return action.execute()
    return {"status": "error", "message": "No CodeSearchAction found"}


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_code_search_with_explicit_query(test_repo):
    """
    Test that the agent correctly identifies an explicit code search query
    and uses the CodeSearchAction.
    """
    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a LLM with default config
    from openhands.core.config.llm_config import LLMConfig
    llm = LLM(config=LLMConfig())
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a state with an explicit code search query
    state = State()
    # Add a message action to the state
    from openhands.events.event import EventSource
    state.history.append(
        MessageAction(
            content=f"Search for code that handles API keys in the repository at {test_repo}",
            _source=EventSource.USER
        )
    )
    
    # Run the agent for multiple steps to complete the task
    max_steps = 5
    for _ in range(max_steps):
        action = agent.step(state)
        if action is None:
            break
        
        # Execute the action and add it to the state
        state.add_action(action)
        
        # If we found a CodeSearchAction, we're done
        if isinstance(action, CodeSearchAction):
            break
    
    # Get all actions from the state
    actions = get_all_actions_from_state(state)
    
    # Verify that a CodeSearchAction was used
    assert contains_code_search_action(actions), "The agent should have used a CodeSearchAction"
    
    # Get the results of the CodeSearchAction
    results = get_code_search_results(actions)
    
    # Verify that the search was successful
    assert results["status"] == "success", f"Code search failed: {results.get('message', 'Unknown error')}"
    
    # Verify that the search found the API keys file
    found_api_keys = False
    for result in results.get("results", []):
        if "api_key" in result.get("content", "").lower():
            found_api_keys = True
            break
    
    assert found_api_keys, "The search should have found the API keys file"


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_code_search_with_implicit_query(test_repo):
    """
    Test that the agent correctly identifies an implicit code search query
    and uses the CodeSearchAction.
    """
    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a LLM with default config
    from openhands.core.config.llm_config import LLMConfig
    llm = LLM(config=LLMConfig())
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a state with an implicit code search query
    state = State()
    # Add a message action to the state
    from openhands.events.event import EventSource
    state.history.append(
        MessageAction(
            content=f"I need to understand how API keys are handled in the codebase at {test_repo}. Can you help me find the relevant code?",
            _source=EventSource.USER
        )
    )
    
    # Run the agent for multiple steps to complete the task
    max_steps = 5
    for _ in range(max_steps):
        action = agent.step(state)
        if action is None:
            break
        
        # Execute the action and add it to the state
        state.add_action(action)
        
        # If we found a CodeSearchAction, we're done
        if isinstance(action, CodeSearchAction):
            break
    
    # Get all actions from the state
    actions = get_all_actions_from_state(state)
    
    # Verify that a CodeSearchAction was used
    assert contains_code_search_action(actions), "The agent should have used a CodeSearchAction"
    
    # Get the results of the CodeSearchAction
    results = get_code_search_results(actions)
    
    # Verify that the search was successful
    assert results["status"] == "success", f"Code search failed: {results.get('message', 'Unknown error')}"
    
    # Verify that the search found the API keys file
    found_api_keys = False
    for result in results.get("results", []):
        if "api_key" in result.get("content", "").lower():
            found_api_keys = True
            break
    
    assert found_api_keys, "The search should have found the API keys file"


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_code_search_with_multi_intent_query(test_repo):
    """
    Test that the agent correctly handles a query with multiple intents,
    including code search.
    """
    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a LLM with default config
    from openhands.core.config.llm_config import LLMConfig
    llm = LLM(config=LLMConfig())
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a state with a multi-intent query
    state = State()
    # Add a message action to the state
    from openhands.events.event import EventSource
    state.history.append(
        MessageAction(
            content=f"I need to understand the codebase at {test_repo}. First, can you find functions that add numbers? Then, tell me about the overall structure of the repository.",
            _source=EventSource.USER
        )
    )
    
    # Run the agent for multiple steps to complete the task
    max_steps = 10
    for _ in range(max_steps):
        action = agent.step(state)
        if action is None:
            break
        
        # Execute the action and add it to the state
        state.add_action(action)
        
        # If we've run for enough steps, stop
        if len(get_all_actions_from_state(state)) >= 3:
            break
    
    # Get all actions from the state
    actions = get_all_actions_from_state(state)
    
    # Verify that a CodeSearchAction was used
    assert contains_code_search_action(actions), "The agent should have used a CodeSearchAction"
    
    # Get the results of the CodeSearchAction
    results = get_code_search_results(actions)
    
    # Verify that the search was successful
    assert results["status"] == "success", f"Code search failed: {results.get('message', 'Unknown error')}"
    
    # Verify that the search found the add function
    found_add_function = False
    for result in results.get("results", []):
        if "add" in result.get("content", "").lower() and "return a + b" in result.get("content", "").lower():
            found_add_function = True
            break
    
    assert found_add_function, "The search should have found the add function"