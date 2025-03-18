"""
End-to-end integration test for code search functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest
from git import Repo

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.controller.state.state import State
from openhands.events.action import CodeSearchAction, MessageAction
from openhands.llm.llm import LLM
# Remove OpenAILLM import as we're using LLM directly


@pytest.fixture
def test_repo():
    """Create a temporary git repository with some test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize git repo
        repo = Repo.init(temp_dir)

        # Create test files
        files = {
            'main.py': 'def hello():\n    print("Hello, World!")',
            'utils/helper.py': 'def add(a, b):\n    return a + b',
            'README.md': '# Test Repository\n This is a test.',
        }

        for path, content in files.items():
            file_path = Path(temp_dir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

        # Add and commit files
        repo.index.add('*')
        repo.index.commit('Initial commit')

        yield temp_dir


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_code_search_e2e(test_repo):
    """Test code search functionality end-to-end with a real API key."""
    # Create a config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a LLM with OpenAI config
    from openhands.core.config.llm_config import LLMConfig
    llm_config = LLMConfig()
    llm_config.custom_llm_provider = "openai"
    llm_config.model = "gpt-4o"
    llm = LLM(config=llm_config)
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a state with a user message asking to search for code
    state = State()
    # Add a message action to the state
    from openhands.events.event import EventSource
    message = MessageAction(
        content=f"Search for a function that adds two numbers in the repository at {test_repo}"
    )
    message._source = EventSource.USER
    state.history.append(message)
    
    # Execute the agent step
    action = agent.step(state)
    
    # Verify that the action is a CodeSearchAction or contains code search results
    assert action is not None
    
    # If the action is not a CodeSearchAction, it might be a message with the search results
    # or another action that leads to code search
    if isinstance(action, CodeSearchAction):
        # Execute the action to get the results
        result = action.execute()
        assert result['status'] == 'success', f"Code search failed: {result.get('message', 'Unknown error')}"
        
        # Check if this is an initialization result or a search result
        if 'results' in result:
            # This is a search result
            assert len(result.get('results', [])) > 0, "The search results should not be empty"
        else:
            # This is an initialization result
            assert 'num_documents' in result, "The initialization result should contain a 'num_documents' field"
            assert result['num_documents'] > 0, "The number of indexed documents should be greater than 0"
    else:
        # If not a CodeSearchAction, the agent might have chosen to initialize first
        # or to respond with a message
        # We'll add the action to the state and step again
        state.add_action(action)
        
        # Execute another step
        action = agent.step(state)
        
        # Now we should have a CodeSearchAction or results
        if isinstance(action, CodeSearchAction):
            result = action.execute()
            assert result['status'] == 'success'
            
            # Verify that the search results contain the add function
            found_add = False
            for doc in result.get('results', []):
                if 'add' in doc.get('content', ''):
                    found_add = True
                    break
            
            assert found_add, "The search results should contain the add function"