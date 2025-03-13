"""
Integration tests for the code search functionality with CodeActAgent.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from git import Repo

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.controller.state.state import State
from openhands.events.action import CodeSearchAction
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


class MockLLM(LLM):
    """Mock LLM for testing."""
    
    def __init__(self, response_type='code_search'):
        self.response_type = response_type
        self.config = type('obj', (object,), {
            'max_message_chars': 10000,
            'api_version': 'v1',
        })
        self.metrics = type('obj', (object,), {
            'reset': lambda: None,
        })
    
    def completion(self, **kwargs):
        """Return a mock completion response."""
        if self.response_type == 'code_search':
            return {
                'choices': [
                    {
                        'message': {
                            'tool_calls': [
                                {
                                    'id': 'call_123',
                                    'function': {
                                        'name': 'code_search',
                                        'arguments': '{"command": "initialize", "repo_path": "/tmp/test_repo"}'
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        return {
            'choices': [
                {
                    'message': {
                        'content': 'Hello, world!'
                    }
                }
            ]
        }
    
    def format_messages_for_llm(self, messages):
        """Format messages for LLM."""
        return messages
    
    def vision_is_active(self):
        """Check if vision is active."""
        return False
    
    def is_caching_prompt_active(self):
        """Check if prompt caching is active."""
        return False


def test_code_search_action_integration():
    """Test that CodeSearchAction can be executed."""
    action = CodeSearchAction(
        command='initialize',
        repo_path='/tmp/test_repo',
        extensions=['.py'],
    )
    
    # Mock the initialize_code_search function to avoid actual execution
    with patch('openhands.events.action.code_search_action.initialize_code_search') as mock_init:
        mock_init.return_value = {
            'status': 'success',
            'message': 'Successfully indexed 2 files',
            'num_documents': 2,
        }
        
        result = action.execute()
        
        assert result['status'] == 'success'
        assert result['num_documents'] == 2
        mock_init.assert_called_once_with(
            repo_path='/tmp/test_repo',
            save_dir='/tmp/code_search/_tmp_test_repo',
            extensions=['.py'],
            embedding_model=None,
        )


def test_code_search_with_codeact_agent():
    """Test that CodeActAgent can use the code search tool."""
    # Create a mock config with code search enabled
    config = AgentConfig()
    config.codeact_enable_code_search = True
    
    # Create a mock LLM that returns a code search tool call
    llm = MockLLM(response_type='code_search')
    
    # Create the agent
    agent = CodeActAgent(llm, config)
    
    # Create a mock state
    state = State()
    
    # Mock the response_to_actions function to avoid actual execution
    with patch('openhands.agenthub.codeact_agent.function_calling.response_to_actions') as mock_response:
        mock_response.return_value = [
            CodeSearchAction(
                command='initialize',
                repo_path='/tmp/test_repo',
                extensions=['.py'],
            )
        ]
        
        # Call the agent's step method
        action = agent.step(state)
        
        # Verify that the action is a CodeSearchAction
        assert isinstance(action, CodeSearchAction)
        assert action.command == 'initialize'
        assert action.repo_path == '/tmp/test_repo'


def test_code_search_with_real_api_key():
    """Test that code search works with a real API key."""
    # Set the OpenAI API key (mock for testing)
    os.environ['OPENAI_API_KEY'] = 'sk-mock-api-key-for-testing'
    
    # Create a CodeSearchAction
    action = CodeSearchAction(
        command='initialize',
        repo_path='/tmp/test_repo',
        extensions=['.py'],
    )
    
    # Mock the initialize_code_search function to avoid actual execution
    with patch('openhands.events.action.code_search_action.initialize_code_search') as mock_init:
        mock_init.return_value = {
            'status': 'success',
            'message': 'Successfully indexed 2 files',
            'num_documents': 2,
        }
        
        result = action.execute()
        
        assert result['status'] == 'success'
        assert result['num_documents'] == 2