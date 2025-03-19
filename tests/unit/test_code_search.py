from unittest.mock import Mock, patch

import pytest
from litellm import ChatCompletionMessageToolCall, Choices, Message, ModelResponse

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.function_calling import (
    get_tools,
    response_to_actions,
)
from openhands.agenthub.codeact_agent.tools.code_search import CodeSearchTool
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import FunctionCallNotExistsError, FunctionCallValidationError
from openhands.events.action import (
    CodeSearchAction,
    MessageAction,
)
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.llm.llm import LLM


@pytest.fixture
def agent() -> CodeActAgent:
    config = AgentConfig()
    config.codeact_enable_code_search = True
    agent = CodeActAgent(llm=LLM(LLMConfig()), config=config)
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 1000
    agent.llm.config.model = "mock_model"
    return agent


@pytest.fixture
def mock_state() -> State:
    state = Mock(spec=State)
    state.history = []
    state.extra_data = {}
    return state


def test_get_tools_with_code_search():
    """Test that code_search tool is included when enabled."""
    # Test with code search enabled
    tools = get_tools(
        codeact_enable_browsing=True,
        codeact_enable_jupyter=True,
        codeact_enable_llm_editor=True,
        codeact_enable_code_search=True,
    )
    tool_names = [tool['function']['name'] for tool in tools]
    assert 'code_search' in tool_names

    # Test with code search disabled
    tools = get_tools(
        codeact_enable_browsing=True,
        codeact_enable_jupyter=True,
        codeact_enable_llm_editor=True,
        codeact_enable_code_search=False,
    )
    tool_names = [tool['function']['name'] for tool in tools]
    assert 'code_search' not in tool_names


def test_code_search_tool_definition():
    """Test that the CodeSearchTool is defined correctly."""
    assert CodeSearchTool['type'] == 'function'
    assert CodeSearchTool['function']['name'] == 'code_search'
    assert 'query' in CodeSearchTool['function']['parameters']['properties']
    assert 'repo_path' in CodeSearchTool['function']['parameters']['properties']
    assert 'extensions' in CodeSearchTool['function']['parameters']['properties']
    assert 'k' in CodeSearchTool['function']['parameters']['properties']
    assert CodeSearchTool['function']['parameters']['required'] == ['query']


def test_response_to_actions_code_search():
    """Test that code_search tool calls are converted to CodeSearchAction."""
    # Create a mock response with a code_search tool call
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message=Message(
                    content='Let me search the code for that',
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='tool_call_10',
                            function={
                                'name': 'code_search',
                                'arguments': '{"query": "function that handles HTTP requests", "repo_path": "/path/to/repo", "extensions": [".py", ".js"], "k": 5}',
                            },
                            type='function',
                        )
                    ],
                    role='assistant',
                ),
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Convert the response to actions
    actions = response_to_actions(mock_response)
    
    # Verify the result
    assert len(actions) == 1
    assert isinstance(actions[0], CodeSearchAction)
    assert actions[0].query == 'function that handles HTTP requests'
    assert actions[0].repo_path == '/path/to/repo'
    assert actions[0].extensions == ['.py', '.js']
    assert actions[0].k == 5


def test_response_to_actions_code_search_missing_query():
    """Test that code_search tool calls without a query raise an error."""
    # Create a mock response with a code_search tool call missing the query
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message=Message(
                    content='Let me search the code',
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='tool_call_10',
                            function={
                                'name': 'code_search',
                                'arguments': '{"repo_path": "/path/to/repo"}',
                            },
                            type='function',
                        )
                    ],
                    role='assistant',
                ),
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Verify that an error is raised
    with pytest.raises(FunctionCallValidationError):
        response_to_actions(mock_response)


def test_agent_tools_include_code_search(agent: CodeActAgent):
    """Test that the agent includes the code_search tool when enabled."""
    # Verify that the code_search tool is included in the agent's tools
    tool_names = [tool['function']['name'] for tool in agent.tools]
    assert 'code_search' in tool_names


@patch('openhands.runtime.search_engine.code_search.code_search')
def test_agent_step_with_code_search(mock_code_search, agent: CodeActAgent, mock_state: State):
    """Test that the agent can execute a code_search action."""
    # Mock the code_search function to return a CodeSearchObservation
    mock_results = [
        {
            "file": "file1.py",
            "score": 0.95,
            "content": "def handle_http_request():\n    pass",
        }
    ]
    
    # Create a CodeSearchObservation with content parameter
    content_str = "Found 1 result for query: 'function that handles HTTP requests'"
    obs = CodeSearchObservation(
        query="function that handles HTTP requests",
        results=mock_results,
        status="success",
        content=content_str
    )
    mock_code_search.return_value = obs

    # Create a CodeSearchAction
    action = CodeSearchAction(
        query="function that handles HTTP requests",
        repo_path="/path/to/repo",
    )

    # Add the action to the agent's pending actions
    agent.pending_actions.append(action)

    # Call step to execute the action
    result = agent.step(mock_state)

    # Verify that the action was executed
    assert result == action
    assert len(agent.pending_actions) == 0


@patch('openhands.runtime.search_engine.code_search.code_search')
def test_agent_handles_code_search_response(mock_code_search, agent: CodeActAgent, mock_state: State):
    """Test that the agent can handle a response from a code_search action."""
    # Mock the LLM response to include a code_search tool call
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message=Message(
                    content='Let me search the code for that',
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id='tool_call_10',
                            function={
                                'name': 'code_search',
                                'arguments': '{"query": "function that handles HTTP requests", "repo_path": "/path/to/repo"}',
                            },
                            type='function',
                        )
                    ],
                    role='assistant',
                ),
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Mock the LLM to return the response
    agent.llm.completion = Mock(return_value=mock_response)
    agent.llm.is_function_calling_active = Mock(return_value=True)
    agent.llm.is_caching_prompt_active = Mock(return_value=False)

    # Mock the code_search function to return a CodeSearchObservation
    mock_results = [
        {
            "file": "file1.py",
            "score": 0.95,
            "content": "def handle_http_request():\n    pass",
        }
    ]
    
    # Create a CodeSearchObservation with content parameter
    content_str = "Found 1 result for query: 'function that handles HTTP requests'"
    obs = CodeSearchObservation(
        query="function that handles HTTP requests",
        results=mock_results,
        status="success",
        content=content_str
    )
    mock_code_search.return_value = obs

    # Set up the state
    mock_state.latest_user_message = None
    mock_state.latest_user_message_id = None
    mock_state.latest_user_message_timestamp = None
    mock_state.latest_user_message_cause = None
    mock_state.latest_user_message_timeout = None
    mock_state.latest_user_message_llm_metrics = None
    mock_state.latest_user_message_tool_call_metadata = None

    # Call step to generate and execute the action
    action = agent.step(mock_state)

    # Verify that a CodeSearchAction was created
    assert isinstance(action, CodeSearchAction)
    assert action.query == 'function that handles HTTP requests'
    assert action.repo_path == '/path/to/repo'

    # Add the observation to the history
    observation = CodeSearchObservation(
        query="function that handles HTTP requests",
        results=mock_results,
        status="success",
        content=content_str
    )
    mock_state.history = [action, observation]

    # Mock a follow-up response
    follow_up_response = ModelResponse(
        id='mock_id_2',
        choices=[
            Choices(
                message=Message(
                    content='I found a function that handles HTTP requests in file1.py',
                    tool_calls=[],
                    role='assistant',
                ),
                index=0,
                finish_reason='stop',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )
    agent.llm.completion = Mock(return_value=follow_up_response)

    # Call step again to process the observation
    follow_up_action = agent.step(mock_state)

    # Verify that a MessageAction was created with the expected content
    assert isinstance(follow_up_action, MessageAction)
    assert 'I found a function that handles HTTP requests in file1.py' in follow_up_action.content