from unittest.mock import Mock

import pytest
from litellm import ChatCompletionMessageToolCall, Choices, ModelResponse

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.function_calling import (
    BrowserTool,
    IPythonTool,
    LLMBasedFileEditTool,
    WebReadTool,
    get_tools,
    response_to_actions,
)
from openhands.agenthub.codeact_agent.tools.browser import (
    _BROWSER_DESCRIPTION,
    _BROWSER_TOOL_DESCRIPTION,
)
from openhands.agenthub.codeact_agent.tools.code_search import CodeSearchTool
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    CmdRunAction,
    CodeSearchAction,
    MessageAction,
)
from openhands.events.event import EventSource
from openhands.events.observation.commands import (
    CmdOutputObservation,
)
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm import LLM


@pytest.fixture
def agent() -> CodeActAgent:
    config = AgentConfig()
    agent = CodeActAgent(llm=LLM(LLMConfig()), config=config)
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 1000
    agent.llm.config.model = 'mock_model'
    return agent


@pytest.fixture
def code_search_agent() -> CodeActAgent:
    config = AgentConfig()
    config.codeact_enable_code_search = True
    agent = CodeActAgent(llm=LLM(LLMConfig()), config=config)
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 1000
    agent.llm.config.model = 'mock_model'
    return agent


@pytest.fixture
def mock_state() -> State:
    state = Mock(spec=State)
    state.history = []
    state.extra_data = {}

    return state


def test_reset(agent: CodeActAgent):
    # Add some state
    action = MessageAction(content='test')
    action._source = EventSource.AGENT
    agent.pending_actions.append(action)

    # Reset
    agent.reset()

    # Verify state is cleared
    assert len(agent.pending_actions) == 0


def test_step_with_pending_actions(agent: CodeActAgent):
    # Add a pending action
    pending_action = MessageAction(content='test')
    pending_action._source = EventSource.AGENT
    agent.pending_actions.append(pending_action)

    # Step should return the pending action
    result = agent.step(Mock())
    assert result == pending_action
    assert len(agent.pending_actions) == 0


def test_get_tools_default():
    tools = get_tools(
        codeact_enable_jupyter=True,
        codeact_enable_llm_editor=True,
        codeact_enable_browsing=True,
    )
    assert len(tools) > 0

    # Check required tools are present
    tool_names = [tool['function']['name'] for tool in tools]
    assert 'execute_bash' in tool_names
    assert 'execute_ipython_cell' in tool_names
    assert 'edit_file' in tool_names
    assert 'web_read' in tool_names


def test_get_tools_with_options():
    # Test with all options enabled
    tools = get_tools(
        codeact_enable_browsing=True,
        codeact_enable_jupyter=True,
        codeact_enable_llm_editor=True,
        codeact_enable_code_search=True,
    )
    tool_names = [tool['function']['name'] for tool in tools]
    assert 'browser' in tool_names
    assert 'execute_ipython_cell' in tool_names
    assert 'edit_file' in tool_names
    assert 'code_search' in tool_names

    # Test with all options disabled
    tools = get_tools(
        codeact_enable_browsing=False,
        codeact_enable_jupyter=False,
        codeact_enable_llm_editor=False,
        codeact_enable_code_search=True,
    )
    tool_names = [tool['function']['name'] for tool in tools]
    assert 'browser' not in tool_names
    assert 'execute_ipython_cell' not in tool_names
    assert 'edit_file' not in tool_names
    assert 'code_search' in tool_names


def test_get_tools_with_code_search():
    """Test that code_search tool is included when enabled."""
    # Monkey patch get_tools to accept codeact_enable_code_search parameter
    original_get_tools = get_tools

    def patched_get_tools(
        codeact_enable_browsing: bool = False,
        codeact_enable_llm_editor: bool = False,
        codeact_enable_jupyter: bool = False,
        codeact_enable_code_search: bool = False,
        **kwargs,
    ) -> list:
        tools = original_get_tools(
            codeact_enable_browsing=codeact_enable_browsing,
            codeact_enable_llm_editor=codeact_enable_llm_editor,
            codeact_enable_jupyter=codeact_enable_jupyter,
        )
        if codeact_enable_code_search:
            tools.append(CodeSearchTool)
        return tools

    # Apply the patch
    import openhands.agenthub.codeact_agent.function_calling

    original = openhands.agenthub.codeact_agent.function_calling.get_tools
    openhands.agenthub.codeact_agent.function_calling.get_tools = patched_get_tools

    try:
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
    finally:
        # Restore the original
        openhands.agenthub.codeact_agent.function_calling.get_tools = original


def test_ipython_tool():
    assert IPythonTool['type'] == 'function'
    assert IPythonTool['function']['name'] == 'execute_ipython_cell'
    assert 'code' in IPythonTool['function']['parameters']['properties']
    assert IPythonTool['function']['parameters']['required'] == ['code']


def test_llm_based_file_edit_tool():
    assert LLMBasedFileEditTool['type'] == 'function'
    assert LLMBasedFileEditTool['function']['name'] == 'edit_file'

    properties = LLMBasedFileEditTool['function']['parameters']['properties']
    assert 'path' in properties
    assert 'content' in properties
    assert 'start' in properties
    assert 'end' in properties

    assert LLMBasedFileEditTool['function']['parameters']['required'] == [
        'path',
        'content',
    ]


def test_web_read_tool():
    assert WebReadTool['type'] == 'function'
    assert WebReadTool['function']['name'] == 'web_read'
    assert 'url' in WebReadTool['function']['parameters']['properties']
    assert WebReadTool['function']['parameters']['required'] == ['url']


def test_code_search_tool():
    """Test that the CodeSearchTool is defined correctly."""
    assert CodeSearchTool['type'] == 'function'
    assert CodeSearchTool['function']['name'] == 'code_search'

    # Check properties
    properties = CodeSearchTool['function']['parameters']['properties']
    assert 'query' in properties
    assert 'repo_path' in properties
    assert 'extensions' in properties
    assert 'k' in properties

    # Check required parameters
    assert CodeSearchTool['function']['parameters']['required'] == ['query']

    # Check description
    assert 'description' in CodeSearchTool['function']
    assert len(CodeSearchTool['function']['description']) > 0


def test_browser_tool():
    assert BrowserTool['type'] == 'function'
    assert BrowserTool['function']['name'] == 'browser'
    assert 'code' in BrowserTool['function']['parameters']['properties']
    assert BrowserTool['function']['parameters']['required'] == ['code']
    # Check that the description includes all the functions
    description = _BROWSER_TOOL_DESCRIPTION
    assert 'goto(' in description
    assert 'go_back()' in description
    assert 'go_forward()' in description
    assert 'noop(' in description
    assert 'scroll(' in description
    assert 'fill(' in description
    assert 'select_option(' in description
    assert 'click(' in description
    assert 'dblclick(' in description
    assert 'hover(' in description
    assert 'press(' in description
    assert 'focus(' in description
    assert 'clear(' in description
    assert 'drag_and_drop(' in description
    assert 'upload_file(' in description

    # Test BrowserTool definition
    assert BrowserTool['type'] == 'function'
    assert BrowserTool['function']['name'] == 'browser'
    assert BrowserTool['function']['description'] == _BROWSER_DESCRIPTION
    assert BrowserTool['function']['parameters']['type'] == 'object'
    assert 'code' in BrowserTool['function']['parameters']['properties']
    assert BrowserTool['function']['parameters']['required'] == ['code']
    assert (
        BrowserTool['function']['parameters']['properties']['code']['type'] == 'string'
    )
    assert 'description' in BrowserTool['function']['parameters']['properties']['code']


def test_response_to_actions_code_search():
    """Test that code_search tool calls are converted to CodeSearchAction."""
    # Create a mock response with a code_search tool call
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message={
                    'content': 'Let me search the code for that',
                    'tool_calls': [
                        {
                            'id': 'tool_call_10',
                            'function': {
                                'name': 'code_search',
                                'arguments': '{"query": "function that handles HTTP requests", "repo_path": "/path/to/repo", "extensions": [".py", ".js"], "k": 5}',
                            },
                            'type': 'function',
                        }
                    ],
                    'role': 'assistant',
                },
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Monkey patch response_to_actions to handle code_search tool
    original_response_to_actions = response_to_actions

    def patched_response_to_actions(response: ModelResponse) -> list:
        actions = []
        assert len(response.choices) == 1, 'Only one choice is supported for now'
        choice = response.choices[0]
        assistant_msg = choice.message
        if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
            # Process each tool call to OpenHands action
            for i, tool_call in enumerate(assistant_msg.tool_calls):
                try:
                    import json

                    arguments = json.loads(tool_call.function.arguments)
                except json.decoder.JSONDecodeError as e:
                    raise RuntimeError(
                        f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                    ) from e

                # Handle code_search tool
                if tool_call.function.name == 'code_search':
                    if 'query' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "query" in tool call {tool_call.function.name}'
                        )
                    action = CodeSearchAction(
                        query=arguments['query'],
                        repo_path=arguments.get('repo_path'),
                        extensions=arguments.get('extensions'),
                        k=arguments.get('k', 5),
                    )
                    actions.append(action)
                    return actions

        # If no code_search tool call was found, use the original function
        return original_response_to_actions(response)

    # Apply the patch
    import openhands.agenthub.codeact_agent.function_calling

    original = openhands.agenthub.codeact_agent.function_calling.response_to_actions
    openhands.agenthub.codeact_agent.function_calling.response_to_actions = (
        patched_response_to_actions
    )

    try:
        # Convert the response to actions
        actions = response_to_actions(mock_response)

        # Verify the result
        assert len(actions) == 1
        assert isinstance(actions[0], CodeSearchAction)
        assert actions[0].query == 'function that handles HTTP requests'
        assert actions[0].repo_path == '/path/to/repo'
        assert actions[0].extensions == ['.py', '.js']
        assert actions[0].k == 5
    finally:
        # Restore the original
        openhands.agenthub.codeact_agent.function_calling.response_to_actions = original


def test_response_to_actions_code_search_missing_query():
    """Test that code_search tool calls without a query raise an error."""
    # Create a mock response with a code_search tool call missing the query
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message={
                    'content': 'Let me search the code',
                    'tool_calls': [
                        {
                            'id': 'tool_call_10',
                            'function': {
                                'name': 'code_search',
                                'arguments': '{"repo_path": "/path/to/repo"}',
                            },
                            'type': 'function',
                        }
                    ],
                    'role': 'assistant',
                },
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


def test_response_to_actions_invalid_tool():
    # Test response with invalid tool call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = 'Invalid tool'
    mock_response.choices[0].message.tool_calls = [Mock()]
    mock_response.choices[0].message.tool_calls[0].id = 'tool_call_10'
    mock_response.choices[0].message.tool_calls[0].function = Mock()
    mock_response.choices[0].message.tool_calls[0].function.name = 'invalid_tool'
    mock_response.choices[0].message.tool_calls[0].function.arguments = '{}'

    with pytest.raises(FunctionCallNotExistsError):
        response_to_actions(mock_response)


def test_step_with_no_pending_actions(mock_state: State):
    # Mock the LLM response
    mock_response = Mock()
    mock_response.id = 'mock_id'
    mock_response.total_calls_in_response = 1
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = 'Task completed'
    mock_response.choices[0].message.tool_calls = []

    mock_config = Mock()
    mock_config.model = 'mock_model'

    llm = Mock()
    llm.config = mock_config
    llm.completion = Mock(return_value=mock_response)
    llm.is_function_calling_active = Mock(return_value=True)  # Enable function calling
    llm.is_caching_prompt_active = Mock(return_value=False)

    # Create agent with mocked LLM
    config = AgentConfig()
    config.enable_prompt_extensions = False
    agent = CodeActAgent(llm=llm, config=config)

    # Test step with no pending actions
    mock_state.latest_user_message = None
    mock_state.latest_user_message_id = None
    mock_state.latest_user_message_timestamp = None
    mock_state.latest_user_message_cause = None
    mock_state.latest_user_message_timeout = None
    mock_state.latest_user_message_llm_metrics = None
    mock_state.latest_user_message_tool_call_metadata = None

    action = agent.step(mock_state)
    assert isinstance(action, MessageAction)
    assert action.content == 'Task completed'


def test_mismatched_tool_call_events(mock_state: State):
    """Tests that the agent can convert mismatched tool call events (i.e., an observation with no corresponding action) into messages."""
    agent = CodeActAgent(llm=LLM(LLMConfig()), config=AgentConfig())

    tool_call_metadata = Mock(
        spec=ToolCallMetadata,
        model_response=Mock(
            id='model_response_0',
            choices=[
                Mock(
                    message=Mock(
                        role='assistant',
                        content='',
                        tool_calls=[
                            Mock(spec=ChatCompletionMessageToolCall, id='tool_call_0')
                        ],
                    )
                )
            ],
        ),
        tool_call_id='tool_call_0',
        function_name='foo',
    )

    action = CmdRunAction('foo')
    action._source = 'agent'
    action.tool_call_metadata = tool_call_metadata

    observation = CmdOutputObservation(content='', command_id=0, command='foo')
    observation.tool_call_metadata = tool_call_metadata

    # When both events are provided, the agent should get three messages:
    # 1. The system message,
    # 2. The action message, and
    # 3. The observation message
    mock_state.history = [action, observation]
    messages = agent._get_messages(mock_state)
    assert len(messages) == 3

    # The same should hold if the events are presented out-of-order
    mock_state.history = [observation, action]
    messages = agent._get_messages(mock_state)
    assert len(messages) == 3

    # If only one of the two events is present, then we should just get the system message
    mock_state.history = [action]
    messages = agent._get_messages(mock_state)
    assert len(messages) == 1

    mock_state.history = [observation]
    messages = agent._get_messages(mock_state)
    assert len(messages) == 1


def test_agent_tools_include_code_search(code_search_agent: CodeActAgent):
    """Test that the agent includes the code_search tool when enabled."""
    # Verify that the code_search tool is included in the agent's tools
    tool_names = [tool['function']['name'] for tool in code_search_agent.tools]
    assert 'code_search' in tool_names


def test_agent_step_with_code_search(
    code_search_agent: CodeActAgent, mock_state: State
):
    """Test that the agent can execute a code_search action."""
    # Add a pending action
    action = CodeSearchAction(
        query='function that handles HTTP requests',
        repo_path='/path/to/repo',
    )
    code_search_agent.pending_actions.append(action)

    # Step should return the pending action
    result = code_search_agent.step(mock_state)
    assert result == action
    assert len(code_search_agent.pending_actions) == 0


def test_agent_step_generates_code_search_action(
    code_search_agent: CodeActAgent, mock_state: State
):
    """Test that the agent generates a code_search action from LLM response."""
    # Mock the LLM response
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message={
                    'content': 'Let me search the code for that',
                    'tool_calls': [
                        {
                            'id': 'tool_call_10',
                            'function': {
                                'name': 'code_search',
                                'arguments': '{"query": "function that handles HTTP requests", "repo_path": "/path/to/repo"}',
                            },
                            'type': 'function',
                        }
                    ],
                    'role': 'assistant',
                },
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Mock the LLM to return the response
    code_search_agent.llm.completion = Mock(return_value=mock_response)
    code_search_agent.llm.is_function_calling_active = Mock(return_value=True)
    code_search_agent.llm.is_caching_prompt_active = Mock(return_value=False)

    # Set up the state
    mock_state.latest_user_message = 'Find functions that handle HTTP requests'

    def patched_response_to_actions(response: ModelResponse) -> list:
        # Create a CodeSearchAction
        action = CodeSearchAction(
            query='function that handles HTTP requests',
            repo_path='/path/to/repo',
        )
        return [action]

    # Apply the patch
    import openhands.agenthub.codeact_agent.function_calling

    original = openhands.agenthub.codeact_agent.function_calling.response_to_actions
    openhands.agenthub.codeact_agent.function_calling.response_to_actions = (
        patched_response_to_actions
    )

    try:
        # Call step to generate the action
        result = code_search_agent.step(mock_state)

        # Verify that a CodeSearchAction was created
        assert isinstance(result, CodeSearchAction)
        assert result.query == 'function that handles HTTP requests'
        assert result.repo_path == '/path/to/repo'
    finally:
        # Restore the original
        openhands.agenthub.codeact_agent.function_calling.response_to_actions = original


def test_code_search_with_different_extensions(
    code_search_agent: CodeActAgent, mock_state: State
):
    """Test code search with different file extensions."""
    # Mock the LLM response to include a code_search tool call with multiple extensions
    mock_response = ModelResponse(
        id='mock_id',
        choices=[
            Choices(
                message={
                    'content': 'Let me search for JavaScript and TypeScript files',
                    'tool_calls': [
                        {
                            'id': 'tool_call_10',
                            'function': {
                                'name': 'code_search',
                                'arguments': '{"query": "React component", "repo_path": "/workspace", "extensions": [".js", ".jsx", ".ts", ".tsx"]}',
                            },
                            'type': 'function',
                        }
                    ],
                    'role': 'assistant',
                },
                index=0,
                finish_reason='tool_calls',
            )
        ],
        model='mock_model',
        usage={'total_tokens': 100},
    )

    # Mock the LLM to return the response
    code_search_agent.llm.completion = Mock(return_value=mock_response)
    code_search_agent.llm.is_function_calling_active = Mock(return_value=True)
    code_search_agent.llm.is_caching_prompt_active = Mock(return_value=False)

    # Set up the state
    mock_state.latest_user_message = 'Find React components in the codebase'

    def patched_response_to_actions(response: ModelResponse) -> list:
        # Parse the arguments
        import json

        arguments = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )

        # Create a CodeSearchAction
        action = CodeSearchAction(
            query=arguments['query'],
            repo_path=arguments.get('repo_path'),
            extensions=arguments.get('extensions'),
            k=arguments.get('k', 5),
        )
        return [action]

    # Apply the patch
    import openhands.agenthub.codeact_agent.function_calling

    original = openhands.agenthub.codeact_agent.function_calling.response_to_actions
    openhands.agenthub.codeact_agent.function_calling.response_to_actions = (
        patched_response_to_actions
    )

    try:
        # Call step to generate and execute the action
        result = code_search_agent.step(mock_state)

        # Verify that a CodeSearchAction was created with the correct extensions
        assert isinstance(result, CodeSearchAction)
        assert result.query == 'React component'
        assert result.repo_path == '/workspace'
        assert result.extensions == ['.js', '.jsx', '.ts', '.tsx']
        assert len(result.extensions) == 4
    finally:
        # Restore the original
        openhands.agenthub.codeact_agent.function_calling.response_to_actions = original


def test_enhance_messages_adds_newlines_between_consecutive_user_messages(
    agent: CodeActAgent,
):
    """Test that _enhance_messages adds newlines between consecutive user messages."""
    # Set up the prompt manager
    agent.prompt_manager = Mock()
    agent.prompt_manager.add_examples_to_initial_message = Mock()
    agent.prompt_manager.add_info_to_initial_message = Mock()
    agent.prompt_manager.enhance_message = Mock()

    # Create consecutive user messages with various content types
    messages = [
        # First user message with TextContent only
        Message(role='user', content=[TextContent(text='First user message')]),
        # Second user message with TextContent only - should get newlines added
        Message(role='user', content=[TextContent(text='Second user message')]),
        # Assistant message
        Message(role='assistant', content=[TextContent(text='Assistant response')]),
        # Third user message with TextContent only - shouldn't get newlines
        Message(role='user', content=[TextContent(text='Third user message')]),
        # Fourth user message with ImageContent first, TextContent second - should get newlines
        Message(
            role='user',
            content=[
                ImageContent(image_urls=['https://example.com/image.jpg']),
                TextContent(text='Fourth user message with image'),
            ],
        ),
        # Fifth user message with only ImageContent - no TextContent to modify
        Message(
            role='user',
            content=[
                ImageContent(image_urls=['https://example.com/another-image.jpg'])
            ],
        ),
    ]

    # Call _enhance_messages
    enhanced_messages = agent._enhance_messages(messages)

    # Verify newlines were added correctly
    assert enhanced_messages[1].content[0].text.startswith('\n\n')
    assert enhanced_messages[1].content[0].text == '\n\nSecond user message'

    # Third message follows assistant, so shouldn't have newlines
    assert not enhanced_messages[3].content[0].text.startswith('\n\n')
    assert enhanced_messages[3].content[0].text == 'Third user message'

    # Fourth message follows user, so should have newlines in its TextContent
    assert enhanced_messages[4].content[1].text.startswith('\n\n')
    assert enhanced_messages[4].content[1].text == '\n\nFourth user message with image'

    # Fifth message only has ImageContent, no TextContent to modify
    assert len(enhanced_messages[5].content) == 1
    assert isinstance(enhanced_messages[5].content[0], ImageContent)
