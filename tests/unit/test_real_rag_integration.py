"""Test real RAG integration with code search."""

import os
import json
import unittest
from unittest.mock import patch, Mock, MagicMock

from openhands.agenthub.codeact_agent.function_calling import get_registered_tools, response_to_actions
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.schema.action import ActionType

from openhands_aci.rag.code_search import CodeSearchAction, CodeSearchObservation, execute_code_search
from openhands_aci.rag.function_calling import register_code_search_tools

from openhands.core.config import AgentConfig, LLMConfig
from openhands.controller.state.state import State
from openhands.llm.llm import LLM
from openhands.core.message import Message, TextContent
from openhands.events.action import MessageAction

class TestRealRagIntegration(unittest.TestCase):
    """Test real RAG integration with code search."""

    def setUp(self):
        """Set up the test."""
        # Register the code search tool
        register_code_search_tools(lambda tool: get_registered_tools().append(tool))
        
        # Set up a real OpenAI API key for testing
        os.environ["OPENAI_API_KEY"] = "enter your api key"

    def test_code_search_action_execution(self):
        """Test that code search action can be executed."""
        # Create a code search action
        action = CodeSearchAction(
            query="Find code related to file operations",
            repo_path="/workspace/OpenHands",
            extensions=[".py"],
            k=3
        )
        # Ensure mock mode is disabled
        if 'OPENHANDS_TEST_MOCK_MODE' in os.environ:
            del os.environ['OPENHANDS_TEST_MOCK_MODE']
        # # Set mock mode environment variable
        # os.environ['OPENHANDS_TEST_MOCK_MODE'] = 'true'

        try:
            # Execute the action
            observation = execute_code_search(action)
            
            # Check that the observation is a CodeSearchObservation
            self.assertIsInstance(observation, CodeSearchObservation)
            
            # Check that the observation has content
            self.assertIsNotNone(observation.content)
            
            # Check that the observation has results
            self.assertIsNotNone(observation.results)
        finally:
            # Clean up environment variable
            os.environ.pop('OPENHANDS_TEST_MOCK_MODE', None)
    def test_code_search_integration_with_agent(self):
        """Test code search integration with the CodeActAgent."""
        # Ensure mock mode is disabled
        if 'OPENHANDS_TEST_MOCK_MODE' in os.environ:
            del os.environ['OPENHANDS_TEST_MOCK_MODE']
        
        REPO_PATH = "/path/to/repo"

        try:
            # Create a real CodeActAgent with a real LLM

            config = AgentConfig()

            agent = CodeActAgent(llm=Mock(spec=LLM), config=config)

            # Use a real LLM with the provided API key
            real_llm = LLM(LLMConfig(
                model="gpt-3.5-turbo",  # Using a faster model for testing
                api_key=os.environ["OPENAI_API_KEY"]
            ))
            agent = CodeActAgent(llm=real_llm, config=config)
            # # Mock the prompt manager
            # agent.prompt_manager = MagicMock()
            # agent.prompt_manager.get_system_message = MagicMock(return_value="You are a helpful AI assistant.")
            # Create a real state
            state = State()
            state.extra_data = {}           




            # Add a user message to the state history with a specific query about file operations

            # Create a user message first
            user_message = MessageAction(
                content="Find code related to file operations in this repository"

            )
            user_message._source = "user"
        

            state.history = [user_message]

            # Create a method to get the last user message
            state.get_last_user_message = lambda: user_message



            with patch('openhands.agenthub.codeact_agent.function_calling.CODE_SEARCH_AVAILABLE', True):
                # This will use the real LLM to process the user's query
                action = agent.step(state)

                # # Verify that an action was returned
                # self.assertIsNotNone(action)

                # # If the action is a CodeSearchAction, verify it works
                # if isinstance(action, CodeSearchAction):
                #     # Verify the properties of the CodeSearchAction
                #     self.assertIsNotNone(action.query)
                #     self.assertIsNotNone(action.repo_path)

                # Execute the action to verify it works end-to-end
                observation = execute_code_search(action)

                # Verify the observation
                self.assertIsInstance(observation, CodeSearchObservation)
                self.assertIsNotNone(observation.content)
                self.assertIsNotNone(observation.results)
                self.assertGreaterEqual(len(observation.results), 1)
                # Print some details about the results for verification
                print(f"\nCode search found {len(observation.results)} results:")
                for i, result in enumerate(observation.results):
                    print(f"Result {i+1}: {result['file']} (Score: {result['score']:.2f})")
                    print(f"Content snippet: {result['content'][:100]}...\n")
            # Create a CodeActAgent
                config = AgentConfig()
                agent = CodeActAgent(llm=Mock(spec=LLM), config=config)
                # Check that code search is in the tools
                self.assertTrue(any(tool['function']['name'] == 'code_search' for tool in agent.tools))



        finally:
        # Clean up environment variable
            if 'OPENHANDS_TEST_MOCK_MODE' in os.environ:
                del os.environ['OPENHANDS_TEST_MOCK_MODE']

                
    def test_code_search_tool_registration(self):
        """Test that code search tool is registered."""
        # Check that the code search tool is registered
        tools = get_registered_tools()
        code_search_tools = [tool for tool in tools if tool["name"] == "code_search"]
        
        # There should be at least one code search tool
        self.assertGreaterEqual(len(code_search_tools), 1)
        
        # Check that the tool has the correct properties
        tool = code_search_tools[0]
        self.assertEqual(tool["name"], "code_search")
        self.assertEqual(tool["action_type"], ActionType.CODE_SEARCH)
        self.assertIn("function", tool)
        self.assertIn("executor", tool)

    def test_end_to_end_code_search_with_real_llm(self):
            """
            Test the complete end-to-end flow:
            1. Create a real user message with a code search query
            2. Let a real LLM process this message and decide to use the code search tool
            3. Verify the entire process works correctly
            """
            # Ensure mock mode is disabled
            if 'OPENHANDS_TEST_MOCK_MODE' in os.environ:
                del os.environ['OPENHANDS_TEST_MOCK_MODE']

            try:
                # Set the real API key
                os.environ["OPENAI_API_KEY"] = "enter your api key"

                # Create a real CodeActAgent with a real LLM
                config = AgentConfig()
                # Use a real LLM with the provided API key
                real_llm = LLM(LLMConfig(
                    model="gpt-4o",  # Using a more capable model to ensure it understands the task
                    api_key=os.environ["OPENAI_API_KEY"]
                ))
                agent = CodeActAgent(llm=real_llm, config=config)

                # Create a real state
                state = State()
                state.extra_data = {}

                # Add a user message to the state history with a specific query about file operations
                # Using a very explicit query to increase chances the LLM will choose code search
                user_message = MessageAction(
                    content="Please use the code search tool to find code related to file operations in the openhands/core directory"
                )
                user_message._source = "user"
                state.history = [user_message]

                # Create a method to get the last user message
                state.get_last_user_message = lambda: user_message

                # Process the message with the agent
                with patch('openhands.agenthub.codeact_agent.function_calling.CODE_SEARCH_AVAILABLE', True):
                    # This will use the real LLM to process the user's query
                    print("\nSending query to real LLM. This may take a moment...")
                    action = agent.step(state)

                    # Verify that an action was returned
                    self.assertIsNotNone(action)

                    print(f"\nLLM chose action: {type(action).__name__}")

                    # If the action is a CodeSearchAction, verify it works
                    if isinstance(action, CodeSearchAction):
                        print(f"CodeSearchAction details:")
                        print(f"  Query: {action.query}")
                        print(f"  Repo path: {action.repo_path}")
                        print(f"  Extensions: {action.extensions}")
                        print(f"  k: {action.k}")

                        # Execute the action to verify it works end-to-end
                        observation = execute_code_search(action)

                        # Verify the observation
                        self.assertIsInstance(observation, CodeSearchObservation)
                        self.assertIsNotNone(observation.content)
                        self.assertIsNotNone(observation.results)
                        self.assertGreaterEqual(len(observation.results), 1)

                        # Print some details about the results
                        print(f"\nCode search found {len(observation.results)} results:")
                        for i, result in enumerate(observation.results):
                            print(f"Result {i+1}: {result['file']} (Score: {result['score']:.2f})")
                            print(f"Content snippet: {result['content'][:100]}...\n")

                        # Test passed
                        self.assertTrue(True)
                    else:
                        # If the LLM chose a different action, we'll print what it did
                        # This is not necessarily a failure, as the LLM might have chosen a different approach
                        print(f"LLM chose a different action: {type(action).__name__}")
                        print(f"Action details: {action}")

                        # For this test, we'll mark it as a failure if code search wasn't chosen
                        self.fail("LLM did not choose to use the code search tool as requested")
            finally:
                # Clean up environment variable
                if 'OPENHANDS_TEST_MOCK_MODE' in os.environ:
                    del os.environ['OPENHANDS_TEST_MOCK_MODE']

if __name__ == "__main__":
    unittest.main()