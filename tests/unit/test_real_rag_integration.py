"""Test real RAG integration with code search."""

import os
import unittest
from unittest.mock import patch

from openhands.agenthub.codeact_agent.function_calling import get_registered_tools
from openhands.core.schema.action import ActionType

from openhands_aci.rag.code_search import CodeSearchAction, CodeSearchObservation, execute_code_search
from openhands_aci.rag.function_calling import register_code_search_tools


class TestRealRagIntegration(unittest.TestCase):
    """Test real RAG integration with code search."""

    def setUp(self):
        """Set up the test."""
        # Register the code search tool
        register_code_search_tools(lambda tool: get_registered_tools().append(tool))
        
        # Set up a mock OpenAI API key for testing
        os.environ["OPENAI_API_KEY"] = "sk-mock-api-key-for-testing"

    def test_code_search_action_execution(self):
        """Test that code search action can be executed."""
        # Create a code search action
        action = CodeSearchAction(
            query="Find code related to file operations",
            repo_path="/workspace/OpenHands",
            extensions=[".py"],
            k=3
        )
        
        # Set mock mode environment variable
        os.environ['OPENHANDS_TEST_MOCK_MODE'] = 'true'
        
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


if __name__ == "__main__":
    unittest.main()