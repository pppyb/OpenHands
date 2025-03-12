"""
Unit tests for code search integration in OpenHands.

These tests verify that the RAG code search functionality is properly
integrated into the OpenHands framework and can be used by agents.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.core.schema.action import ActionType
from openhands.core.schema.observation import ObservationType


class TestCodeSearchIntegration(unittest.TestCase):
    """Test the integration of code search functionality in OpenHands."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = self.temp_dir.name
        
        # Create some test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def create_test_files(self):
        """Create test files in the temporary directory."""
        # Create a simple Python file
        with open(os.path.join(self.repo_path, "test_file.py"), "w") as f:
            f.write("""
def search_code(query, repo_path):
    \"\"\"Search for code in a repository.
    
    This function uses RAG to find relevant code based on a query.
    
    Args:
        query: The search query
        repo_path: Path to the repository
        
    Returns:
        List of search results
    \"\"\"
    # Implementation details
    return ["result1", "result2"]
""")
        
        # Create a simple JavaScript file
        with open(os.path.join(self.repo_path, "test_file.js"), "w") as f:
            f.write("""
/**
 * Search for code in a repository
 * @param {string} query - The search query
 * @param {string} repoPath - Path to the repository
 * @returns {Array} List of search results
 */
function searchCode(query, repoPath) {
    // Implementation details
    return ["result1", "result2"];
}
""")
    
    def test_code_search_action_creation(self):
        """Test creating a CodeSearchAction."""
        action = CodeSearchAction(
            query="search function",
            repo_path=self.repo_path,
            extensions=[".py", ".js"],
            k=5
        )
        
        # Verify action properties
        self.assertEqual(action.query, "search function")
        self.assertEqual(action.repo_path, self.repo_path)
        self.assertEqual(action.extensions, [".py", ".js"])
        self.assertEqual(action.k, 5)
        self.assertEqual(action.action, ActionType.CODE_SEARCH)
        self.assertTrue(action.runnable)
        self.assertTrue(action.blocking)
    
    def test_code_search_observation_creation(self):
        """Test creating a CodeSearchObservation."""
        results = [
            {
                "file": "test_file.py",
                "score": 0.85,
                "content": "def search_code(query, repo_path):"
            }
        ]
        
        observation = CodeSearchObservation(results=results)
        # _content is initialized as None, no need to pass it explicitly
        
        # Verify observation properties
        self.assertEqual(observation.results, results)
        self.assertEqual(observation.observation, ObservationType.CODE_SEARCH)
        self.assertIn("Found 1 code", observation.message)
    
    @patch('openhands_aci.tools.code_search_tool.code_search_tool')
    def test_action_executor_integration(self, mock_code_search_tool):
        """Test integration with ActionExecutor."""
        # Mock the code_search_tool function
        mock_code_search_tool.return_value = {
            "status": "success",
            "results": [
                {
                    "file": "test_file.py",
                    "score": 0.85,
                    "content": "def search_code(query, repo_path):"
                }
            ]
        }
        
        # Import here to avoid circular imports
        from openhands.runtime.action_execution_server import ActionExecutor
        
        # Create a mock ActionExecutor with necessary attributes
        executor = MagicMock(spec=ActionExecutor)
        executor.bash_session = MagicMock()
        executor.bash_session.cwd = self.repo_path
        
        # Create a code search action
        action = CodeSearchAction(
            query="search function",
            repo_path=self.repo_path
        )
        
        # Call the code_search method directly
        # Note: In a real test, we would use asyncio to run this
        from openhands.runtime.action_execution_server import ActionExecutor
        code_search_method = ActionExecutor.code_search
        observation = code_search_method(executor, action)
        
        # Verify the observation
        self.assertIsInstance(observation, CodeSearchObservation)
        self.assertEqual(len(observation.results), 1)
        self.assertEqual(observation.results[0]["file"], "test_file.py")
        
        # Verify that code_search_tool was called with the right parameters
        mock_code_search_tool.assert_called_once()
        args, kwargs = mock_code_search_tool.call_args
        self.assertEqual(kwargs["query"], "search function")
        self.assertEqual(kwargs["repo_path"], self.repo_path)
    
    def test_schema_integration(self):
        """Test integration with OpenHands schema."""
        # Verify that CODE_SEARCH is defined in ActionType
        self.assertTrue(hasattr(ActionType, "CODE_SEARCH"))
        
        # Verify that CODE_SEARCH is defined in ObservationType
        self.assertTrue(hasattr(ObservationType, "CODE_SEARCH"))


if __name__ == "__main__":
    unittest.main()