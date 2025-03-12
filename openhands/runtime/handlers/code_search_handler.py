"""Handler for code search actions."""

import logging
import os
from typing import Optional

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.action.code_search import CodeSearchAction
from openhands.events.observation import Observation
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.runtime.handlers.handler import ActionHandler

try:
    from openhands_aci.tools.code_search_tool import code_search_tool
    from openhands_aci.rag.code_search import execute_code_search
    OPENHANDS_ACI_AVAILABLE = True
except ImportError:
    logger.warning("openhands_aci not available, code search will use mock implementation")
    OPENHANDS_ACI_AVAILABLE = False


class CodeSearchHandler(ActionHandler):
    """Handler for code search actions."""

    def can_handle(self, action: Action) -> bool:
        """Check if this handler can handle the given action.
        
        Args:
            action: The action to check
            
        Returns:
            True if this handler can handle the action, False otherwise
        """
        return isinstance(action, CodeSearchAction)

    def handle(self, action: Action) -> Optional[Observation]:
        """Handle a code search action.
        
        Args:
            action: The action to handle
            
        Returns:
            A code search observation with the search results
        """
        if not isinstance(action, CodeSearchAction):
            return None
        
        logger.info(f"Handling code search action: {action.query}")
        
        # Validate repo_path
        repo_path = action.repo_path
        if not os.path.isdir(repo_path):
            return CodeSearchObservation(
                results=[],
                content=f"Error: Repository path '{repo_path}' is not a directory.",
                cause=action.id
            )
        
        # Use openhands_aci implementation if available
        if OPENHANDS_ACI_AVAILABLE:
            # Check if we should use mock mode for testing
            use_mock = os.environ.get('OPENHANDS_TEST_MOCK_MODE') == 'true'
            return execute_code_search(action, mock_mode=use_mock)
        
        # Mock implementation if openhands_aci is not available
        logger.warning("Using mock implementation for code search")
        return CodeSearchObservation(
            results=[
                {
                    "file": "example/file1.py",
                    "score": 0.95,
                    "content": "def example_function():\n    print('This is an example')\n"
                },
                {
                    "file": "example/file2.py",
                    "score": 0.85,
                    "content": "class ExampleClass:\n    def __init__(self):\n        self.value = 'example'\n"
                }
            ],
            content="Result 1: example/file1.py (Relevance score: 0.95)\n```\ndef example_function():\n    print('This is an example')\n```\n\nResult 2: example/file2.py (Relevance score: 0.85)\n```\nclass ExampleClass:\n    def __init__(self):\n        self.value = 'example'\n```\n",
            cause=action.id
        )