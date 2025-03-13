"""
Action for code search functionality.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.integrations.openhands_aci.code_search import (
    initialize_code_search,
    search_code,
)


class CodeSearchAction(Action):
    """Action for code search functionality."""

    def __init__(
        self,
        command: str,
        repo_path: Optional[str] = None,
        query: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        k: int = 5,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the code search action.

        Args:
            command: Command to execute. One of 'initialize', 'search'.
            repo_path: Path to the git repository. Required for 'initialize' command.
            query: Search query. Required for 'search' command.
            extensions: List of file extensions to include (e.g. ['.py', '.js']).
                        If None, include all files.
            k: Number of results to return for 'search' command.
            embedding_model: Name or path of the sentence transformer model to use.
                             If None, will use the model specified in the EMBEDDING_MODEL environment variable.
        """
        super().__init__()
        self.command = command
        self.repo_path = repo_path
        self.query = query
        self.extensions = extensions
        self.k = k
        self.embedding_model = embedding_model

    def execute(self) -> Dict[str, Any]:
        """Execute the code search action.

        Returns:
            Dictionary with status and message.
        """
        # Check if OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. Code search may not work properly.")
            
        if self.command == "initialize":
            if not self.repo_path:
                return {"status": "error", "message": "repo_path is required for initialize command"}
            
            # Create a save directory based on the repository name
            repo_name = Path(self.repo_path).name
            save_dir = f"/tmp/code_search/_{repo_name}"
            
            return initialize_code_search(
                repo_path=self.repo_path,
                save_dir=save_dir,
                extensions=self.extensions,
                embedding_model=self.embedding_model,
            )
        
        elif self.command == "search":
            if not self.query:
                return {"status": "error", "message": "query is required for search command"}
            
            # If repo_path is provided, construct the save directory
            if self.repo_path:
                repo_name = Path(self.repo_path).name
                save_dir = f"/tmp/code_search/_{repo_name}"
            else:
                return {"status": "error", "message": "repo_path is required for search command"}
            
            return search_code(
                save_dir=save_dir,
                query=self.query,
                k=self.k,
            )
        
        else:
            return {"status": "error", "message": f"Unknown command: {self.command}"}