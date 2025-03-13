"""
Action class for code search functionality.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openhands.events.action.action import Action
from openhands.tools.code_search import initialize_code_search, search_code, update_code_search


@dataclass
class CodeSearchAction(Action):
    """Action to search code in a repository."""

    command: str  # 'initialize', 'search', or 'update'
    repo_path: Optional[str] = None
    query: Optional[str] = None
    k: int = 5
    extensions: Optional[List[str]] = None
    embedding_model: Optional[str] = None
    
    blocking: bool = True
    
    def __post_init__(self):
        # Validate required parameters based on command
        if self.command == 'initialize' or self.command == 'update':
            if not self.repo_path:
                raise ValueError(f"repo_path is required for command '{self.command}'")
        elif self.command == 'search':
            if not self.query:
                raise ValueError("query is required for command 'search'")
    
    def execute(self) -> Dict[str, Any]:
        """Execute the code search action.
        
        Returns:
            Dictionary with status and results.
        """
        if self.command == 'initialize':
            return initialize_code_search(
                repo_path=self.repo_path,
                save_dir=f"/tmp/code_search/{self.repo_path.replace('/', '_')}",
                extensions=self.extensions,
                embedding_model=self.embedding_model,
            )
        elif self.command == 'search':
            # Determine the save_dir from repo_path if provided, otherwise use a default
            save_dir = f"/tmp/code_search/{self.repo_path.replace('/', '_')}" if self.repo_path else "/tmp/code_search/default"
            return search_code(
                save_dir=save_dir,
                query=self.query,
                k=self.k,
            )
        elif self.command == 'update':
            return update_code_search(
                repo_path=self.repo_path,
                save_dir=f"/tmp/code_search/{self.repo_path.replace('/', '_')}",
                extensions=self.extensions,
            )
        else:
            return {
                'status': 'error',
                'message': f"Unknown command: {self.command}",
            }