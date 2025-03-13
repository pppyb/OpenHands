"""
Tool for code search functionality.
"""

import json
from typing import Dict, List, Optional, Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CodeSearchAction


class CodeSearchTool:
    """Tool for code search functionality."""

    def __init__(self):
        """Initialize the code search tool."""
        self.name = "code_search"
        self.description = (
            "Search code in a repository using natural language queries. "
            "First initialize with a repository path, then search with a query."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["initialize", "search"],
                    "description": "Command to execute. 'initialize' to index a repository, 'search' to search in an indexed repository.",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to the git repository. Required for both 'initialize' and 'search' commands.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query. Required for 'search' command.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g. ['.py', '.js']). If not provided, include all files.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return for 'search' command. Default is 5.",
                },
            },
            "required": ["command", "repo_path"],
        }

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute the code search tool.

        Args:
            command: Command to execute. One of 'initialize', 'search'.
            repo_path: Path to the git repository.
            query: Search query. Required for 'search' command.
            extensions: List of file extensions to include (e.g. ['.py', '.js']).
                        If None, include all files.
            k: Number of results to return for 'search' command.

        Returns:
            Dictionary with status and message.
        """
        command = kwargs.get("command")
        repo_path = kwargs.get("repo_path")
        query = kwargs.get("query")
        extensions = kwargs.get("extensions")
        k = kwargs.get("k", 5)

        if command == "initialize":
            action = CodeSearchAction(
                command=command,
                repo_path=repo_path,
                extensions=extensions,
            )
        elif command == "search":
            if not query:
                return {"status": "error", "message": "query is required for search command"}
            
            action = CodeSearchAction(
                command=command,
                repo_path=repo_path,
                query=query,
                k=k,
            )
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}

        result = action.execute()
        return result