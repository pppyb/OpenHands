"""
Tool for code search functionality.

This tool enables searching code repositories using natural language queries.
It integrates with the CodeSearchAction to provide a seamless experience for users.
"""

import json
import os
import re
from typing import Dict, List, Optional, Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CodeSearchAction


class CodeSearchTool:
    """Tool for code search functionality.
    
    This tool enables searching code repositories using natural language queries.
    It integrates with the CodeSearchAction to provide a seamless experience for users.
    """

    def __init__(self):
        """Initialize the code search tool."""
        self.name = "code_search"
        self.description = (
            "Search code in a repository using natural language queries. "
            "First initialize with a repository path, then search with a query. "
            "This tool can help find relevant code snippets, functions, or files based on natural language descriptions."
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
                    "description": "Path to the git repository. Required for both 'initialize' and 'search' commands. Can be absolute or relative path.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query in natural language. Required for 'search' command. Example: 'function that handles API authentication' or 'code that processes user input'.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g. ['.py', '.js']). If not provided, common code file extensions will be used.",
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
        # Extract and validate parameters
        command = kwargs.get("command")
        if not command:
            return {"status": "error", "message": "command is required"}
        
        repo_path = kwargs.get("repo_path")
        if not repo_path:
            return {"status": "error", "message": "repo_path is required"}
        
        # Normalize repository path
        repo_path = os.path.abspath(os.path.expanduser(repo_path))
        
        # Check if repository exists
        if not os.path.exists(repo_path):
            return {
                "status": "error", 
                "message": f"Repository path does not exist: {repo_path}. Please provide a valid path."
            }
        
        query = kwargs.get("query")
        extensions = kwargs.get("extensions")
        k = kwargs.get("k", 5)
        
        # Process based on command
        if command == "initialize":
            logger.info(f"Initializing code search for repository: {repo_path}")
            
            # Create the action
            action = CodeSearchAction(
                command=command,
                repo_path=repo_path,
                extensions=extensions,
            )
            
            # Execute and return the result
            result = action.execute()
            
            # Enhance the result with more user-friendly information
            if result["status"] == "success":
                result["user_message"] = (
                    f"Successfully indexed {result.get('num_documents', 0)} files from {repo_path}. "
                    f"You can now search this repository using the 'search' command."
                )
            
            return result
            
        elif command == "search":
            # Validate query for search command
            if not query:
                return {"status": "error", "message": "query is required for search command"}
            
            logger.info(f"Searching for: {query}")
            logger.info(f"In repository: {repo_path}")
            
            # Clean and normalize query
            query = query.strip()
            
            # Create the action
            action = CodeSearchAction(
                command=command,
                repo_path=repo_path,
                query=query,
                k=k,
            )
            
            # Execute and return the result
            result = action.execute()
            
            # Enhance the result with more user-friendly information
            if result["status"] == "success":
                num_results = len(result.get("results", []))
                result["user_message"] = (
                    f"Found {num_results} results for query: '{query}' in repository {repo_path}."
                )
                
                # Add a summary of the results
                if num_results > 0:
                    result["summary"] = []
                    for i, doc in enumerate(result.get("results", [])):
                        result["summary"].append({
                            "index": i + 1,
                            "path": doc.get("path", "Unknown"),
                            "score": doc.get("score", 0.0),
                            "snippet": self._get_snippet(doc.get("content", "")),
                        })
            
            return result
            
        else:
            return {"status": "error", "message": f"Unknown command: {command}. Valid commands are 'initialize' and 'search'."}
    
    def _get_snippet(self, content: str, max_length: int = 150) -> str:
        """Get a snippet of the content.
        
        Args:
            content: The full content.
            max_length: Maximum length of the snippet.
            
        Returns:
            A snippet of the content.
        """
        if not content:
            return ""
        
        # If content is short enough, return it as is
        if len(content) <= max_length:
            return content
        
        # Otherwise, return a snippet
        return content[:max_length] + "..."