"""
Action for code search functionality.

This action enables searching code repositories using natural language queries.
It supports both initialization of code search indexes and searching through indexed repositories.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.integrations.openhands_aci.code_search import (
    initialize_code_search,
    search_code,
)


class CodeSearchAction(Action):
    """Action for code search functionality.
    
    This action enables searching code repositories using natural language queries.
    It supports both initialization of code search indexes and searching through indexed repositories.
    """
    
    action = "code_search"

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
        
        # Normalize repository path
        if repo_path:
            self.repo_path = os.path.abspath(os.path.expanduser(repo_path))
        else:
            self.repo_path = None
            
        # Clean and normalize query
        if query:
            # Remove any leading/trailing whitespace and normalize quotes
            self.query = query.strip()
        else:
            self.query = None
            
        # Default extensions if none provided
        if extensions is None and command == 'initialize':
            self.extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.rb']
        else:
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
            
            # Verify that the repository path exists
            if not os.path.exists(self.repo_path):
                return {"status": "error", "message": f"Repository path does not exist: {self.repo_path}"}
            
            # Create a save directory based on the repository name
            repo_name = Path(self.repo_path).name
            save_dir = f"/tmp/code_search/_{repo_name}"
            
            # Log the initialization
            logger.info(f"Initializing code search for repository: {self.repo_path}")
            logger.info(f"Including file extensions: {self.extensions}")
            
            try:
                result = initialize_code_search(
                    repo_path=self.repo_path,
                    save_dir=save_dir,
                    extensions=self.extensions,
                    embedding_model=self.embedding_model,
                )
                
                # Add more context to the result
                if result["status"] == "success":
                    result["message"] = f"Successfully indexed {result.get('num_documents', 0)} files from {self.repo_path}"
                    logger.info(f"Code search initialization successful: {result['message']}")
                else:
                    logger.error(f"Code search initialization failed: {result.get('message', 'Unknown error')}")
                
                # Create and return a CodeSearchObservation
                return CodeSearchObservation(
                    status=result["status"],
                    message=result.get("message", ""),
                    num_documents=result.get("num_documents"),
                    command=self.command,
                    repo_path=self.repo_path,
                )
            except Exception as e:
                error_msg = f"Error initializing code search: {str(e)}"
                logger.error(error_msg)
                return CodeSearchObservation(
                    status="error",
                    message=error_msg,
                    command=self.command,
                    repo_path=self.repo_path,
                )
        
        elif self.command == "search":
            if not self.query:
                return CodeSearchObservation(
                    status="error",
                    message="query is required for search command",
                    command=self.command,
                    repo_path=self.repo_path,
                )
            
            # If repo_path is provided, construct the save directory
            if self.repo_path:
                repo_name = Path(self.repo_path).name
                save_dir = f"/tmp/code_search/_{repo_name}"
            else:
                return CodeSearchObservation(
                    status="error",
                    message="repo_path is required for search command",
                    command=self.command,
                )
            
            # Check if the save directory exists
            if not os.path.exists(save_dir):
                # Try to initialize the repository first
                logger.info(f"Save directory {save_dir} does not exist. Attempting to initialize the repository first.")
                init_result = self.execute_initialize()
                if init_result["status"] != "success":
                    return init_result
            
            # Log the search query
            logger.info(f"Searching for: {self.query}")
            logger.info(f"In repository: {self.repo_path}")
            
            try:
                result = search_code(
                    save_dir=save_dir,
                    query=self.query,
                    k=self.k,
                )
                
                # Add more context to the result
                if result["status"] == "success":
                    num_results = len(result.get("results", []))
                    result["message"] = f"Found {num_results} results for query: {self.query}"
                    logger.info(f"Code search successful: {result['message']}")
                else:
                    logger.error(f"Code search failed: {result.get('message', 'Unknown error')}")
                
                # Create and return a CodeSearchObservation
                return CodeSearchObservation(
                    status=result["status"],
                    message=result.get("message", ""),
                    results=result.get("results", []),
                    command=self.command,
                    repo_path=self.repo_path,
                    query=self.query,
                )
            except Exception as e:
                error_msg = f"Error searching code: {str(e)}"
                logger.error(error_msg)
                return CodeSearchObservation(
                    status="error",
                    message=error_msg,
                    command=self.command,
                    repo_path=self.repo_path,
                    query=self.query,
                )
        
        else:
            return CodeSearchObservation(
                status="error",
                message=f"Unknown command: {self.command}",
                command=self.command,
                repo_path=self.repo_path,
            )
    
    def execute_initialize(self) -> CodeSearchObservation:
        """Execute the initialize command.
        
        This is a helper method used when search is called but the repository hasn't been indexed yet.
        
        Returns:
            CodeSearchObservation with the result of the initialization.
        """
        if not self.repo_path:
            return CodeSearchObservation(
                status="error",
                message="repo_path is required for initialize command",
                command="initialize",
            )
        
        # Create a save directory based on the repository name
        repo_name = Path(self.repo_path).name
        save_dir = f"/tmp/code_search/_{repo_name}"
        
        try:
            result = initialize_code_search(
                repo_path=self.repo_path,
                save_dir=save_dir,
                extensions=self.extensions,
                embedding_model=self.embedding_model,
            )
            
            return CodeSearchObservation(
                status=result["status"],
                message=result.get("message", ""),
                num_documents=result.get("num_documents"),
                command="initialize",
                repo_path=self.repo_path,
            )
        except Exception as e:
            error_msg = f"Error initializing code search: {str(e)}"
            logger.error(error_msg)
            return CodeSearchObservation(
                status="error",
                message=error_msg,
                command="initialize",
                repo_path=self.repo_path,
            )