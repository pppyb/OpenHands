"""
Observation for code search functionality.

This module defines the observation class for code search operations.
"""

from typing import Dict, List, Optional, Any

from openhands.events.observation.observation import Observation


class CodeSearchObservation(Observation):
    """Observation for code search operations.
    
    This observation is returned when a code search operation is performed.
    It contains the results of the search or initialization.
    """
    
    observation = "code_search"
    
    def __init__(
        self,
        status: str,
        message: str,
        results: Optional[List[Dict[str, Any]]] = None,
        num_documents: Optional[int] = None,
        command: Optional[str] = None,
        repo_path: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """Initialize a code search observation.
        
        Args:
            status: Status of the operation ('success' or 'error')
            message: Message describing the result
            results: List of search results (for search operations)
            num_documents: Number of documents indexed (for initialize operations)
            command: Command that was executed ('initialize' or 'search')
            repo_path: Path to the repository
            query: Search query (for search operations)
        """
        super().__init__()
        self.status = status
        self.message = message
        self.results = results or []
        self.num_documents = num_documents
        self.command = command
        self.repo_path = repo_path
        self.query = query
    
    @property
    def content(self) -> str:
        """Get the content of the observation.
        
        Returns:
            String representation of the observation
        """
        if self.status == "error":
            return f"Error: {self.message}"
        
        if self.command == "initialize":
            return f"Successfully indexed {self.num_documents} files from {self.repo_path}"
        
        if self.command == "search":
            result_str = f"Found {len(self.results)} results for query: '{self.query}'"
            if self.results:
                result_str += "\n\nResults:"
                for i, result in enumerate(self.results):
                    result_str += f"\n\n{i+1}. {result.get('path', 'Unknown')} (score: {result.get('score', 0):.2f})"
                    content = result.get('content', '')
                    if content:
                        # Add a snippet of the content
                        snippet = content[:200] + "..." if len(content) > 200 else content
                        result_str += f"\n   {snippet}"
            return result_str
        
        return self.message
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSearchObservation':
        """Create an observation from a dictionary.
        
        Args:
            data: Dictionary containing observation data
            
        Returns:
            CodeSearchObservation instance
        """
        return cls(
            status=data.get("status", "error"),
            message=data.get("message", "Unknown error"),
            results=data.get("results", []),
            num_documents=data.get("num_documents"),
            command=data.get("command"),
            repo_path=data.get("repo_path"),
            query=data.get("query"),
        )