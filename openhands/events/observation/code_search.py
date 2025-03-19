"""
Observation for code search functionality.

This module defines the observation class for code search operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class CodeSearchObservation(Observation):
    """Observation from a code search action.

    Attributes:
        query: The search query that was executed.
        results: The search results.
        status: Status of the operation ('success' or 'error').
        message: Message describing the result.
        num_documents: Number of documents indexed (for initialize operations).
        repo_path: Path to the repository.
        observation: The type of observation.
        content: The content of the observation (required by Observation base class).
    """

    query: str
    results: List[Dict[str, Any]]
    status: str = "success"
    message: str = ""
    num_documents: Optional[int] = None
    repo_path: Optional[str] = None
    observation: str = ObservationType.CODE_SEARCH
    content: str = ""  # Required by Observation base class

    @property
    def _formatted_content(self) -> str:
        """Get the content of the observation.
        
        Returns:
            String representation of the observation
        """
        if self.status == "error":
            return f"Error: {self.message}"
        
        if self.num_documents is not None:
            return f"Successfully indexed {self.num_documents} files from {self.repo_path}"
        
        result_str = f"Found {len(self.results)} results for query: '{self.query}'"
        if self.results:
            result_str += "\n\nResults:"
            for i, result in enumerate(self.results, 1):
                result_str += f"\n\n{i}. {result.get('file', 'Unknown')} (score: {result.get('score', 0):.3f})"
                content = result.get('content', '')
                if content:
                    result_str += f"\n{'-' * 80}\n{content}"
        return result_str

    def __str__(self) -> str:
        ret = "**CodeSearchObservation**\n"
        ret += f"Query: {self.query}\n"
        ret += f"Found {len(self.results)} results:\n\n"
        
        for i, result in enumerate(self.results, 1):
            ret += f"Result {i}: {result.get('file', 'Unknown')} (Score: {result.get('score', 0):.3f})\n"
            ret += "-" * 80 + "\n"
            ret += result.get("content", "") + "\n\n"
        
        return ret