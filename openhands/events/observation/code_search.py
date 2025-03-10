"""Code search observation module."""

from dataclasses import dataclass
from typing import Any, Dict, List

from openhands.core.schema.observation import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class CodeSearchObservation(Observation):
    """Result of a code search operation.
    
    This observation contains the results of a semantic code search operation,
    including file paths, relevance scores, and code snippets.
    
    Attributes:
        results: List of dictionaries containing search results.
        observation: Type of observation.
    """
    
    results: List[Dict[str, Any]]
    observation: str = ObservationType.CODE_SEARCH
    # _content: str | None = None
    
    @property
    def message(self) -> str:
        """Get a human-readable message describing the code search results."""
        return f'Found {len(self.results)} code snippets.'
    
    # @property
    # def content(self) -> str:
    #     """Format search results for display."""
    #     if self._content is not None:
    #         return self._content
            
    #     if not self.results:
    #         self._content = "No code snippets matching your query were found."
    #         return self._content
        
    #     output = []
    #     for i, result in enumerate(self.results, 1):
    #         output.append(f"Result {i}: {result['file']} (Relevance score: {result['score']})")
    #         output.append("```")
    #         output.append(result['content'])
    #         output.append("```\n")
        
    #     self._content = "\n".join(output)
    #     return self._content
    
    def __str__(self) -> str:
        """Get a string representation of the code search observation."""
        return f"[Found {len(self.results)} code snippets.]\n{self.content}"