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
        content: Formatted content of the search results.
        observation: Type of observation.
    """
    
    content: str
    results: List[Dict[str, Any]]
    observation: str = ObservationType.CODE_SEARCH
    
    @property
    def message(self) -> str:
        """Get a human-readable message describing the code search results."""
        return f'Found {len(self.results)} code snippets.'
    
    def __str__(self) -> str:
        """Get a string representation of the code search observation."""
        return f"[Found {len(self.results)} code snippets.]\n{self.content}"
