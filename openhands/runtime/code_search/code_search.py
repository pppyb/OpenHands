"""Code search runtime implementation."""

from typing import Dict, List, Optional

from openhands_aci.code_search import CodeSearchIndex
from openhands_aci.code_search import (
    initialize_code_search as aci_initialize_code_search,
)
from openhands_aci.code_search import search_code as aci_search_code

from openhands.core.config.code_search_config import CodeSearchConfig


class CodeSearchRuntime:
    """Runtime for code search functionality."""

    def __init__(self, config: CodeSearchConfig):
        """Initialize code search runtime.

        Args:
            config: Code search configuration
        """
        self.config = config
        self._index: Optional[CodeSearchIndex] = None

    def initialize(self) -> Dict[str, str]:
        """Initialize code search for a repository.

        Returns:
            Dictionary with status and message
        """
        result = aci_initialize_code_search(
            repo_path=self.config.repo_path,
            save_dir=self.config.save_dir,
            extensions=self.config.extensions,
            embedding_model=self.config.embedding_model,
            batch_size=self.config.batch_size,
        )
        return result

    def search(
        self, query: str, k: Optional[int] = None
    ) -> Dict[str, List[Dict[str, str]]]:
        """Search code in the indexed repository.

        Args:
            query: Search query
            k: Number of results to return (overrides config.top_k if provided)

        Returns:
            Dictionary with status and search results
        """
        k = k or self.config.top_k
        result = aci_search_code(
            save_dir=self.config.save_dir,
            query=query,
            k=k,
        )
        return result
