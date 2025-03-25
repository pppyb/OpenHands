"""Configuration for code search functionality."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeSearchConfig:
    """Configuration for code search functionality."""

    # Path to the repository to search
    repo_path: str = ''

    # Directory to save the search index
    save_dir: str = 'code_search_index'

    # File extensions to include in the search
    extensions: Optional[List[str]] = None

    # Name or path of the embedding model to use
    embedding_model: Optional[str] = None

    # Batch size for embedding generation
    batch_size: int = 32

    # Number of results to return in search
    top_k: int = 5
