"""Code search action implementations."""

from typing import List, Optional

from pydantic import BaseModel, Field

from openhands.core.schema.action import ActionType


class InitializeCodeSearchAction(BaseModel):
    """Action to initialize code search for a repository."""

    type: str = Field(default=ActionType.INITIALIZE_CODE_SEARCH)
    repo_path: str = Field(description='Path to the repository to search')
    save_dir: str = Field(
        default='code_search_index', description='Directory to save the search index'
    )
    extensions: Optional[List[str]] = Field(
        default=None, description='File extensions to include in the search'
    )
    embedding_model: Optional[str] = Field(
        default=None, description='Name or path of the embedding model to use'
    )
    batch_size: int = Field(
        default=32, description='Batch size for embedding generation'
    )


class SearchCodeAction(BaseModel):
    """Action to search code in an indexed repository."""

    type: str = Field(default=ActionType.SEARCH_CODE)
    query: str = Field(description='Search query')
    save_dir: str = Field(
        default='code_search_index', description='Directory containing the search index'
    )
    k: Optional[int] = Field(default=None, description='Number of results to return')
