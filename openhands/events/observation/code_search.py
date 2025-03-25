"""Code search observation implementations."""

from typing import List, Optional

from pydantic import BaseModel, Field

from openhands.core.schema.observation import ObservationType


class CodeSearchResult(BaseModel):
    """A single code search result."""

    id: str = Field(description='Unique identifier for the result')
    content: str = Field(description='Content of the file')
    path: str = Field(description='Path to the file')
    score: float = Field(description='Search relevance score')


class CodeSearchInitializedObservation(BaseModel):
    """Observation returned after initializing code search."""

    type: str = Field(default=ObservationType.CODE_SEARCH_INITIALIZED)
    status: str = Field(description='Status of the initialization (success/error)')
    message: str = Field(description='Status message')
    num_documents: Optional[int] = Field(
        default=None, description='Number of documents indexed'
    )


class CodeSearchResultsObservation(BaseModel):
    """Observation containing code search results."""

    type: str = Field(default=ObservationType.CODE_SEARCH_RESULTS)
    status: str = Field(description='Status of the search (success/error)')
    message: Optional[str] = Field(
        default=None, description='Error message if status is error'
    )
    results: Optional[List[CodeSearchResult]] = Field(
        default=None, description='Search results if status is success'
    )
