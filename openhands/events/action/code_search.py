"""Code search action module."""

from dataclasses import dataclass
from typing import ClassVar, List, Optional

from openhands.core.schema.action import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


@dataclass
class CodeSearchAction(Action):
    """Search for relevant code in a codebase using semantic search.
    
    This action uses Retrieval Augmented Generation (RAG) to find relevant code
    based on natural language queries. It first indexes the codebase (if needed)
    and then performs a semantic search.
    
    Attributes:
        query: Natural language query.
        repo_path: Path to the Git repository to search (optional if save_dir exists).
        save_dir: Directory to save/load the search index (defaults to .code_search_index).
        extensions: List of file extensions to include (e.g. [".py", ".js"]).
        k: Number of results to return.
        remove_duplicates: Whether to remove duplicate file results.
        min_score: Minimum score threshold to filter out low-quality matches.
        thought: Reasoning behind the search.
        action: Type of action to execute.
        runnable: Indicates whether the action is executable.
        security_risk: Indicates any security risks associated with the action.
        blocking: Indicates whether the action is a blocking operation.
    """
    
    query: str
    repo_path: Optional[str] = None
    save_dir: Optional[str] = None
    extensions: Optional[List[str]] = None
    k: int = 5
    remove_duplicates: bool = True
    min_score: float = 0.5
    thought: str = ''
    action: str = ActionType.CODE_SEARCH
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None
    blocking: bool = True  # Set as a blocking operation
    
    @property
    def message(self) -> str:
        """Get a human-readable message describing the code search action."""
        return f'Search code: {self.query}'
    
    def __repr__(self) -> str:
        """Get a string representation of the code search action."""
        ret = '**Code Search Action**\n'
        ret += f'Query: {self.query}\n'
        if self.repo_path:
            ret += f'Repository: {self.repo_path}\n'
        if self.extensions:
            ret += f'Extensions: {", ".join(self.extensions)}\n'
        ret += f'Number of results: {self.k}\n'
        ret += f'Thought: {self.thought}\n'
        return ret