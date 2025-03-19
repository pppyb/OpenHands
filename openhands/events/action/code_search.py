from dataclasses import dataclass
from typing import ClassVar, List, Optional

from openhands.core.schema import ActionType
from openhands.events.action.action import Action


@dataclass
class CodeSearchAction(Action):
    """Action to search code in a repository.

    Attributes:
        query: The search query in natural language.
        repo_path: Path to the git repository to search (optional if save_dir exists).
        save_dir: Directory to save/load the search index.
        extensions: List of file extensions to include.
        k: Number of results to return.
        thought: The thought process behind the action.
        action: The type of action.
        runnable: Whether the action is runnable.
    """

    query: str
    repo_path: Optional[str] = None
    save_dir: Optional[str] = None
    extensions: Optional[List[str]] = None
    k: int = 5
    thought: str = ""
    action: str = ActionType.CODE_SEARCH
    runnable: ClassVar[bool] = True

    @property
    def message(self) -> str:
        return f"I am searching the codebase for: {self.query}"

    def __str__(self) -> str:
        ret = "**CodeSearchAction**\n"
        if self.thought:
            ret += f"THOUGHT: {self.thought}\n"
        ret += f"QUERY: {self.query}"
        if self.repo_path:
            ret += f"\nREPO_PATH: {self.repo_path}"
        if self.save_dir:
            ret += f"\nSAVE_DIR: {self.save_dir}"
        if self.extensions:
            ret += f"\nEXTENSIONS: {self.extensions}"
        ret += f"\nRESULTS_COUNT: {self.k}"
        return ret