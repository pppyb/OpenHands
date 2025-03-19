from pydantic import BaseModel, Field


class CodeSearchConfig(BaseModel):
    """Configuration for code search functionality.

    Attributes:
        enable_code_search: Whether to enable the code search feature.
        embedding_model: The embedding model to use for code search.
        default_save_dir: The default directory to save code search indices.
        default_extensions: The default file extensions to include in code search.
        default_results_count: The default number of results to return.
    """

    enable_code_search: bool = Field(default=False)
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    default_save_dir: str = Field(default=".code_search_index")
    default_extensions: list[str] = Field(default=[".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp"])
    default_results_count: int = Field(default=5)

    model_config = {"extra": "forbid"}