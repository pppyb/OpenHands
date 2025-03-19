"""
Tool for code search functionality.

This tool enables searching code repositories using natural language queries.
It integrates with the CodeSearchAction to provide a seamless experience for users.
"""

from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
from typing import Dict, List, Optional, Any

_CODE_SEARCH_DESCRIPTION = """Search code in a repository using semantic search.

This tool uses Retrieval Augmented Generation (RAG) to find relevant code based on natural language queries. 
It can search through the codebase to find functions, classes, or code snippets that match your description.

Use this tool when you need to:
1. Find specific implementations in a large codebase
2. Locate code related to a particular feature or functionality
3. Understand how certain operations are implemented
4. Find examples of code patterns or usage
"""

CodeSearchTool = ChatCompletionToolParam(
    type="function",
    function=ChatCompletionToolParamFunctionChunk(
        name="code_search",
        description=_CODE_SEARCH_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query in natural language (e.g., 'function that handles HTTP requests').",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to the git repository to search. Optional if the index already exists.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g., ['.py', '.js']). Optional.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return. Default is 5.",
                },
            },
            "required": ["query"],
        },
    ),
)

