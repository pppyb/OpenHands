"""Code search tool for OpenHands."""

from litellm import ChatCompletionToolParam

CodeSearchTool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "code_search",
        "description": "Search for relevant code in a codebase using semantic search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant code",
                },
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to search in",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (e.g., [\".py\", \".js\"])",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                },
                "thought": {
                    "type": "string",
                    "description": "Optional thought process behind the search",
                },
            },
            "required": ["query", "repo_path"],
        },
    },
}