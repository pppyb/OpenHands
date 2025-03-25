"""Code search tool for searching code in a repository."""

from litellm import ChatCompletionToolParam

CodeSearchTool: ChatCompletionToolParam = {
    'type': 'function',
    'function': {
        'name': 'code_search',
        'description': 'Search code in a repository using semantic search.',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The command to execute. Must be one of: "initialize_code_search", "search_code".',
                    'enum': ['initialize_code_search', 'search_code'],
                },
                'repo_path': {
                    'type': 'string',
                    'description': 'Path to the repository to search. Required for initialize_code_search.',
                },
                'save_dir': {
                    'type': 'string',
                    'description': 'Directory to save the search index. Default is "code_search_index".',
                },
                'extensions': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of file extensions to include in the search (e.g. [".py", ".js"]). If not provided, includes all files.',
                },
                'embedding_model': {
                    'type': 'string',
                    'description': 'Name or path of the embedding model to use. Default is "BAAI/bge-base-en-v1.5".',
                },
                'batch_size': {
                    'type': 'integer',
                    'description': 'Batch size for embedding generation. Default is 32.',
                },
                'query': {
                    'type': 'string',
                    'description': 'Search query. Required for search_code.',
                },
                'k': {
                    'type': 'integer',
                    'description': 'Number of results to return. Default is 5.',
                },
            },
            'required': ['command'],
            'allOf': [
                {
                    'if': {
                        'properties': {'command': {'const': 'initialize_code_search'}},
                    },
                    'then': {
                        'required': ['repo_path'],
                    },
                },
                {
                    'if': {
                        'properties': {'command': {'const': 'search_code'}},
                    },
                    'then': {
                        'required': ['query'],
                    },
                },
            ],
        },
    },
}
