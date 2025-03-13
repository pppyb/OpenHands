from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_CODE_SEARCH_DESCRIPTION = """Search code in a repository using semantic search.
* This tool allows you to search for code snippets in a repository based on natural language queries.
* The search is powered by embeddings and returns the most relevant code snippets.
* You can specify the number of results to return.
* The search index must be initialized before searching.
"""

CodeSearchTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='code_search',
        description=_CODE_SEARCH_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'description': 'The command to run. Allowed options are: `initialize`, `search`, `update`.',
                    'enum': ['initialize', 'search', 'update'],
                    'type': 'string',
                },
                'repo_path': {
                    'description': 'Path to the git repository to search or index. Required for `initialize` and `update` commands.',
                    'type': 'string',
                },
                'query': {
                    'description': 'The search query. Required for `search` command.',
                    'type': 'string',
                },
                'k': {
                    'description': 'Number of results to return. Default is 5.',
                    'type': 'integer',
                },
                'extensions': {
                    'description': 'List of file extensions to include in the index. For example: [".py", ".js", ".ts"]',
                    'type': 'array',
                    'items': {
                        'type': 'string'
                    }
                },
                'embedding_model': {
                    'description': 'Name or path of the embedding model to use. If not provided, will use the model specified in the EMBEDDING_MODEL environment variable or the default model.',
                    'type': 'string',
                },
            },
            'required': ['command'],
        },
    ),
)