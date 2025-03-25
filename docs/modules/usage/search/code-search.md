# Code Search

OpenHands provides a code search capability that allows agents to search through codebases using natural language queries. This guide explains how to configure and use the code search feature.

## Overview

The code search feature enables agents to:
- Search through codebases using natural language queries
- Find relevant code snippets, functions, and files
- Understand code structure and implementation details
- Navigate large codebases efficiently

## How It Works

The code search feature uses semantic search with embeddings to find relevant code based on natural language queries. It works by:

1. **Indexing**: First, the codebase is indexed by creating embeddings for each file or code snippet
2. **Searching**: When a query is made, it's converted to an embedding and compared with the indexed code
3. **Ranking**: Results are ranked by relevance and returned with context

## Configuration

### Configuration Setup

The code search feature is configured through the `code_search` section in your `config.toml`:

```toml
[code_search]
# Enable the code search feature
enable_code_search = true

# Set the embedding model to use
embedding_model = "BAAI/bge-base-en-v1.5"

# Set the default directory to save code search indices
default_save_dir = ".code_search_index"

# Set the default file extensions to include in code search
default_extensions = [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp"]

# Set the default number of results to return
default_results_count = 5
```

Or when using Docker, set the environment variables:
```bash
# Enable the code search feature
-e CODE_SEARCH_ENABLE_CODE_SEARCH=true

# Set the embedding model to use
-e CODE_SEARCH_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"

# Set the default directory to save code search indices
-e CODE_SEARCH_DEFAULT_SAVE_DIR=".code_search_index"

# Set the default number of results to return
-e CODE_SEARCH_DEFAULT_RESULTS_COUNT=5
```

### Agent Configuration

You also need to enable code search in the agent configuration:

```toml
[agent]
codeact_enable_code_search = true
```

Or when using Docker:
```bash
-e AGENT_CODEACT_ENABLE_CODE_SEARCH=true
```

## Usage Example

When the code search feature is enabled, agents can use the `code_search` tool to search through codebases. For example:

```python
# The agent can make a tool call like this:
{
    "name": "code_search",
    "arguments": {
        "query": "function that handles HTTP requests",
        "repo_path": "/path/to/repo",  # Optional if already indexed
        "extensions": [".py", ".js"],  # Optional
        "k": 5  # Optional, number of results
    }
}
```

The search results will be returned in a structured format that includes:
- File path
- Relevance score
- Code snippet with context

## Best Practices

1. **Query Formulation**
   - Use natural language to describe what you're looking for
   - Be specific about functionality, patterns, or concepts
   - Include relevant technical terms when appropriate

2. **Repository Organization**
   - Index repositories separately for better context
   - Use file extensions to focus on relevant code
   - Consider indexing only specific directories for large repositories

3. **Performance Considerations**
   - Indexing large repositories can be memory-intensive
   - The first search in a session may take longer due to model loading
   - Consider pre-indexing repositories for frequently used codebases

## Troubleshooting

Common issues and solutions:

1. **Search Not Working**
   - Verify `code_search.enable_code_search` is set to `true` in your config
   - Confirm `agent.codeact_enable_code_search` is set to `true`
   - Check that the required dependencies are installed

2. **Poor Search Results**
   - Try reformulating your query to be more specific
   - Check if the code you're looking for is in the indexed repository
   - Consider using different file extensions to narrow the search

3. **Memory Issues**
   - Large repositories may require significant memory for indexing
   - Consider indexing smaller portions of the repository
   - Use a smaller embedding model if memory is limited