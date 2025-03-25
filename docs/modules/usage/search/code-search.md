# Code Search

OpenHands provides a powerful code search functionality that allows you to search through codebases using semantic search. This feature uses embeddings to find code snippets that are semantically related to your search query, even if they don't contain the exact keywords.

## Configuration

To enable code search in your agent, set `codeact_enable_code_search = true` in your configuration:

```toml
[agent]
codeact_enable_code_search = true
```

## Usage

The code search functionality provides two main operations:

### 1. Initialize Code Search

Before you can search code, you need to initialize the search index for your repository:

```python
from openhands.events.action.code_search import InitializeCodeSearchAction
from openhands.events.observation.code_search import CodeSearchInitializedObservation

# Initialize code search for a repository
action = InitializeCodeSearchAction(
    repo_path="/path/to/your/repo",
    save_dir="code_search_index",  # Optional, defaults to "code_search_index"
    extensions=[".py", ".js"],     # Optional, filter by file extensions
    embedding_model="BAAI/bge-base-en-v1.5",  # Optional, defaults to "BAAI/bge-base-en-v1.5"
    batch_size=32                  # Optional, defaults to 32
)

# The observation will contain the initialization status
observation = await agent.run_action(action)
assert isinstance(observation, CodeSearchInitializedObservation)
print(f"Status: {observation.status}")
print(f"Message: {observation.message}")
print(f"Number of documents indexed: {observation.num_documents}")
```

### 2. Search Code

Once the index is initialized, you can search for code:

```python
from openhands.events.action.code_search import SearchCodeAction
from openhands.events.observation.code_search import CodeSearchResultsObservation

# Search for code
action = SearchCodeAction(
    query="function that adds two numbers",
    save_dir="code_search_index",  # Optional, defaults to "code_search_index"
    k=5                           # Optional, number of results to return
)

# The observation will contain the search results
observation = await agent.run_action(action)
assert isinstance(observation, CodeSearchResultsObservation)
print(f"Status: {observation.status}")
if observation.status == "success":
    for result in observation.results:
        print(f"\nFile: {result.path}")
        print(f"Score: {result.score}")
        print(f"Content:\n{result.content}")
```

## Implementation Details

The code search functionality is implemented using:

- [sentence-transformers](https://www.sbert.net/) for generating embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) as the default embedding model

The search process works by:

1. Converting code files into embeddings using the sentence transformer model
2. Building a FAISS index for efficient similarity search
3. Converting search queries into embeddings using the same model
4. Finding the most similar code snippets using cosine similarity

## Configuration Options

The code search functionality can be configured through the `CodeSearchConfig` class:

```python
from openhands.core.config.code_search_config import CodeSearchConfig

config = CodeSearchConfig(
    repo_path="",                          # Path to the repository to search
    save_dir="code_search_index",          # Directory to save the search index
    extensions=None,                       # List of file extensions to include
    embedding_model="BAAI/bge-base-en-v1.5", # Name or path of the embedding model
    batch_size=32,                         # Batch size for embedding generation
    top_k=5                                # Default number of results to return
)
```

## Best Practices

1. **Index Management**:
   - Keep the index in a separate directory from your code
   - Update the index when the codebase changes significantly
   - Consider using different indices for different parts of your codebase

2. **Search Queries**:
   - Use natural language queries that describe what you're looking for
   - Include relevant technical terms to improve search accuracy
   - Be specific but not too verbose

3. **Performance**:
   - Use appropriate batch sizes based on your available memory
   - Filter files by extension to reduce index size and improve search speed
   - Consider using a smaller embedding model if speed is critical

## Example Use Cases

1. **Finding Similar Code**:
   ```python
   # Search for similar implementations
   action = SearchCodeAction(
       query="function to parse JSON from file",
       extensions=[".py", ".js"]
   )
   ```

2. **Finding Usage Examples**:
   ```python
   # Search for examples of using a specific API
   action = SearchCodeAction(
       query="example of using requests library to make POST request",
       extensions=[".py"]
   )
   ```

3. **Finding Documentation**:
   ```python
   # Search for documentation about specific features
   action = SearchCodeAction(
       query="documentation about authentication configuration",
       extensions=[".md", ".rst", ".txt"]
   )
   ```

## Integration with Other Tools

The code search functionality can be combined with other OpenHands tools:

1. **With File Editor**:
   ```python
   # Search for code and then edit it
   search_action = SearchCodeAction(query="function to validate email")
   search_result = await agent.run_action(search_action)

   if search_result.status == "success" and search_result.results:
       edit_action = FileEditAction(
           path=search_result.results[0].path,
           content="# Add new validation logic here\n" + search_result.results[0].content
       )
       await agent.run_action(edit_action)
   ```

2. **With IPython**:
   ```python
   # Search for code and execute it in IPython
   search_action = SearchCodeAction(query="example unittest setup")
   search_result = await agent.run_action(search_action)

   if search_result.status == "success" and search_result.results:
       ipython_action = IPythonRunCellAction(code=search_result.results[0].content)
       await agent.run_action(ipython_action)
   ```
