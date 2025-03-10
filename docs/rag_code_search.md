# RAG Code Search in OpenHands

This document describes the Retrieval Augmented Generation (RAG) code search functionality in OpenHands, how it's integrated into the agent system, and how to test it.

## Overview

The RAG code search functionality allows OpenHands agents to search for relevant code in a repository using natural language queries. This is particularly useful for tasks that require understanding or modifying code, as it helps the agent quickly find relevant parts of the codebase.

## How It Works

1. **Indexing**: The first time a repository is searched, an index is created using sentence embeddings of the code files.
2. **Searching**: When a query is made, it's converted to an embedding and compared to the indexed code files.
3. **Ranking**: Results are ranked by similarity score and returned to the agent.
4. **Integration**: The functionality is integrated into OpenHands as an action-observation pair (`CodeSearchAction` and `CodeSearchObservation`).

## Components

### Core Components

- **Code Search Tool**: Implemented in `openhands_aci.tools.code_search_tool`, this is the core functionality that indexes and searches code.
- **Action**: `CodeSearchAction` in `openhands.events.action.code_search` defines how agents can request code searches.
- **Observation**: `CodeSearchObservation` in `openhands.events.observation.code_search` defines how search results are returned to agents.
- **Schema Integration**: The action and observation types are defined in `openhands.core.schema.action` and `openhands.core.schema.observation`.

### Integration with Agent System

The code search functionality is integrated into the OpenHands agent system through:

1. **Action Execution**: The `ActionExecutor` in `openhands.runtime.action_execution_server` can execute `CodeSearchAction` and return `CodeSearchObservation`.
2. **Agent Usage**: Agents can create `CodeSearchAction` objects to search for code and process the resulting `CodeSearchObservation`.

## Usage

### Basic Usage

```python
from openhands.events.action.code_search import CodeSearchAction

# Create a code search action
action = CodeSearchAction(
    query="function that handles API requests",
    repo_path="/path/to/repo",
    extensions=[".py", ".js"],
    k=5
)

# Execute the action (in a real agent, this would be done by the agent system)
observation = agent.execute_action(action)

# Process the observation
if isinstance(observation, CodeSearchObservation):
    for result in observation.results:
        print(f"File: {result['file']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}")
```

### In an Agent

In a real OpenHands agent, the code search functionality would be used as part of the agent's reasoning process:

1. The agent identifies a need to understand some part of the codebase.
2. The agent creates a `CodeSearchAction` with an appropriate query.
3. The agent system executes the action and returns a `CodeSearchObservation`.
4. The agent processes the observation and uses the results to inform its next actions.

## Testing

### Unit Tests

Unit tests for the code search functionality are in `tests/unit/test_code_search_integration.py`. These tests verify that:

1. `CodeSearchAction` and `CodeSearchObservation` can be created correctly.
2. The code search functionality is properly integrated with the `ActionExecutor`.
3. The schema integration is correct.

To run the unit tests:

```bash
python -m pytest tests/unit/test_code_search_integration.py -v
```

### Integration Tests

Integration tests that simulate how an agent would use the code search functionality are in `scripts/test_agent_code_search.py`. This script:

1. Creates a `CodeSearchAction` with a specified query.
2. Executes the action using an `ActionExecutor`.
3. Processes the resulting `CodeSearchObservation`.
4. Simulates how an agent would reason about the results.

To run the integration test:

```bash
python scripts/test_agent_code_search.py --repo /path/to/repo --query "your search query"
```

### Full Agent Tests

For a more comprehensive test of how the code search functionality is used in a real agent, use `scripts/test_rag_agent_integration.py`. This script:

1. Initializes a full OpenHands agent with a specified repository.
2. Gives the agent tasks that would benefit from code search.
3. Analyzes how the agent uses the code search functionality to complete these tasks.
4. Generates a detailed report of the agent's code search usage.

To run the full agent test:

```bash
python scripts/test_rag_agent_integration.py --repo /path/to/repo --output results.json
```

## Limitations and Future Work

### Current Limitations

- The code search functionality currently only works on a single repository at a time.
- The indexing process can be slow for large repositories.
- The search results are based purely on semantic similarity and don't consider code structure.

### Future Work

- Improve indexing performance for large repositories.
- Add support for searching across multiple repositories.
- Incorporate code structure and dependencies into the search process.
- Add support for more fine-grained queries (e.g., "find all functions that call X").
- Integrate with other tools like static analysis to provide more context to the agent.