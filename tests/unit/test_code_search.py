import os
import pytest
from unittest.mock import patch, MagicMock

from openhands.agenthub.codeact_agent.tools import CodeSearchTool
from openhands.events.action import CodeSearchAction
from openhands.events.observation.code_search import CodeSearchObservation
from openhands.events.observation.error import ErrorObservation
from openhands.runtime.search_engine.code_search import code_search


def test_code_search_tool_definition():
    """Test that the CodeSearchTool is defined correctly."""
    assert CodeSearchTool['type'] == 'function'
    assert CodeSearchTool.function.name == 'code_search'
    assert 'query' in CodeSearchTool.function.parameters['properties']
    assert 'repo_path' in CodeSearchTool.function.parameters['properties']
    assert 'extensions' in CodeSearchTool.function.parameters['properties']
    assert 'k' in CodeSearchTool.function.parameters['properties']
    assert CodeSearchTool.function.parameters['required'] == ['query']


def test_code_search_action():
    """Test that the CodeSearchAction is defined correctly."""
    action = CodeSearchAction(
        query="function that handles HTTP requests",
        repo_path="/path/to/repo",
        extensions=[".py", ".js"],
        k=10,
    )
    
    assert action.query == "function that handles HTTP requests"
    assert action.repo_path == "/path/to/repo"
    assert action.extensions == [".py", ".js"]
    assert action.k == 10
    
    # Test string representation
    str_repr = str(action)
    assert "function that handles HTTP requests" in str_repr
    assert "/path/to/repo" in str_repr
    assert ".py" in str_repr
    assert ".js" in str_repr
    assert "10" in str_repr


def test_code_search_observation():
    """Test that the CodeSearchObservation is defined correctly."""
    results = [
        {
            "file": "file1.py",
            "score": 0.95,
            "content": "def handle_http_request():\n    pass",
        },
        {
            "file": "file2.js",
            "score": 0.85,
            "content": "function handleHttpRequest() {\n    // code\n}",
        },
    ]
    
    observation = CodeSearchObservation(
        query="function that handles HTTP requests",
        results=results,
        repo_path="/path/to/repo",
    )
    
    assert observation.query == "function that handles HTTP requests"
    assert observation.results == results
    assert observation.repo_path == "/path/to/repo"
    
    # Test content property
    content = observation.content
    assert "function that handles HTTP requests" in content
    assert "file1.py" in content
    assert "file2.js" in content
    assert "0.95" in content
    assert "0.85" in content
    assert "def handle_http_request()" in content
    assert "function handleHttpRequest()" in content


@patch('openhands_aci.code_search.initialize_code_search')
@patch('openhands_aci.code_search.search_code')
def test_code_search_function(mock_search_code, mock_initialize_code_search):
    """Test the code_search function."""
    # Mock the initialize_code_search function
    mock_initialize_code_search.return_value = {
        "status": "success",
        "num_documents": 100,
    }
    
    # Mock the search_code function
    mock_search_code.return_value = {
        "status": "success",
        "results": [
            {
                "path": "file1.py",
                "score": 0.95,
                "content": "def handle_http_request():\n    pass",
            },
        ],
    }
    
    # Test initialization
    action = CodeSearchAction(
        query="initialize",
        repo_path=os.getcwd(),  # Use current directory for testing
        extensions=[".py"],
    )
    
    with patch('os.path.exists', return_value=False):  # Simulate index doesn't exist
        result = code_search(action)
        
        assert isinstance(result, CodeSearchObservation)
        assert result.query == "initialize"
        assert result.num_documents == 100
        assert len(result.results) == 0
        
        mock_initialize_code_search.assert_called_once()
    
    # Test search
    action = CodeSearchAction(
        query="function that handles HTTP requests",
        repo_path=None,  # No repo_path needed for search
        extensions=[".py"],
        k=5,
    )
    
    with patch('os.path.exists', return_value=True):  # Simulate index exists
        result = code_search(action)
        
        assert isinstance(result, CodeSearchObservation)
        assert result.query == "function that handles HTTP requests"
        assert len(result.results) == 1
        assert result.results[0]["file"] == "file1.py"
        assert result.results[0]["score"] == 0.95
        
        mock_search_code.assert_called_once()


@patch('openhands_aci.code_search.search_code')
def test_code_search_error_handling(mock_search_code):
    """Test error handling in the code_search function."""
    # Mock the search_code function to return an error
    mock_search_code.return_value = {
        "status": "error",
        "message": "Failed to search code",
    }
    
    # Test search with error
    action = CodeSearchAction(
        query="function that handles HTTP requests",
    )
    
    with patch('os.path.exists', return_value=True):  # Simulate index exists
        result = code_search(action)
        
        assert isinstance(result, ErrorObservation)
        assert "Failed to search code" in result.content