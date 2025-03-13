"""
Example script demonstrating how to use the code search functionality.
"""

import os
import tempfile
from pathlib import Path

from git import Repo

from openhands.tools.code_search import initialize_code_search, search_code


def create_test_repo():
    """Create a temporary git repository with some test files."""
    # Create a persistent directory instead of a temporary one
    temp_dir = "/tmp/test_repo"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize git repo
    try:
        repo = Repo(temp_dir)
    except:
        repo = Repo.init(temp_dir)

    # Create test files
    files = {
        'main.py': 'def hello():\n    print("Hello, World!")',
        'utils/helper.py': 'def add(a, b):\n    return a + b',
        'README.md': '# Test Repository\n This is a test.',
    }

    for path, content in files.items():
        file_path = Path(temp_dir) / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)

    # Add and commit files
    repo.git.add(all=True)
    try:
        repo.git.commit('-m', 'Initial commit')
    except:
        # If there are no changes, it's fine
        pass

    return temp_dir


def main():
    """Run the code search example."""
    # Set the OpenAI API key (if needed)
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

    # Create a test repository
    repo_path = create_test_repo()
    print(f"Created test repository at {repo_path}")

    # Initialize code search
    save_dir = "/tmp/code_search/test_repo"
    result = initialize_code_search(
        repo_path=repo_path,
        save_dir=save_dir,
        extensions=['.py'],
        embedding_model='BAAI/bge-base-en-v1.5',
    )
    print(f"Initialization result: {result}")

    # Search for code
    search_result = search_code(
        save_dir=save_dir,
        query='function that adds two numbers',
        k=5,
    )
    print(f"Search result status: {search_result['status']}")
    
    # Print the search results
    if search_result['status'] == 'success':
        for i, doc in enumerate(search_result['results']):
            print(f"\nResult {i+1}:")
            print(f"Path: {doc['path']}")
            print(f"Score: {doc['score']:.4f}")
            print("Content:")
            print(doc['content'])


if __name__ == "__main__":
    main()