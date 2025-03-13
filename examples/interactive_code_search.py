"""
Interactive code search example that allows users to search their own repositories.
"""

import argparse
import os
import sys
from pathlib import Path

from openhands.integrations.openhands_aci.code_search import (
    initialize_code_search,
    search_code,
)


def main():
    """Run the interactive code search example."""
    parser = argparse.ArgumentParser(description="Interactive code search for code repositories")
    parser.add_argument(
        "--repo", 
        type=str, 
        help="Path to the git repository to search",
        default=None
    )
    parser.add_argument(
        "--extensions", 
        type=str, 
        nargs="+", 
        help="File extensions to include (e.g. .py .js)",
        default=[".py"]
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to index",
        default=100
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Embedding model to use",
        default="BAAI/bge-base-en-v1.5"
    )
    args = parser.parse_args()

    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to use the code search functionality.")
        print("Example: export OPENAI_API_KEY=your-api-key")
        return

    # Get repository path from user if not provided
    repo_path = args.repo
    if not repo_path:
        repo_path = input("Enter the path to the git repository to search: ")
        if not repo_path:
            print("Repository path is required.")
            return
    
    repo_path = os.path.abspath(repo_path)
    if not os.path.exists(repo_path):
        print(f"Repository path {repo_path} does not exist.")
        return
    
    # Create a save directory based on the repository name
    repo_name = Path(repo_path).name
    save_dir = f"/tmp/code_search/{repo_name}"
    
    # Initialize code search
    print(f"Initializing code search for repository at {repo_path}...")
    print(f"Including files with extensions: {args.extensions}")
    print(f"Maximum files to index: {args.max_files}")
    
    # Create a smaller test directory if needed
    if args.max_files < 100:
        test_dir = f"/tmp/test_repo_{Path(repo_path).name}"
        os.makedirs(test_dir, exist_ok=True)
        
        # Find files with the specified extensions
        all_files = []
        for ext in args.extensions:
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(ext):
                        all_files.append(os.path.join(root, file))
        
        # Limit the number of files
        selected_files = all_files[:args.max_files]
        print(f"Selected {len(selected_files)} files out of {len(all_files)} total files")
        
        # Copy the selected files to the test directory
        for file_path in selected_files:
            rel_path = os.path.relpath(file_path, repo_path)
            dest_path = os.path.join(test_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as src_file:
                try:
                    content = src_file.read()
                    with open(dest_path, 'w', encoding='utf-8') as dest_file:
                        dest_file.write(content)
                except Exception as e:
                    print(f"Error copying file {file_path}: {e}")
        
        # Initialize git repository in the test directory
        try:
            from git import Repo
            if os.path.exists(os.path.join(test_dir, ".git")):
                print(f"Git repository already exists in {test_dir}")
            else:
                Repo.init(test_dir)
                print(f"Initialized new git repository in {test_dir}")
                
            # Add and commit files
            repo = Repo(test_dir)
            repo.git.add(all=True)
            try:
                repo.git.commit('-m', 'Initial commit for code search test')
                print("Committed files to test repository")
            except Exception as e:
                print(f"Note: Could not commit files: {e}")
        except Exception as e:
            print(f"Warning: Could not initialize git repository: {e}")
            
        # Use the test directory instead
        repo_path = test_dir
        print(f"Using test directory: {repo_path}")
    
    result = initialize_code_search(
        repo_path=repo_path,
        save_dir=save_dir,
        extensions=args.extensions,
        embedding_model=args.model,
    )
    
    if result["status"] != "success":
        print(f"Error initializing code search: {result['message']}")
        return
    
    print(f"Successfully indexed {result['num_documents']} files.")
    
    # Interactive search loop
    while True:
        query = input("\nEnter your search query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        print(f"Searching for: {query}")
        search_result = search_code(
            save_dir=save_dir,
            query=query,
            k=5,
        )
        
        if search_result["status"] != "success":
            print(f"Error searching code: {search_result['message']}")
            continue
        
        # Print the search results
        results = search_result["results"]
        if not results:
            print("No results found.")
            continue
        
        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Path: {doc['path']}")
            print(f"Score: {doc['score']:.4f}")
            print("Content:")
            print(doc["content"])
            print("-" * 80)


if __name__ == "__main__":
    main()