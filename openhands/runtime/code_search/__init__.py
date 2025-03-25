"""
Search engine functionality for OpenHands.
This package provides search engine functionality for OpenHands, including:
- Web search using Brave Search
- Code search using semantic search
"""

from .code_search import CodeSearchRuntime

__all__ = ['CodeSearchRuntime']
