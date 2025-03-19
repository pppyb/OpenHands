"""
Search engine functionality for OpenHands.

This package provides search engine functionality for OpenHands, including:
- Web search using Brave Search
- Code search using semantic search
"""

# from openhands.runtime.search_engine.brave_search import search
from openhands.runtime.search_engine.code_search import code_search

__all__ = ['search', 'code_search']