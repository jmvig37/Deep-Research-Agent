"""Tavily search tool for web search functionality."""
from typing import Dict, Any, List, Optional
import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


def create_tavily_search_tool(
    tavily_api_key: Optional[str] = None,
    max_results: int = 5,
    time_range: Optional[str] = None
):
    """Factory function to create a Tavily search tool.
    
    Args:
        tavily_api_key: Tavily API key. If not provided, will attempt to load from TAVILY_API_KEY environment variable.
        max_results: Maximum number of results per search
        time_range: Time range for search ("day", "week", "month", "year", or None)
        
    Returns:
        Tool instance created with @tool decorator
        
    Raises:
        ValueError: If API key is not provided and not found in environment variables, or if initialization fails
    """
    # Handle API key - try parameter first, then environment variable
    if not tavily_api_key:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Validate API key
    if not tavily_api_key or not tavily_api_key.strip():
        raise ValueError(
            "Tavily API key is required. Please provide it as a parameter or set TAVILY_API_KEY in your .env file. "
            "Get an API key at https://tavily.com/"
        )
    
    # Initialize TavilySearch
    search_kwargs = {
        "max_results": max_results,
        "tavily_api_key": tavily_api_key
    }
    if time_range:
        search_kwargs["time_range"] = time_range
    
    try:
        tavily_search_instance = TavilySearch(**search_kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize Tavily search tool. "
            f"Please check your API key at https://tavily.com/ and ensure it's valid. "
            f"Error: {str(e)}"
        )
    
    @tool
    def tavily_search(query: str) -> List[Dict[str, Any]]:
        """Search the web for information using Tavily Search API.
        
        Use this tool when you need to:
        - Find current information, facts, dates, statistics, and recent developments
        - Answer factual questions that require up-to-date data
        - Research topics that may have changed since your training data
        - Get information about events, people, places, or concepts that need verification
        
        CRITICAL RULES:
        - For ANY factual question (dates, stats, historical data, current events), you MUST use this tool
        - Do NOT answer factual questions from memory - always search first
        - If search returns no results, say you don't know rather than guessing
        - Use specific, searchable query terms for best results
        
        Args:
            query: A search query string (e.g., "quantum computing breakthroughs 2025", "Boston Bruins 2025 season stats")
            
        Returns:
            List of search result dictionaries, or error dict if search fails
        """
        try:
            results = tavily_search_instance.invoke({"query": query})
            
            if not results:
                return []
            
            # TavilySearch returns a dict with 'results' key or a list
            if isinstance(results, dict):
                return results.get('results', [])
            elif isinstance(results, list):
                return results
            else:
                return []
        except Exception as e:
            return [{
                "error": True,
                "message": f"Tavily search failed: {str(e)}"
            }]
    
    return tavily_search

