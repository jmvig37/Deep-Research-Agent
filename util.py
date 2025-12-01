"""Utility functions for the deep research agent."""
from typing import List, Dict, Any, Optional, Set
import re
from models import SearchResult


# ============================================================================
# Search Result Processing
# ============================================================================

def process_search_result(result: Dict[str, Any]) -> Optional[SearchResult]:
    """Process a single search result into pydantic model.
    
    Args:
        result: Raw search result from Tavily
        
    Returns:
        Processed SearchResult model or None if invalid
    """
    if not isinstance(result, dict):
        return None
    
    # Check for error
    if result.get("error") is True:
        return None
    
    # Validate that we have at least some content
    # Tavily results might have 'content', 'raw_content', or 'snippet'
    has_content = result.get('content') or result.get('raw_content') or result.get('snippet')
    has_url = result.get('url')
    
    if not (has_url or has_content):
        return None
    
    try:
        return SearchResult(
            url=result.get('url', ''),
            title=result.get('title', result.get('name', 'No title')),
            content=result.get('content') or result.get('raw_content') or result.get('snippet', ''),
            score=result.get('score', 0.0),
            published_date=result.get('published_date')
        )
    except Exception:
        return None


def sort_by_relevance(results: List[SearchResult]) -> List[SearchResult]:
    """Sort search results by relevance score.
    
    Args:
        results: List of SearchResult models
        
    Returns:
        Sorted list of results (most relevant first)
    """
    def sort_key(result: SearchResult) -> float:
        return result.score
    return sorted(results, key=sort_key, reverse=True)


def select_top_results(search_results: List[SearchResult], limit: int = 15) -> List[SearchResult]:
    """Select the top N most relevant search results.
    
    Args:
        search_results: List of SearchResult models
        limit: Maximum number of results to return
        
    Returns:
        Top N results sorted by relevance (most relevant first)
    """
    sorted_results = sort_by_relevance(search_results)
    return sorted_results[:limit]


# ============================================================================
# Report Formatting
# ============================================================================



def format_sources_section(results: List[SearchResult], preserve_original_indices: bool = False) -> str:
    """Format sources section for the prompt.
    
    Args:
        results: List of SearchResult models
        preserve_original_indices: If True, use original_index from results instead of renumbering
        
    Returns:
        Formatted sources section string
    """
    source_urls = []
    for i, result in enumerate(results, 1):
        url = result.url
        title = result.title
        if url:
            # Use original_index if preserving indices, otherwise use position
            if preserve_original_indices and result.original_index is not None:
                source_num = result.original_index
            else:
                source_num = i
            source_urls.append(f"{source_num}. {title}\n   URL: {url}")
    return "\n\n".join(source_urls) if source_urls else "No sources available."


# ============================================================================
# Citation Extraction and Filtering
# ============================================================================

def extract_cited_sources(report: str) -> Set[int]:
    """Extract source numbers that are actually cited in the report body.
    
    Args:
        report: Generated report text
        
    Returns:
        Set of source numbers (1-indexed) that are cited in the report
    """
    # Find all citations like [Source 1], [Source 2], etc.
    # Also handle variations like Source 1, Source 2 (without brackets)
    cited_sources = set()
    
    # Pattern for [Source N] or [Source N] (most common format)
    bracket_pattern = r'\[Source\s+(\d+)\]'
    # Pattern for Source N (without brackets, but in context)
    text_pattern = r'(?:^|\s)Source\s+(\d+)(?:\s|$|,|\.)'
    
    # Find all matches
    for match in re.finditer(bracket_pattern, report, re.IGNORECASE):
        cited_sources.add(int(match.group(1)))
    
    for match in re.finditer(text_pattern, report, re.IGNORECASE):
        cited_sources.add(int(match.group(1)))
    
    return cited_sources


def filter_sources_by_citations(
    results: List[SearchResult], 
    cited_sources: Set[int]
) -> List[SearchResult]:
    """Filter results to only include sources that were cited in the report.
    Preserves original source indices.
    
    Args:
        results: All search results (1-indexed by position, with original_index preserved)
        cited_sources: Set of source numbers that were cited
        
    Returns:
        Filtered list of results that were actually cited, with original_index preserved
    """
    if not cited_sources:
        return results  # If no citations found, return all (fallback)
    
    # Filter to only include cited sources (1-indexed)
    filtered = []
    for i, result in enumerate(results, 1):
        if i in cited_sources:
            if result.original_index is None:
                result.original_index = i
            filtered.append(result)
    
    return filtered


def has_sources_section(report: str) -> bool:
    """Check if report already contains a sources section.
    
    Args:
        report: Generated report text
        
    Returns:
        True if sources section exists, False otherwise
    """
    report_lower = report.lower()
    return (
        "## sources" in report_lower or
        "# sources" in report_lower or
        "sources section" in report_lower or
        "sources:" in report_lower or
        "## references" in report_lower or
        "# references" in report_lower or
        "references:" in report_lower or
        "sources list" in report_lower or
        "list of sources" in report_lower
    )

