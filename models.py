"""Pydantic models for structured outputs."""
from typing import List, Optional
from pydantic import BaseModel, Field
from config import AgentConfig

class SearchQueries(BaseModel):
    """Structured output for search query generation."""
    queries: List[str] = Field(min_items=AgentConfig().num_searches, max_items=AgentConfig().num_searches,
        description="List of diverse search queries to gather comprehensive information"
    )


class RefinementDecision(BaseModel):
    """Structured output for search refinement analysis."""
    should_continue: bool = Field(
        description="Whether to continue with more research iterations. Set to false when all key subquestions are answered."
    )
    reason: str = Field(
        description="Brief explanation for the decision (e.g., 'All key subquestions answered' or 'Missing information about X')"
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range for refined searches: 'day', 'week', 'month', 'year', or null. Use null to search all time. Only set if should_continue is true and you need to focus on a specific time period."
    )
    topic_summaries: List[str] = Field(
        default_factory=list,
        description="Brief summaries of topics/gaps that need more research (only if should_continue is true)"
    )
    refined_queries: List[str] = Field(
        default_factory=list,
        description="Refined search queries to gather missing information (only if should_continue is true)"
    )


class SearchResult(BaseModel):
    """Structured representation of a search result."""
    url: str
    title: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None

