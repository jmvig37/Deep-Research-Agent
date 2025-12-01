from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime


class BasePrompt(ABC):
    """Abstract base class for all prompt types."""
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt.
        
        Returns:
            System prompt string
        """
        pass
    
    def get_user_prompt(self, **kwargs) -> Optional[str]:
        """Get the user prompt if applicable.
        
        Args:
            **kwargs: Prompt-specific parameters
            
        Returns:
            User prompt string or None if not applicable
        """
        return None


class SearchQueryGenerationPrompt(BasePrompt):
    """Prompt for generating initial search queries."""
    
    def __init__(self, num_searches: int):
        """Initialize the search query generation prompt.
        
        Args:
            num_searches: Number of search queries to generate
        """
        self.num_searches = num_searches
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for generating search queries."""
        current_year = datetime.now().year
        return f"""You are a research assistant. Given a user's query, generate {self.num_searches} diverse and specific search queries that will help gather comprehensive information to answer the question.

IMPORTANT: For factual questions (dates, stats, historical seasons, etc.), you MUST generate search queries to find this information. Do NOT rely on memory - always search for factual data.

The tavily_search tool will be used to execute these queries. This tool searches the web for current information, facts, dates, statistics, and recent developments. For ANY factual question, you must use this tool rather than relying on memory.

CRITICAL INSTRUCTIONS:
- Produce {self.num_searches} distinct search queries that explore different angles of the topic
- Do NOT generate paraphrases - each query should target a different aspect or sub-question
- Each query should be unique and explore a different dimension of the topic
- Always generate time-specific queries using the current year ({current_year})
- Do NOT use {current_year - 2} or {current_year - 1} unless the question is explicitly historical

Generate queries that:
- Cover different aspects of the topic (not just rewordings of the same question)
- Use specific, searchable terms
- Are likely to return relevant results
- When the question is about current events, trends, or ongoing developments, include recency terms when appropriate (e.g., "2025", "recent", "latest", "current").
- When the question is about historical events or long-term implications, focus on clear topic terms rather than recency indicators.

IMPORTANT: For queries about current events, trends, or developments, always include recency indicators to ensure you get the most up-to-date information.

Return ONLY a JSON object with this example structure:
{{
  "queries": ["query 1", "query 2", "query 3", "query n"]
}}
"""
    
    def get_user_prompt(self, query: str) -> str:
        """Get the user prompt for generating search queries.
        
        Args:
            query: The original user query
            
        Returns:
            User prompt for search query generation
        """
        return f"User query: {query}\n\nGenerate {self.num_searches} search queries."
class RefineSearchesPrompt(BasePrompt):
    """Prompt for refining searches based on initial results."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for refining searches."""
        return """You are a research analyst. Analyze the search results provided and identify specific unanswered or under-answered sub-questions.

Return ONLY a JSON object with this exact structure:
{
  "should_continue": true/false,
  "reason": "brief explanation of your decision",
  "time_range": "day" | "week" | "month" | "year" | null,
  "topic_summaries": ["brief summary of gap 1", "brief summary of gap 2", ...],
  "refined_queries": ["query 1", "query 2", ...]
}

CRITICAL GOAL:
- Your goal is to determine whether the EXISTING RESULTS are already enough to answer the ORIGINAL QUERY WELL.
- You should prefer to STOP (should_continue = false) unless there is a CLEAR, IMPORTANT gap that prevents writing a good answer to the original question.

Rules:
- Set "should_continue" to false by DEFAULT.
- Only set "should_continue" to true if:
  - There is a clearly IDENTIFIABLE sub-question that is necessary to answer the original query, AND
  - That sub-question is NOT well covered by the current results.
- Do NOT request more research just to add extra color, trivia, or minor details that are not required by the original question.
- If the original question asks for:
  - a specific list, OR
  - a small set of factual fields (e.g., opponents and series results),
  and the current results already contain consistent answers for each required item,
  then you MUST set "should_continue" to false.

More concrete rules:
- If you can already write a correct, well-supported answer to the original question, set "should_continue" to false,
  even if there are additional interesting details you could still look up.

Example when more research is needed:
{
  "should_continue": true,
  "reason": "Missing information about playoff opponents in first round and conference finals",
  "time_range": null,
  "topic_summaries": ["First round playoff opponents", "Conference finals opponents"],
  "refined_queries": ["Boston Bruins 2011 first round playoff opponents", "Boston Bruins 2011 conference finals"]
}

Example when research is sufficient:
{
  "should_continue": false,
  "reason": "All key subquestions answered - comprehensive information found",
  "time_range": null,
  "topic_summaries": [],
  "refined_queries": []
}"""
    
    def get_user_prompt(self, query: str, results_summary: str, previous_queries: Optional[List[str]] = None) -> str:
        """Get the user prompt for refining searches.
        
        Args:
            query: The original user query
            results_summary: Formatted summary of current search results
            previous_queries: List of previously executed queries (for context)
            
        Returns:
            User prompt for search refinement
        """
        previous_queries_text = ""
        if previous_queries:
            previous_queries_text = f"\n\nPreviously executed queries:\n" + "\n".join(f"- {q}" for q in previous_queries[:10])  # Show up to 10
            if len(previous_queries) > 10:
                previous_queries_text += f"\n... and {len(previous_queries) - 10} more"
        
        return f"""Original query: {query}

Current search results:
{results_summary}{previous_queries_text}

Given the existing results, identify up to 3 specific unanswered or under-answered sub-questions. For each sub-question, generate one targeted search query. Do not repeat queries that are semantically equivalent to previous ones."""


class ReportGenerationPrompt(BasePrompt):
    """Prompt for generating the final research report."""
    
    def __init__(self, max_length: int):
        """Initialize the report generation prompt.
        
        Args:
            max_length: Maximum word count for the report
        """
        self.max_length = max_length
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for generating the final research report."""
        return f"""You are an expert research analyst. Your task is to synthesize information from multiple web sources to create a comprehensive, well-structured research report.

CRITICAL REQUIREMENTS:
1.You MUST prefer authoritative sources (.gov, .edu, academic publications, major news outlets). If a high-authority source exists in the results, use it INSTEAD OF lower-quality ones in all cases (YouTube, social media, student notes, commerce sites, or user-generated content).
2. You MUST ONLY use information from the provided search results below. Do NOT use any knowledge from your training data.
3. EVERY claim, fact, or statement MUST be cited using [Source N] notation where N is the source number.
4. If information is not in the provided sources, do NOT include it in the report.
5. If nothing relevant is found in the search results, you MUST say you don't know rather than making up information.
6. When the research question is about current events, trends, or recent developments, prioritize recent information. When the question is about historical events, long-term implications, or fixed facts, prioritize the most authoritative and relevant sources, even if they are older.
7. The search results are from recent web searches - use this current information, not outdated training data.

CAUSAL RELATIONSHIP CONSTRAINTS:
- Only state causal relationships (X led to Y, X influenced Y, X caused Y) when at least two independent sources support that connection.
- If sources don't clearly say X caused Y, describe the relationship more cautiously using phrases like "coincided with", "occurred after", "was related to", "was associated with", or "happened alongside".
- If you are not confident about a causal claim, either omit it or explicitly flag it as uncertain (e.g., "may have contributed to", "possibly influenced", "uncertain relationship").
- Avoid making strong causal claims based on correlation alone or single-source evidence.

SOURCE QUALITY PREFERENCES:
- Prefer citing authoritative sources (e.g., official organizations, academic sites, established news outlets, reputable reference sites) over social media posts, forums, or low-credibility sites.
- Only rely on social media, forums, or user-generated content when higher-quality sources are unavailable or when the question explicitly concerns opinions or social media content.
- When multiple sources cover the same information, prioritize the most authoritative source.
- If you must use lower-credibility sources, acknowledge their limitations in your report.
- When multiple sources provide the same information, choose the most authoritative source even if it is not the most recent.

Your report structure:
1. Executive Summary (2-3 sentences) - cite sources
2. Detailed Analysis organized into clear sections - cite sources for every claim
3. Key Findings and Insights - cite sources
4. Conclusion - cite sources
5. Sources Section - list all URLs used

Citation format: Use [Source 1], [Source 2], etc. after every claim or fact.

- In the "Key Findings and Insights" section, list each key finding as a bullet point. Each bullet should:
  - Make one clear claim or insight
  - Be supported by at least one [Source N] citation
  - Optionally indicate your confidence level (e.g., "High confidence", "Medium confidence").

Keep the report concise but thorough (approximately {self.max_length} words).
"""
    
    def get_user_prompt(self, query: str, results_text: str, sources_section: str) -> str:
        """Get the user prompt for generating the final research report.
        
        Args:
            query: The original research question
            results_text: Formatted search results
            sources_section: Formatted sources with URLs
            
        Returns:
            User prompt for report generation
        """
        return f"""Research Question: {query}

IMPORTANT: You have been provided with recent web search results below. You MUST base your entire report ONLY on these search results. Do NOT use information from your training data.

CRITICAL RULES:
- For factual questions (dates, stats, historical seasons, etc.), you must use information from the search results
- You must not answer purely from memory
- Only answer based on information found in the search results
- If nothing relevant is found, say you don't know

CAUSAL RELATIONSHIP RULES:
- Only state causal relationships (X led to Y, X influenced Y) when at least two independent sources support that connection
- If sources don't clearly say X caused Y, describe the relationship more cautiously ("coincided with", "occurred after", "was related to")
- If you are not confident about a causal claim, either omit it or explicitly flag it as uncertain

SOURCE QUALITY RULES:
- Prefer citing authoritative sources (e.g., official orgs, academic sites, established news outlets, reputable reference sites) over social media posts, forums, or low-credibility sites
- Only rely on social media, forums, or user-generated content when higher-quality sources are unavailable or when the question explicitly concerns opinions or social media content

Search Results from Web:
{results_text}

Sources with URLs:
{sources_section}

Generate a comprehensive research report based EXCLUSIVELY on the search results provided above. 
- Cite every claim with [Source N] notation
- Include the Sources Section at the end with all URLs
- Do not make up information or use knowledge from training data
- All information must come from the provided search results
- If the search results don't contain relevant information, explicitly state that you don't know"""


class ErrorReportPrompt(BasePrompt):
    """Prompt template for error reports when no search results are found."""
    
    def get_system_prompt(self) -> str:
        """Error reports don't use system prompts."""
        return ""
    
    def get_user_prompt(self, query: str) -> str:
        """Get the error report template.
        
        Args:
            query: The original research question
            
        Returns:
            Error report template
        """
        return f"""# Research Report: {query}

## Executive Summary
Unable to generate a comprehensive report as no search results were retrieved. This may be due to:
- Missing or invalid Tavily API key
- Network connectivity issues
- Search query returning no results

Please check your Tavily API key in the .env file and ensure it is valid.

## Sources Section
No sources available - search functionality did not return any results."""

# ============================================================================
# Evaluation Prompts
# ============================================================================
class JudgePrompt(BasePrompt):
    """Prompt for LLM judge evaluation of agent reports."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the judge."""
        return """You are an expert evaluator assessing the quality of a research report.

Always respond with valid JSON only. Do not include any text outside the JSON object."""

    def get_user_prompt(self, prompt: str, agent_report: str, gold_response: str) -> str:
        """Get the user prompt for judge evaluation.
        
        Args:
            prompt: Original user query
            agent_report: Agent's generated report
            gold_response: Expected/gold standard response
            
        Returns:
            Formatted prompt for the judge
        """
        return f"""You are an expert evaluator assessing the quality of a research report.

Original Query: {prompt}

Gold Standard Response:
{gold_response}

Agent's Generated Report:
{agent_report}

Evaluate the agent's report on the following criteria:
1. **Accuracy**: Does the report contain accurate information that aligns with the gold standard?
2. **Completeness**: Does the report cover the key points from the gold standard?
3. **Relevance**: Is the information directly relevant to the original query?
4. **Citation Quality**: Are sources properly cited and relevant?

Provide your evaluation as a JSON object with this structure:
{{
    "score": <0-100 integer>,
    "accuracy": <0-100 integer>,
    "completeness": <0-100 integer>,
    "relevance": <0-100 integer>,
    "citation_quality": <0-100 integer>,
    "reasoning": "<brief explanation of your scores>"
}}"""

