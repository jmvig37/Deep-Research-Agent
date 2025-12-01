"""Node for generating the final research report."""

from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable
import re

from models import SearchResult
from prompts import ReportGenerationPrompt, ErrorReportPrompt
from util import (
    select_top_results,
    format_sources_section,
    extract_cited_sources,
    filter_sources_by_citations,
    has_sources_section,
)


def _summarize_result_for_prompt(
    result: SearchResult,
    llm,
) -> str:
    """Summarize a search result using LLM for inclusion in the prompt.
    
    Args:
        result: SearchResult model
        llm: Language model instance
        
    Returns:
        Summarized content string
    """
    # If content is short enough, return as-is
    if len(result.content) <= 800:
        return result.content
    
    # Use LLM to summarize longer content
    try:
        summary_prompt = f"""Summarize the following content in a concise way (aim for ~500-800 characters) while preserving key facts and information:
            Title: {result.title}
            URL: {result.url}
            Content: {result.content}"""
            
        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes content concisely while preserving key information."),
            HumanMessage(content=summary_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        # Simple truncation if LLM call fails
        return result.content[:800] + "..."


@traceable(name="generate_report")
def generate_report(
    state: TypedDict,
    llm,
    config,
) -> TypedDict:
    """Generate a comprehensive report based on search results.
    
    Args:
        state: Current agent state
        llm: Language model instance
        config: Agent configuration
        
    Returns:
        Updated agent state
    """
    query = state["query"]
    search_results = state.get("search_results", [])
    
    print("📝 Generating research report...")
    
    # Check if we have any results
    if not search_results:
        error_prompt = ErrorReportPrompt()
        error_report = error_prompt.get_user_prompt(query)
        state["messages"].append(AIMessage(content=error_report))
        state["report"] = error_report
        return state
    
    # Select top results for report generation
    top_results = select_top_results(search_results, limit=config.max_results_for_report)
    
    # Store original indices in results for later reference
    for i, result in enumerate(top_results, 1):
        result.original_index = i
    
    # Format results for prompt with LLM summarization
    formatted_results = []
    for i, result in enumerate(top_results, 1):
        summary = _summarize_result_for_prompt(result, llm)
        formatted_results.append(f"[Source {i}] URL: {result.url}\nTitle: {result.title}\nContent: {summary}\n")
    results_text = "\n\n".join(formatted_results) if formatted_results else "No search results available."
    
    # Format sources section (initial numbering based on max_results_for_report)
    sources_section = format_sources_section(top_results)
    
    # Generate report
    prompt = ReportGenerationPrompt(config.max_report_length_words)
    messages = [
        SystemMessage(content=prompt.get_system_prompt()),
        HumanMessage(content=prompt.get_user_prompt(query, results_text, sources_section))
    ]
    
    response = llm.invoke(messages)
    report = response.content
    
    # Extract which sources were actually cited in the report body
    cited_sources = extract_cited_sources(report)
    
    # Filter results to only include cited sources, preserving original indices
    if cited_sources:
        cited_results = filter_sources_by_citations(top_results, cited_sources)
        # Re-format sources section with only cited sources, preserving original indices
        filtered_sources_section = format_sources_section(cited_results, preserve_original_indices=True)
    else:
        # If no citations found, use all sources (fallback - shouldn't happen if LLM follows instructions)
        filtered_sources_section = sources_section
    
    # Replace or add the sources section with filtered version
    if has_sources_section(report):
        # Replace existing sources section with filtered one
        # Match from ## Sources or # Sources to end of document or next ## heading
        sources_pattern = r'(##\s+Sources?[^\#]*(?=\n##|\Z))'
        replacement = f"## Sources\n\n{filtered_sources_section}"
        report = re.sub(sources_pattern, replacement, report, flags=re.IGNORECASE | re.DOTALL)
        
        # Also try single # pattern
        if "## Sources" not in report and "# Sources" in report:
            sources_pattern2 = r'(#\s+Sources?[^\#]*(?=\n##|\Z))'
            report = re.sub(sources_pattern2, replacement, report, flags=re.IGNORECASE | re.DOTALL)
    elif filtered_sources_section != "No sources available.":
        # Add sources section if LLM didn't include it
        report += f"\n\n## Sources\n\n{filtered_sources_section}"
    
    # Store report in state
    state["messages"].append(AIMessage(content=report))
    state["report"] = report
    
    print("✓ Report generated successfully")
    
    return state

