"""Node for executing web searches."""

from typing import TypedDict, List
from langchain_core.messages import AIMessage
from langsmith import traceable

from models import SearchResult
from util import process_search_result, sort_by_relevance


def _execute_single_search(
    query: str,
    tavily_search_tool,
) -> List[SearchResult]:
    """Execute a single search query and return processed results.
    
    Args:
        query: Search query string
        tavily_search_tool: Tavily search tool instance
        
    Returns:
        List of processed search results
    """
    try:
        # Use the tool to execute search
        results_list = tavily_search_tool.invoke(query)
        
        if not results_list:
            return []
        
        # Check for error responses from the tool
        # Error format: [{"error": True, "message": "..."}]
        if len(results_list) == 1 and isinstance(results_list[0], dict) and results_list[0].get("error") is True:
            # Tool returned an error, log it and return empty
            error_msg = results_list[0].get("message", "Unknown error")
            print(f"  ⚠️  Search error: {error_msg}")
            return []
        
        # Process and validate results
        processed_results = []
        for r in results_list:
            # Skip error dicts (format: {"error": True, "message": "..."})
            if isinstance(r, dict) and r.get("error") is True:
                continue
            processed = process_search_result(r)
            if processed:
                processed_results.append(processed)
        
        if not processed_results:
            return []
        
        # Sort by relevance
        return sort_by_relevance(processed_results)
        
    except Exception as e:
        # Log error but don't fail the entire search
        print(f"  ⚠️  Search execution error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


@traceable(name="execute_searches")
def execute_searches(
    state: TypedDict,
    tavily_search_tool,
) -> TypedDict:
    """Execute web searches using the generated queries and merge results into state.
    
    Args:
        state: Current agent state
        tavily_search_tool: Tavily search tool instance
        
    Returns:
        Updated agent state
    """
    search_queries = state.get("search_queries", [])
    if not search_queries:
        # Fallback to original query if no queries generated
        search_queries = [state["query"]]

    # Check if this is a refinement iteration
    refinement_iterations = state.get("refinement_iterations", 0)

    if refinement_iterations > 0:
        print(f"🔄 Refining search (iteration {refinement_iterations})...")
        state["messages"].append(
            AIMessage(content=f"Executing refined searches (iteration {refinement_iterations})")
        )
    else:
        print(f"🛜 Executing {len(search_queries)} web searches...")

    # Get existing results to merge with (for refinement loops)
    existing_results = state.get("search_results", [])
    existing_urls = {r.url for r in existing_results}

    # Track queries in all_queries for deduplication/debug
    state["all_queries"].extend(search_queries)

    # Execute all searches and collect raw results from this iteration
    all_results: List[SearchResult] = []
    for i, query in enumerate(search_queries, 1):
        print(f"  [{i}/{len(search_queries)}] Searching: {query[:60]}{'...' if len(query) > 100 else ''}")
        query_results = _execute_single_search(query, tavily_search_tool)

        if query_results:
            all_results.extend(query_results)
            state["messages"].append(
                AIMessage(
                    content=f"Search executed for: '{query}' - Found {len(query_results)} valid results"
                )
            )
        else:
            state["messages"].append(
                AIMessage(content=f"Search executed for: '{query}' - No results returned")
            )

    # Merge new results with existing results, avoiding duplicates by URL
    previous_total = len(existing_results)
    if all_results:
        # Sort raw results by relevance before merging (all_results are SearchResult models)
        all_results = sort_by_relevance(all_results)

        for result in all_results:
            url = result.url
            if url and url not in existing_urls:
                existing_results.append(result)
                existing_urls.add(url)

    # Sort final results by relevance
    if existing_results:
        existing_results = sort_by_relevance(existing_results)

    if not existing_results:
        # No results at all – warn and keep state consistent
        print("⚠️  Warning: No search results found")
        state["messages"].append(
            AIMessage(
                content=(
                    "Warning: No search results were found. "
                    "Report will be generated with limited information."
                )
            )
        )
    else:
        new_total = len(existing_results)
        deduplicated_count = new_total - previous_total
        total_count = new_total

        if deduplicated_count > 0:
            print(f"✓ Added {deduplicated_count} unique results (total: {total_count})")
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Search execution completed — added {deduplicated_count} unique results. "
                        f"Total results: {total_count}"
                    )
                )
            )
        else:
            print(f"✓ No unique new results this iteration (total: {total_count})")
            state["messages"].append(
                AIMessage(
                    content=(
                        "Search execution completed — no unique new results this iteration. "
                        f"Total results: {total_count}"
                    )
                )
            )

    state["search_results"] = existing_results
    return state

