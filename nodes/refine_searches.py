"""Node for refining searches based on gap analysis."""

from typing import TypedDict, List, Tuple, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable
import numpy as np

from models import RefinementDecision
from prompts import RefineSearchesPrompt
from tools.searchtool import create_tavily_search_tool


def _deduplicate_queries(
    new_queries: List[str],
    previous_queries: List[str],
    embeddings,
    embedding_cache: dict,
    similarity_threshold: float,
) -> List[str]:
    """Remove queries that are semantically similar to previous ones using embeddings.
    
    Args:
        new_queries: New queries to check
        previous_queries: Previously executed queries
        embeddings: Embeddings model instance
        embedding_cache: Cache for embeddings
        similarity_threshold: Threshold for similarity (0-1)
        
    Returns:
        List of deduplicated queries
    """
    if not previous_queries or not new_queries:
        return new_queries
    
    try:
        all_queries = previous_queries + new_queries
        embeddings_list = []
        uncached_queries = []
        uncached_indices = []
        
        # Check cache first
        for i, query in enumerate(all_queries):
            if query in embedding_cache:
                embeddings_list.append(embedding_cache[query])
            else:
                uncached_queries.append(query)
                uncached_indices.append(i)
                embeddings_list.append(None)  # Placeholder
        
        # Get embeddings for uncached queries
        if uncached_queries:
            new_embeddings = embeddings.embed_documents(uncached_queries)
            # Store in cache and fill in placeholders
            for query, embedding, idx in zip(uncached_queries, new_embeddings, uncached_indices):
                embedding_cache[query] = embedding
                embeddings_list[idx] = embedding
        
        # Split embeddings
        prev_embeddings = np.array(embeddings_list[:len(previous_queries)])
        new_embeddings = np.array(embeddings_list[len(previous_queries):])
        
        # Normalize embeddings for cosine similarity
        # Handle edge case where we have a single embedding
        if len(prev_embeddings.shape) == 1:
            prev_embeddings = prev_embeddings.reshape(1, -1)
        if len(new_embeddings.shape) == 1:
            new_embeddings = new_embeddings.reshape(1, -1)
        
        prev_norms = prev_embeddings / (np.linalg.norm(prev_embeddings, axis=1, keepdims=True) + 1e-8)
        new_norms = new_embeddings / (np.linalg.norm(new_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity between new queries and previous queries
        # Shape: (len(new_queries), len(previous_queries))
        similarity_matrix = np.dot(new_norms, prev_norms.T)
        
        # Filter out queries that are too similar to any previous query
        deduplicated = []
        for i, query in enumerate(new_queries):
            # Check if this query is similar to any previous query
            max_similarity = float(np.max(similarity_matrix[i]))
            if max_similarity < similarity_threshold:
                deduplicated.append(query)
        
        return deduplicated
    except Exception as e:
        # If embedding fails, return all new queries (fallback)
        print(f"  ⚠️  Embedding deduplication failed: {str(e)}, keeping all queries")
        return new_queries


@traceable(name="refine_searches")
def refine_searches(
    state: TypedDict,
    llm,
    config,
    embeddings,
    embedding_cache: dict,
) -> Tuple[TypedDict, Optional[object]]:
    """Analyze search results and generate refined search queries to fill gaps.
    
    Args:
        state: Current agent state
        llm: Language model instance
        config: Agent configuration
        embeddings: Embeddings model instance
        embedding_cache: Cache for embeddings
        
    Returns:
        Tuple of (updated agent state, new_tavily_search_tool or None)
    """
    query = state["query"]
    search_results = state.get("search_results", [])
    refinement_iterations = state.get("refinement_iterations", 0)
    
    print("🔎 Analyzing search results for gaps...")
    
    # Increment refinement counter
    refinement_iterations += 1
    state["refinement_iterations"] = refinement_iterations
    
    # If no results, skip refinement
    if not search_results:
        print("  No results to analyze, skipping refinement")
        state["messages"].append(
            AIMessage(content="Skipping search refinement - no initial results to analyze.")
        )
        state["search_queries"] = []  # Clear queries to stop loop
        state["should_continue_research"] = False
        return state, None
    
    # Check if we've hit the iteration limit BEFORE asking the model
    if refinement_iterations >= config.num_refinement_iterations:
        print(f"  Reached max refinement iterations ({config.num_refinement_iterations}); proceeding to report generation")
        state["messages"].append(
            AIMessage(content=f"Reached maximum refinement iterations ({config.num_refinement_iterations}). Proceeding to report generation.")
        )
        state["search_queries"] = []  # Clear queries to stop loop
        state["should_continue_research"] = False
        return state, None
    
    # Format search results for analysis
    # Limit the number of results to avoid overwhelming context
    results_to_analyze = search_results[:config.max_results_for_report]
    results_text_parts = []
    for i, result in enumerate(results_to_analyze, 1):
        title = result.title
        content = result.content
        # Truncate content to a reasonable length
        content_summary = content[:300] + "..." if len(content) > 300 else content
        results_text_parts.append(f"Result {i}: {title} - {content_summary}")
    
    if len(search_results) > len(results_to_analyze):
        results_text_parts.append(f"... and {len(search_results) - len(results_to_analyze)} more results")
    
    results_text = "\n".join(results_text_parts) if results_text_parts else "No results available."
    
    # Get previous queries for context and deduplication
    previous_queries = state.get("all_queries", [])
    
    prompt = RefineSearchesPrompt()
    system_prompt = prompt.get_system_prompt()
    user_prompt = prompt.get_user_prompt(query, results_text, previous_queries)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]             

    try:
        structured_llm = llm.with_structured_output(RefinementDecision)
        result = structured_llm.invoke(messages)
        should_continue = result.should_continue
        reason = result.reason
        time_range = result.time_range
        topic_summaries = result.topic_summaries
        refined_queries = result.refined_queries
    except Exception as e:
        # If parsing fails, default to no more research needed
        state["messages"].append(
            AIMessage(content=f"Error analyzing results for refinement: {str(e)}. Proceeding with current results.")
        )
        refined_queries = []
        topic_summaries = []
        should_continue = False
        reason = "Error in refinement analysis"
        time_range = None
    

    # Check if we can actually continue (haven't hit max iterations)
    can_continue = refinement_iterations < config.num_refinement_iterations
    
    # Deduplicate queries against all previously executed queries
    if refined_queries and can_continue:
        # Get all previous queries from state
        all_previous_queries = state.get("all_queries", [])
        original_count = len(refined_queries)
        refined_queries = _deduplicate_queries(
            refined_queries,
            all_previous_queries,
            embeddings,
            embedding_cache,
            config.query_similarity_threshold,
        )
        
        if original_count > len(refined_queries):
            filtered_count = original_count - len(refined_queries)
            print(f"  Filtered out {filtered_count} duplicate querie{'s' if filtered_count > 1 else ''}")
        
        if not refined_queries:
            # All queries were duplicates, no point continuing
            print("  All refined queries were duplicates of previous searches; research is complete")
            should_continue = False
            reason = "All suggested queries duplicate previous searches"
    
    # Store the should_continue flag, refined queries, and time_range
    state["should_continue_research"] = should_continue and can_continue
    state["search_queries"] = refined_queries if should_continue and can_continue else []
    # Update time_range if specified (only if continuing)
    new_tool = None
    if should_continue and can_continue:
        # Validate and update time_range
        if time_range is not None:
            # Validate time_range value
            valid_time_ranges = {"day", "week", "month", "year"}
            if time_range not in valid_time_ranges:
                print(f"  ⚠️  Invalid time_range '{time_range}', ignoring")
                time_range = None
        
        # Update time_range if specified (None means all time)
        if time_range is not None:
            # Recreate the search tool with the new time_range
            new_tool = create_tavily_search_tool(
                tavily_api_key=config.tavily_api_key,
                max_results=config.max_results_per_search,
                time_range=time_range
            )
            print(f"  Updated search time range to: {time_range}")
        # If time_range is None, keep current tool (no change needed)
    
    # Log the decision with proper context
    if should_continue and can_continue and refined_queries:
        time_range_text = f" (time_range: {time_range})" if time_range else ""
        print(f"  More research needed - generated {len(refined_queries)} refined queries{time_range_text}")
        print(f"    Reason: {reason}")
        topics_text = ""
        if topic_summaries:
            topics_text = f" Topics to research: {', '.join(topic_summaries[:2])}{'...' if len(topic_summaries) > 2 else ''}. "
        state["messages"].append(
            AIMessage(content=f"Refinement iteration {refinement_iterations}: More research needed. {reason}.{topics_text}Generated {len(refined_queries)} refined queries: {', '.join(refined_queries[:3])}{'...' if len(refined_queries) > 3 else ''}")
        )
    elif should_continue and not can_continue:
        # Model wants to continue but we've hit max iterations
        print(f"  Model suggests more research needed, but reached max iterations ({config.num_refinement_iterations}); proceeding to report generation")
        print(f"    Model's reason: {reason}")
        state["messages"].append(
            AIMessage(content=f"Refinement iteration {refinement_iterations}: Model suggested more research ({reason}), but reached maximum refinement iterations ({config.num_refinement_iterations}). Proceeding to report generation with current results.")
        )
    else:
        print(f"  Research is comprehensive - proceeding to report generation")
        print(f"    Reason: {reason}")
        state["messages"].append(
            AIMessage(content=f"Refinement iteration {refinement_iterations}: Research is comprehensive. {reason}. Proceeding to report generation.")
        )
    
    return state, new_tool

