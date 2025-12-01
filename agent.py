from typing import TypedDict, Annotated, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from operator import add
import re
import numpy as np

from models import SearchQueries, RefinementDecision, ResearchResult, SearchResult
from tools.searchtool import create_tavily_search_tool
from util import (
    process_search_result,
    sort_by_relevance,
    select_top_results,
    format_sources_section,
    extract_cited_sources,
    filter_sources_by_citations,
    has_sources_section,
)

from config import AgentConfig
from prompts import (
    SearchQueryGenerationPrompt,
    RefineSearchesPrompt,
    ReportGenerationPrompt,
    ErrorReportPrompt,
)


class AgentState(TypedDict):
    """State for the deep research agent.
    The messages field serves dual purposes:
    - Context: Messages that provide context to the LLM (user query, system prompts)
    - Log: Messages that log agent actions for debugging/monitoring (search execution, refinement decisions)
    """
    # User input + final report live here
    messages: Annotated[List[BaseMessage], add]

    # Core task metadata
    query: str                          # canonical user research question
    report: str                         # final synthesized report text

    # Search state
    search_results: List[SearchResult]    # accumulated + deduped results

    # Refinement loop state
    refinement_iterations: int           # how many refinement passes we've done
    should_continue_research: bool       # set by refine node
    search_queries: List[str]            # queries to run in the next execute_searches step
    all_queries: List[str]               # all queries ever run (for dedupe/debug)

class DeepResearchAgent:
    """A deep research agent that uses web search to generate comprehensive reports."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration.
        
        Args:
            config: Agent configuration settings
        """
        self.config = config
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
        
        # Initialize embeddings for query deduplication
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.openai_api_key
        )
        
        # Embedding cache for query strings
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Initialize tool
        self.tavily_search_tool = create_tavily_search_tool(
            tavily_api_key=config.tavily_api_key,
            max_results=config.max_results_per_search,
            time_range=None
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the agent's state graph."""
        agent_graph = StateGraph(AgentState)
        
        # Add agent nodes
        agent_graph.add_node("generate_search_queries", self._generate_search_queries)
        agent_graph.add_node("execute_searches", self._execute_searches)
        agent_graph.add_node("refine_searches", self._refine_searches)
        agent_graph.add_node("generate_report", self._generate_report)
        
        # Set entry point
        agent_graph.set_entry_point("generate_search_queries")
        
        # Add edges defining agent flow
        agent_graph.add_edge("generate_search_queries", "execute_searches")
        agent_graph.add_edge("execute_searches", "refine_searches")
        # Conditional edge: loop back to execute_searches if more refinements needed, else generate report
        agent_graph.add_conditional_edges(
            "refine_searches",
            self._should_continue_refining,
            {
                "continue": "execute_searches",
                "finish": "generate_report"
            }
        )
        agent_graph.add_edge("generate_report", END)
        
        return agent_graph.compile()
    
    def _should_continue_refining(self, state: AgentState) -> str:
        """Determine if we should continue refining searches or proceed to report generation.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" if we should refine again, "finish" if we should generate the report
        """
        refinement_iterations = state.get("refinement_iterations", 0)
        should_continue_research = state.get("should_continue_research", False)
        search_queries = state.get("search_queries", [])
        
        # Continue only if:
        # 1. We haven't hit the iteration limit
        # 2. The LLM determined more research is needed
        # 3. There are refined queries to execute
        if refinement_iterations < self.config.num_refinement_iterations and should_continue_research and search_queries:
            return "continue"
        else:
            return "finish"
    
    @traceable(name="generate_search_queries")
    def _generate_search_queries(self, state: AgentState) -> AgentState:
        """Generate search queries based on the user's question using structured outputs."""
        query = state["query"]
        
        print("🔍 Generating search queries...")
        
        prompt = SearchQueryGenerationPrompt(self.config.num_searches)
        system_prompt = prompt.get_system_prompt()
        user_prompt = prompt.get_user_prompt(query)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            # Use structured output
            structured_llm = self.llm.with_structured_output(SearchQueries)
            result = structured_llm.invoke(messages)
            search_queries = result.queries
                        
        except Exception as e:
            # Fallback: use the original query with variations
            search_queries = [f"{query} aspect {i+1}" for i in range(self.config.num_searches)]
            state["messages"].append(
                AIMessage(content=f"Error generating search queries, using fallback: {str(e)}")
            )
        
        # Store queries in state
        search_queries = search_queries[:self.config.num_searches]
        state["search_queries"] = search_queries
        
        # Track all queries executed so far for deduplication
        state["all_queries"].extend(search_queries)
        
        print(f"✓ Generated {len(search_queries)} search queries")
        
        # Log query generation output in messages
        state["messages"].append(
            AIMessage(content=f"Generated {len(search_queries)} search queries: {', '.join(search_queries)}")
        )
        
        return state
    
    
    def _execute_single_search(self, query: str) -> List[SearchResult]:
        """Execute a single search query and return processed results.
        
        Args:
            query: Search query string
            
        Returns:
            List of processed search results
        """
        try:
            # Use the tool to execute search
            results_list = self.tavily_search_tool.invoke(query)
            
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
    def _execute_searches(self, state: AgentState) -> AgentState:
        """Execute web searches using the generated queries and merge results into state."""
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
            query_results = self._execute_single_search(query)

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
    
    def _deduplicate_queries(self, new_queries: List[str], previous_queries: List[str]) -> List[str]:
        """Remove queries that are semantically similar to previous ones using embeddings.
        
        Args:
            new_queries: New queries to check
            previous_queries: Previously executed queries
            
        Returns:
            List of deduplicated queries
        """
        similarity_threshold = self.config.query_similarity_threshold
        if not previous_queries or not new_queries:
            return new_queries
        
        try:
            all_queries = previous_queries + new_queries
            embeddings = []
            uncached_queries = []
            uncached_indices = []
            
            # Check cache first
            for i, query in enumerate(all_queries):
                if query in self.embedding_cache:
                    embeddings.append(self.embedding_cache[query])
                else:
                    uncached_queries.append(query)
                    uncached_indices.append(i)
                    embeddings.append(None)  # Placeholder
            
            # Get embeddings for uncached queries
            if uncached_queries:
                new_embeddings = self.embeddings.embed_documents(uncached_queries)
                # Store in cache and fill in placeholders
                for query, embedding, idx in zip(uncached_queries, new_embeddings, uncached_indices):
                    self.embedding_cache[query] = embedding
                    embeddings[idx] = embedding
            
            # Split embeddings
            prev_embeddings = np.array(embeddings[:len(previous_queries)])
            new_embeddings = np.array(embeddings[len(previous_queries):])
            
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
    def _refine_searches(self, state: AgentState) -> AgentState:
        """Analyze search results and generate refined search queries to fill gaps."""
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
            return state
        
        # Check if we've hit the iteration limit BEFORE asking the model
        if refinement_iterations >= self.config.num_refinement_iterations:
            print(f"  Reached max refinement iterations ({self.config.num_refinement_iterations}); proceeding to report generation")
            state["messages"].append(
                AIMessage(content=f"Reached maximum refinement iterations ({self.config.num_refinement_iterations}). Proceeding to report generation.")
            )
            state["search_queries"] = []  # Clear queries to stop loop
            state["should_continue_research"] = False
            return state
        
        # Format search results for analysis
        # Limit the number of results to avoid overwhelming context
        results_to_analyze = search_results[:self.config.max_results_for_report]
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
            structured_llm = self.llm.with_structured_output(RefinementDecision)
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
        can_continue = refinement_iterations < self.config.num_refinement_iterations
        
        # Deduplicate queries against all previously executed queries
        if refined_queries and can_continue:
            # Get all previous queries from state
            all_previous_queries = state.get("all_queries", [])
            original_count = len(refined_queries)
            refined_queries = self._deduplicate_queries(refined_queries, all_previous_queries)
            
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
                self.tavily_search_tool = create_tavily_search_tool(
                    tavily_api_key=self.config.tavily_api_key,
                    max_results=self.config.max_results_per_search,
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
            print(f"  Model suggests more research needed, but reached max iterations ({self.config.num_refinement_iterations}); proceeding to report generation")
            print(f"    Model's reason: {reason}")
            state["messages"].append(
                AIMessage(content=f"Refinement iteration {refinement_iterations}: Model suggested more research ({reason}), but reached maximum refinement iterations ({self.config.num_refinement_iterations}). Proceeding to report generation with current results.")
            )
        else:
            print(f"  Research is comprehensive - proceeding to report generation")
            print(f"    Reason: {reason}")
            state["messages"].append(
                AIMessage(content=f"Refinement iteration {refinement_iterations}: Research is comprehensive. {reason}. Proceeding to report generation.")
            )
        
        return state
    
    def _summarize_result_for_prompt(self, result: SearchResult) -> str:
        """Summarize a search result using LLM for inclusion in the prompt.
        
        Args:
            result: SearchResult model
            
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
            
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            # Simple truncation if LLM call fails
            return result.content[:800] + "..."

    @traceable(name="generate_report")
    def _generate_report(self, state: AgentState) -> AgentState:
        """Generate a comprehensive report based on search results."""
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
        top_results = select_top_results(search_results, limit=self.config.max_results_for_report)
        
        # Store original indices in results for later reference
        for i, result in enumerate(top_results, 1):
            result.original_index = i
        
        # Format results for prompt with LLM summarization
        formatted_results = []
        for i, result in enumerate(top_results, 1):
            summary = self._summarize_result_for_prompt(result)
            formatted_results.append(f"[Source {i}] URL: {result.url}\nTitle: {result.title}\nContent: {summary}\n")
        results_text = "\n\n".join(formatted_results) if formatted_results else "No search results available."
        
        # Format sources section (initial numbering based on max_results_for_report)
        sources_section = format_sources_section(top_results)
        
        # Generate report
        prompt = ReportGenerationPrompt(self.config.max_report_length_words)
        messages = [
            SystemMessage(content=prompt.get_system_prompt()),
            HumanMessage(content=prompt.get_user_prompt(query, results_text, sources_section))
        ]
        
        response = self.llm.invoke(messages)
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
    
    @traceable(name="research")
    def research(self, query: str) -> ResearchResult:
        """High-level entry point: runs the LangGraph and returns query/report/sources for external callers."""
        print(f"\n{'='*80}")
        print(f"🔬 Starting research for: {query}")
        print(f"{'='*80}\n")
        
        # Clear our embedding cache for each research task
        self.embedding_cache.clear()
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "search_queries": [],
            "search_results": [],
            "report": "",
            "refinement_iterations": 0,
            "should_continue_research": False,
            "all_queries": []
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print(f"\n{'='*80}")
        print("✅ Research completed!")
        print(f"{'='*80}\n")
        
        search_results = final_state.get("search_results", [])
        
        return ResearchResult(
            query=query,
            report=final_state["report"],
            messages=final_state["messages"],
            sources=search_results
        )