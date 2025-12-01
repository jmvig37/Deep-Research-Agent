from typing import TypedDict, Annotated, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from operator import add

from models import ResearchResult, SearchResult
from config import AgentConfig
from tools.searchtool import create_tavily_search_tool
from nodes import (
    generate_search_queries,
    execute_searches,
    refine_searches,
    generate_report,
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
    search_results: List[SearchResult]   # accumulated + deduped results

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
    
    def _generate_search_queries(self, state: AgentState) -> AgentState:
        """Generate search queries based on the user's question using structured outputs."""
        return generate_search_queries(state, self.llm, self.config)
    
    def _execute_searches(self, state: AgentState) -> AgentState:
        """Execute web searches using the generated queries and merge results into state."""
        return execute_searches(state, self.tavily_search_tool)
    
    def _refine_searches(self, state: AgentState) -> AgentState:
        """Analyze search results and generate refined search queries to fill gaps."""
        state, new_tool = refine_searches(
            state,
            self.llm,
            self.config,
            self.embeddings,
            self.embedding_cache,
        )
        # Update tool if a new one was created
        if new_tool is not None:
            self.tavily_search_tool = new_tool
        return state
    
    def _generate_report(self, state: AgentState) -> AgentState:
        """Generate a comprehensive report based on search results."""
        return generate_report(state, self.llm, self.config)
    
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