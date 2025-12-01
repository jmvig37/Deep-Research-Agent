# Deep Research Agent

A LangGraph-based deep research agent that accepts user queries via CLI and generates comprehensive research reports grounded in web search results.

## Features

- **Web Search Integration**: Uses Tavily Search API to gather information from the web
- **Structured Reports**: Generates well-organized research reports with citations
- **Configurable**: Customize LLM model, number of searches, report length, and more
- **LangGraph**: Built with LangGraph for robust state management and workflow

## Architecture

The agent uses a four-stage iterative workflow:

1. **Generate Search Queries**: Analyzes the user's query and generates diverse search queries
2. **Execute Searches**: Performs web searches using Tavily API and accumulates results
3. **Refine Searches**: Analyzes results for gaps and generates additional targeted queries if needed (iterates back to step 2 up to a configurable limit)
4. **Generate Report**: Synthesizes all accumulated search results into a comprehensive report with citations

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (required)
- Tavily API key (required)
- LangChain API key (optional, for LangSmith observability and tracing)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deep-research-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
```

Edit `.env` and add your API keys (only API keys should be in .env):
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional
```

**Note:** LangSmith settings (tracing enabled, project name) are configured in `config.py`, not in the `.env` file. Only API keys should be stored in `.env`.

**Getting API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Tavily**: https://tavily.com/ (free tier available)
- **LangSmith**: https://smith.langchain.com/ (free tier available, optional for observability)

## Usage

### Basic Usage

Run the agent with a query:

```bash
python main.py "What are the latest developments in quantum computing?"
```

Or run interactively:

```bash
python main.py
# Then enter your query when prompted
```

## Configuration

The agent can be configured through the `AgentConfig` class or environment variables:

**Model Settings:**
- `model`: LLM model to use (default: "gpt-4o-mini")
- `temperature`: Model temperature (default: 0.2)
- `embedding_model`: Embedding model for query deduplication (default: "text-embedding-ada-002")
- `query_similarity_threshold`: Similarity threshold for query deduplication (default: 0.95)

**Search Settings:**
- `num_searches`: Number of search queries to generate (default: 5)
- `max_results_per_search`: Maximum results per search (default: 10)
- `num_refinement_iterations`: Maximum number of refinement loops (default: 3)

**Report Settings:**
- `max_report_length_words`: Maximum approximate word count for the generated report (default: 2000)
- `max_results_for_report`: Maximum number of search results to include in report (default: 15)

**API Keys (set via environment variables or .env file):**
- `openai_api_key`: OpenAI API key (required)
- `tavily_api_key`: Tavily API key (required)
- `langchain_api_key`: LangChain API key (optional, for LangSmith observability)
- `langchain_project`: LangSmith project name (default: "pr-enchanted-butter-18")

Example:

```python
from config import AgentConfig
from agent import DeepResearchAgent

config = AgentConfig(
    model="gpt-4o-mini",
    temperature=0.2,
    num_searches=5,
    max_results_per_search=10,
    num_refinement_iterations=3,
    max_report_length_words=2000,
    max_results_for_report=15,
    langchain_project="my-research-project"  # Optional: custom LangSmith project name
)

agent = DeepResearchAgent(config)
results = agent.research("Your research question here")
print(results.report)
```

**Note:** 
- API keys should be set in your `.env` file (see Setup section)
- LangSmith tracing is automatically enabled when `LANGCHAIN_API_KEY` is provided in `.env`
- You can override `langchain_project` in config or via `LANGCHAIN_PROJECT` environment variable

## Project Structure

```
deep-research-agent/
├── agent.py           # Main agent implementation with LangGraph workflow
├── config.py          # Configuration settings
├── models.py          # Pydantic models for structured outputs
├── prompts.py         # Prompt templates for each workflow stage
├── util.py            # Utility functions for search processing and report formatting
├── main.py            # CLI entry point
├── validate_setup.py  # Setup validation script
├── requirements.txt   # Python dependencies
├── README.md          
├── .gitignore         
└── tools/
    └── searchtool.py  # Tavily search tool factory using @tool decorator
```

## How It Works

### State Graph Architecture

The agent uses LangGraph's `StateGraph` to orchestrate a four-node workflow with conditional looping. The state is managed through a `TypedDict` (`AgentState`) that persists across all nodes, enabling data accumulation and decision-making.

### Graph Structure

```
[Entry] → generate_search_queries → execute_searches → refine_searches
                                                              ↓
                                                         [Decision]
                                                         ↙        ↘
                                              execute_searches   generate_report → [End]
                                                      ↑                ↓
                                                      └────────────────┘
                                                      (conditional loop)
```

### Node Implementations

#### 1. `generate_search_queries` (Entry Node)
**Purpose**: Analyzes the user's query and generates diverse search queries.

**Key Responsibilities**:
- Takes the user's original query from state
- Uses structured LLM output (`SearchQueries` model) to ensure exactly `num_searches` queries
- Tracks all queries in `state["all_queries"]` for deduplication in later stages
- Enforces query diversity through prompt (no paraphrases, different angles)

**Inputs**: `state["query"]`  
**Outputs**: `state["search_queries"]`, `state["all_queries"]`  
**Error Handling**: Falls back to simple query variations if structured output fails  
**Performance**: Single LLM call with structured output

#### 2. `execute_searches` (Execution Node)
**Purpose**: Executes web searches using Tavily API and accumulates results with deduplication.

**Key Responsibilities**:
- Iterates through `state["search_queries"]` and executes each via Tavily tool
- Processes raw Tavily results into `SearchResult` Pydantic models (validates URL, content, score)
- Deduplicates results by URL to prevent duplicate sources in final report
- Merges new results with existing `state["search_results"]` (important for refinement loops)
- Sorts results by relevance score (Tavily-provided)
- Handles individual search failures gracefully (continues with remaining queries)
- Logs progress and results to `state["messages"]` for observability

**Inputs**: `state["search_queries"]`, `state["search_results"]` (for merging)  
**Outputs**: `state["search_results"]` (accumulated, deduplicated, sorted)  
**Error Handling**: Individual search failures don't stop the workflow; empty results are logged  
**Performance**: Parallel execution possible (currently sequential); rate-limited by Tavily API  
**Deduplication**: URL-based to prevent duplicate sources across refinement iterations

#### 3. `refine_searches` (Analysis & Decision Node)
**Purpose**: Analyzes accumulated search results for information gaps and decides whether additional research is needed.

**Key Responsibilities**:
- Formats top N results (configurable via `max_results_for_report`) for LLM analysis
- Uses structured LLM output (`RefinementDecision` model) to determine:
  - `should_continue`: Whether more research is needed
  - `reason`: Explanation for the decision
  - `time_range`: Optional time filter for refined searches (e.g., "day", "week", "month")
  - `refined_queries`: Up to 3 targeted queries to fill identified gaps
- Implements semantic query deduplication using OpenAI embeddings:
  - Compares new queries against `state["all_queries"]` using cosine similarity
  - Filters out queries above similarity threshold (default: 0.95)
  - Uses embedding cache to minimize API calls
- Enforces iteration limit (`num_refinement_iterations`) to prevent infinite loops
- Updates Tavily tool with new `time_range` if specified (recreates tool instance)
- Logs decision rationale to state for debugging

**Inputs**: `state["search_results"]`, `state["all_queries"]`, `state["refinement_iterations"]`  
**Outputs**: `state["search_queries"]` (refined queries), `state["should_continue_research"]`, `state["refinement_iterations"]`  
**Decision Logic**: 
- Stops if: iteration limit reached, LLM says research is complete, or no refined queries generated
- Continues if: under iteration limit, LLM identifies gaps, and deduplicated queries exist  
**Performance**: Single LLM call + embedding API calls for deduplication (N queries × M previous queries)

#### 4. `generate_report` (Synthesis Node)
**Purpose**: Synthesizes all accumulated search results into a comprehensive, cited research report.

**Key Responsibilities**:
- Selects top N most relevant results (via `select_top_results` utility, default: 15)
- Assigns source indices (1-based) and stores `original_index` for citation mapping
- Summarizes search results using LLM for prompt inclusion:
  - Results ≤ 800 characters are included as-is
  - Longer results are intelligently summarized by LLM (target: 500-800 characters) to preserve key facts while reducing context
  - Falls back to truncation if LLM summarization fails
- Generates report using `ReportGenerationPrompt` with strict citation requirements
- Post-processes report to ensure source accuracy:
  - Extracts cited source numbers from report body using regex patterns
  - Filters sources list to only include actually cited sources
  - Preserves original source numbering to maintain citation integrity
  - Adds or replaces sources section if missing/incorrect
- Handles edge cases: no results (error report), no citations found (fallback to all sources)

**Inputs**: `state["search_results"]`, `state["query"]`  
**Outputs**: `state["report"]`  
**Error Handling**: Generates error report if no search results available  
**Performance**: Single LLM call; post-processing is O(N) where N = number of sources  
**Quality Assurance**: Regex-based citation extraction ensures source list matches report citations

### Conditional Edge: `_should_continue_refining`

**Purpose**: Determines workflow direction after `refine_searches` node.

**Decision Logic**:
```python
if (refinement_iterations < max_iterations AND 
    should_continue_research AND 
    search_queries exist):
    return "continue"  # Loop back to execute_searches
else:
    return "finish"    # Proceed to generate_report
```

**Rationale**: Prevents infinite loops while allowing iterative refinement when gaps are identified.

### State Management

The `AgentState` TypedDict maintains:
- **Immutable tracking**: `messages` (append-only log), `all_queries` (append-only)
- **Mutable accumulation**: `search_results` (merged across iterations)
- **Control flow**: `refinement_iterations`, `should_continue_research`, `search_queries`
- **Final output**: `report`, `query`

All nodes read from and write to shared state, enabling data persistence across the workflow without external storage.

### Error Resilience

- **Node-level**: Each node handles its own exceptions and continues workflow
- **Fallback strategies**: Query generation → simple variations; Report generation → error message
- **Graceful degradation**: Missing results don't crash the agent; empty states are handled
- **Observability**: All errors logged to `state["messages"]` for debugging via LangSmith

## Evaluation

The project includes an evaluation system using an LLM judge to assess report quality against gold standard responses.

### Running Evaluations

From the project root:

```bash
python eval/run_eval.py eval/test_cases.json
```

Or from the eval directory:

```bash
cd eval
python run_eval.py test_cases.json
```

### Evaluation Metrics

The LLM judge evaluates each report on:

- **Overall Score** (0-100): Composite quality score
- **Accuracy** (0-100): How well the information aligns with the gold standard
- **Completeness** (0-100): Coverage of key points from the gold standard
- **Relevance** (0-100): Direct relevance to the original query
- **Citation Quality** (0-100): Proper source citation and relevance

Additional metrics tracked:
- Report length
- Number of sources
- Citation presence

### Output

The evaluation runner:
- Prints progress for each test case
- Displays summary statistics
- Saves detailed results to `{test_cases_filename}_results.json`

Results include judge scores, reasoning, and full report previews for analysis.

## Future Improvements

### UX
- [ ] Add support for multiple search providers (Exa, SerpAPI)
- [ ] Support for multi-turn conversations
- [ ] Export reports to various formats (PDF, Markdown)
- [ ] Re-number citations per report so the ## Sources section uses sequential numbering within that report, instead of global indices.
- [ ] Accept a JSON structure for the final report format in the config.
- [ ] Implement a store of reports/RAG lookup across prior reports
- [ ] Implement additional tools (e.g., charting/analytics helpers) that post-process search results into quantitative summaries or visuals (tables, trend charts, comparative stats)
- [ ] Stream the report as it is generated at the final node, removing latent, unresponsive time
 
### Agent Design
- [ ] Implement a planner node that outlines the research plan:
    - [ ] Parallelize sections where possible
    - [ ] Dynamically control the refine loop (start/stop based on coverage instead of a fixed iteration count)
- [ ] Implement an optional query-clarification node to better target research (conditionally routes into the planner node when ambiguity is detected)
- [ ] Deduplicate results by content (via embeddings or lightweight LLM summarization) in the execute step, instead of URL-only dedup, to catch near-duplicate articles from different domains (increase of token cost/latency, traded off for lax URL equivalency).
- [ ] Implement a semantic source-quality filtering node on top of Tavily scores, down-weighting low-quality or social-media sources so they don’t displace reputable articles (currently mitigated via prompting, but non-scholarly topics still surface Facebook, Reddit, TikTok, etc.).
- [ ] Implement a guardrail node to filter or rewrite unsafe search queries and down-rank/flag low-trust or unsafe web sources before they reach the report step.
### Performance & Cost
- [ ] Cache per-query search results and embeddings across runs (disk or external store) to avoid recomputation.
- [ ] Cache per-source LLM summaries instead of recomputing them on every report generation.
- [ ] Add parallelization for web searches (e.g., async Tavily calls) with rate-limit controls.
- [ ] Batch LLM summarization calls to reduce latency and token usage.
      Right now each long search result triggers an individual summarization call. 
      Grouping results into a single structured LLM call per batch (or even a single call) avoids repeated 
      model overhead, cuts token startup costs, and keeps prompt construction cheaper.
### Testing & Observability
- [ ] Increase end-to-end eval coverage, including more variable prompt difficulties, formatting edge cases, and topic diversity.
- [ ] Implement evals for each node of the graph (e.g., search query generation, refinement decisions, report quality) to catch regressions in isolation.
- [ ] Integrate evals with LangSmith for better visibility into agent performance over time (dashboards, comparison runs, and regression alerts).
## License
MIT License

