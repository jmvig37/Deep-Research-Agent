# Deep Research Agent

A LangGraph-based deep research agent that accepts user queries and generates comprehensive reports grounded in web search results.

## Features

- 🔍 **Web Search Integration**: Uses Tavily Search API to gather information from the web
- 📊 **Structured Reports**: Generates well-organized research reports with citations
- ⚙️ **Configurable**: Customize model, number of searches, report length, and more
- 🚀 **LangGraph**: Built with LangGraph for robust state management and workflow

## Architecture

The agent uses a three-stage workflow:

1. **Generate Search Queries**: Analyzes the user's query and generates diverse search queries
2. **Execute Searches**: Performs web searches using Tavily API
3. **Generate Report**: Synthesizes search results into a comprehensive report

## Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (required)
- Tavily API key (required for search functionality)

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

**⚠️ Security Note**: NEVER hardcode API keys in source code files! Always use environment variables or the `.env` file. The `.env` file is automatically ignored by git (see `.gitignore`).

**LangSmith Observability (Optional):**
- LangSmith provides tracing and monitoring for your agent
- Add your `LANGCHAIN_API_KEY` to `.env` file
- Configure tracing in `config.py` by setting `langchain_tracing_v2=True` when creating `AgentConfig`
- Get your free LangSmith API key at: https://smith.langchain.com/
- Traces will appear in your LangSmith dashboard showing each step of the agent's execution

4. Validate your setup (optional):
```bash
python validate_setup.py
```

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

### Programmatic Usage

```python
from dotenv import load_dotenv
from config import AgentConfig
from agent import DeepResearchAgent

load_dotenv()

# Create configuration
config = AgentConfig(
    model="gpt-4o-mini",
    num_searches=3,
    max_results_per_search=5
)

# Initialize agent
agent = DeepResearchAgent(config)

# Run research
results = agent.research("What are the benefits of renewable energy?")

print(results["report"])
print(f"\nSources: {len(results['sources'])}")
```

### Example Script

See `example.py` for more detailed usage examples:

```bash
python example.py
```

## Configuration

The agent can be configured through the `AgentConfig` class or environment variables:

- `model`: LLM model to use (default: "gpt-4o-mini")
- `temperature`: Model temperature (default: 0.6)
- `num_searches`: Number of search queries to generate (default: 3)
- `max_results_per_search`: Maximum results per search (default: 5)
- `max_report_length`: Approximate maximum report length in words (default: 2000)
- `langchain_tracing_v2`: Enable LangSmith tracing (default: False)
- `langchain_project`: LangSmith project name (default: "deep-research-agent")

Example:

```python
config = AgentConfig(
    model="gpt-4",
    num_searches=5,
    max_results_per_search=10,
    max_report_length=3000,
    langchain_tracing_v2=True,  # Enable LangSmith tracing
    langchain_project="my-research-project"  # Optional: custom project name
)
```

**Note:** LangSmith settings can also be set via environment variables:
- `LANGCHAIN_TRACING_V2=true` (or set `langchain_tracing_v2=True` in config)
- `LANGCHAIN_PROJECT=project-name` (or set `langchain_project` in config)
- `LANGCHAIN_API_KEY` must be in `.env` file

## Project Structure

```
deep-research-agent/
├── agent.py           # Main agent implementation
├── config.py          # Configuration settings
├── main.py            # CLI entry point
├── example.py         # Example usage scripts
├── validate_setup.py  # Setup validation script
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── .env.example       # Environment variables template
└── .gitignore         # Git ignore file
```

## How It Works

1. **State Management**: Uses LangGraph's state management with a `messages` field that tracks all interactions
2. **Search Query Generation**: The LLM generates multiple diverse search queries from the user's question
3. **Web Search**: Executes searches using Tavily Search API
4. **Report Synthesis**: Combines search results into a structured, cited report

## Example Output

```
Researching: What are the latest developments in quantum computing?

================================================================================
RESEARCH REPORT
================================================================================

Query: What are the latest developments in quantum computing?

Executive Summary
Quantum computing has seen significant advances in 2024, with major breakthroughs 
in error correction, qubit stability, and practical applications...

[Detailed report with citations]

================================================================================
SOURCES
================================================================================

1. Latest Quantum Computing Breakthroughs 2024
   URL: https://example.com/quantum-2024

2. Quantum Error Correction Advances
   URL: https://example.com/error-correction
...
```

## Limitations

- Rate limits apply based on your API keys
- Report quality depends on search result quality
- Tavily API has usage limits on free tier

## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/test_prompts.py tests/test_agent.py

# Run only integration tests
pytest tests/test_integration.py

# Run with coverage
pytest --cov=agent --cov=prompts --cov=config
```

### Test Structure

- `tests/test_prompts.py` - Tests for prompt classes
- `tests/test_agent.py` - Unit tests for agent nodes
- `tests/test_integration.py` - Integration tests for full workflows
- `tests/conftest.py` - Shared fixtures and test configuration

### Test Coverage

The test suite covers:
- ✅ Prompt class initialization and output
- ✅ Agent node execution (with mocked LLM)
- ✅ Error handling and edge cases
- ✅ State management and transitions
- ✅ Full workflow integration
- ✅ Refinement loop logic

## Future Improvements

- [ ] Add support for multiple search providers (Exa, SerpAPI)
- [ ] Implement evaluation metrics
- [ ] Support for multi-turn conversations
- [ ] Export reports to various formats (PDF, Markdown)

## License

MIT License

