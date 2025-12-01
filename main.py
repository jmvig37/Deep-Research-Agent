"""Main entry point for the deep research agent."""
import os
import sys
from dotenv import load_dotenv
from config import AgentConfig
from agent import DeepResearchAgent

# Load environment variables
load_dotenv()


def main():
    """Main function to run the research agent."""
    # Load configuration (automatically loads from .env or environment variables)
    config = AgentConfig()
    
    # Validate API keys
    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or as an environment variable.")
        sys.exit(1)
    
    if not config.tavily_api_key:
        print("Error: TAVILY_API_KEY not found in environment variables.")
        print("Please set it in your .env file or as an environment variable.")
        sys.exit(1)
    
    # Initialize agent
    agent = DeepResearchAgent(config)
    print("Initializing Deep Research Agent...")
    
    # Get query from user
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research query: ")
    
    print(f"\nResearching: {query}\n")
    print("=" * 80)
    
    # Execute research
    try:
        results = agent.research(query)
        
        # Display results
        print("\n" + "=" * 80)
        print("RESEARCH REPORT")
        print("=" * 80)
        print(f"\nQuery: {results.query}\n")
        print(results.report)
        
        # Note: Sources are already included in the report above
        # The 'sources' field in results is available for programmatic access if needed
        
    except Exception as e:
        print(f"Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

