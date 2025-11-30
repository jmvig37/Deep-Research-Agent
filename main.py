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
    
    tavily_key = config.tavily_api_key
    if not tavily_key or not tavily_key.strip():
        print("=" * 80)
        print("⚠️  TAVILY_API_KEY is missing!")
        print("=" * 80)
        print("The search functionality requires a Tavily API key to work.")
        print("\nTo get a free API key:")
        print("1. Visit: https://tavily.com/")
        print("2. Sign up for a free account")
        print("3. Get your API key from the dashboard")
        print("4. Add it to your .env file: TAVILY_API_KEY=your_actual_key_here")
        print("\nWithout a valid API key, searches will fail with 401 Unauthorized errors.")
        print("=" * 80)
        print()
    
    # Initialize agent
    print("Initializing Deep Research Agent...")
    agent = DeepResearchAgent(config)
    
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
        print(f"\nQuery: {results['query']}\n")
        print(results['report'])
        
        # Note: Sources are already included in the report above
        # The 'sources' field in results is available for programmatic access if needed
        
    except Exception as e:
        print(f"Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

