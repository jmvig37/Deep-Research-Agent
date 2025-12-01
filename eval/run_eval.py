"""Evaluation runner script."""
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import agent modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables from .env file in project root
env_file = parent_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
    
from agent import DeepResearchAgent
from config import AgentConfig

# Import eval functions - use direct file import to avoid package issues
import importlib.util
eval_file = Path(__file__).parent / "eval.py"
spec = importlib.util.spec_from_file_location("eval_module", eval_file)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

run_evaluation = eval_module.run_evaluation
print_summary = eval_module.print_summary


def load_test_cases(json_path: str) -> list:
    """Load test cases from JSON file.
    
    Expected JSON structure:
    {
        "test_cases": [
            {
                "prompt": "user query here",
                "gold_response": "expected response text"
            },
            ...
        ]
    }
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of test case dictionaries
    """
    path = Path(json_path)
    
    # If relative path, try relative to script directory first, then current directory
    if not path.is_absolute():
        script_dir = Path(__file__).parent
        if not path.exists():
            path = script_dir / json_path
    
    if not path.exists():
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = data.get("test_cases", [])
        if not test_cases:
            print("⚠️  Warning: No test cases found in JSON file")
        
        return test_cases
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)


def main():
    """Main evaluation runner."""
    if len(sys.argv) < 2:
        # Try to find example file
        script_dir = Path(__file__).parent
        example_file = script_dir / "test_cases.example.json"
        if example_file.exists():
            json_path = str(example_file)
            print(f"⚠️  No test cases file specified, using example: {json_path}")
            print("   Create your own test_cases.json file to use custom test cases.\n")
        else:
            print("Usage: python run_eval.py <path_to_test_cases.json>")
            print("\nExample:")
            print("  python run_eval.py test_cases.json")
            print("  python run_eval.py eval/test_cases.example.json")
            sys.exit(1)
    else:
        json_path = sys.argv[1]
    
    print("="*60)
    print("DEEP RESEARCH AGENT EVALUATION")
    print("="*60)
    
    # Load test cases
    print(f"\n📂 Loading test cases from: {json_path}")
    test_cases = load_test_cases(json_path)
    print(f"✓ Loaded {len(test_cases)} test cases")
    
    # Initialize agent
    print("\n🤖 Initializing agent...")
    config = AgentConfig()
    
    # Validate API keys before initializing agent
    if not config.openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found.")
        print("   Please set it in your .env file or as an environment variable.")
        print(f"   Expected .env file at: {parent_dir / '.env'}")
        sys.exit(1)
    
    if not config.tavily_api_key:
        print("❌ Error: TAVILY_API_KEY not found.")
        print("   Please set it in your .env file or as an environment variable.")
        sys.exit(1)
    
    agent = DeepResearchAgent(config)
    print("✓ Agent initialized")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation on {len(test_cases)} test cases...")
    results = run_evaluation(agent, test_cases)
    
    # Print summary
    print_summary(results)
    
    # Optionally save results
    output_path = Path(json_path).stem + "_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()

