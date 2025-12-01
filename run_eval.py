"""Evaluation runner script."""
import json
import sys
from pathlib import Path
from agent import DeepResearchAgent
from config import AgentConfig
from eval import run_evaluation, print_summary


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
        print("Usage: python run_eval.py <path_to_test_cases.json>")
        print("\nExample:")
        print("  python run_eval.py test_cases.json")
        sys.exit(1)
    
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

