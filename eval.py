"""Evaluation functions for the deep research agent."""
from typing import Dict, Any, List
from agent import DeepResearchAgent
from config import AgentConfig
from models import ResearchResult


def evaluate_response(
    agent_result: ResearchResult,
    gold_response: str,
    prompt: str
) -> Dict[str, Any]:
    """Evaluate a single agent response against a gold standard.
    
    Args:
        agent_result: The ResearchResult from the agent
        gold_response: The expected/gold standard response
        prompt: The original prompt/query
        
    Returns:
        Dictionary with evaluation metrics
    """
    report = agent_result.report
    sources = agent_result.sources
    
    # Simple metrics
    report_length = len(report)
    num_sources = len(sources)
    
    # Check if report contains citations
    has_citations = "[Source" in report or "Source" in report
    
    # Basic similarity check (simple word overlap)
    report_words = set(report.lower().split())
    gold_words = set(gold_response.lower().split())
    
    if gold_words:
        word_overlap = len(report_words & gold_words) / len(gold_words)
    else:
        word_overlap = 0.0
    
    return {
        "prompt": prompt,
        "report_length": report_length,
        "num_sources": num_sources,
        "has_citations": has_citations,
        "word_overlap": round(word_overlap, 3),
        "report": report[:200] + "..." if len(report) > 200 else report  # Preview
    }


def run_evaluation(
    agent: DeepResearchAgent,
    test_cases: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Run evaluation on multiple test cases.
    
    Args:
        agent: Initialized DeepResearchAgent instance
        test_cases: List of dicts with 'prompt' and 'gold_response' keys
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        prompt = test_case.get("prompt", "")
        gold_response = test_case.get("gold_response", "")
        
        if not prompt:
            print(f"⚠️  Test case {i}: Missing prompt, skipping")
            continue
        
        print(f"\n[{i}/{len(test_cases)}] Evaluating: {prompt[:60]}...")
        
        try:
            # Run agent
            agent_result = agent.research(prompt)
            
            # Evaluate
            eval_result = evaluate_response(agent_result, gold_response, prompt)
            eval_result["test_case_index"] = i
            eval_result["success"] = True
            
            results.append(eval_result)
            
            print(f"  ✓ Report length: {eval_result['report_length']} chars")
            print(f"  ✓ Sources: {eval_result['num_sources']}")
            print(f"  ✓ Citations: {eval_result['has_citations']}")
            print(f"  ✓ Word overlap: {eval_result['word_overlap']:.1%}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                "test_case_index": i,
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
    
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print evaluation summary statistics.
    
    Args:
        results: List of evaluation results
    """
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total test cases: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_length = sum(r.get("report_length", 0) for r in successful) / len(successful)
        avg_sources = sum(r.get("num_sources", 0) for r in successful) / len(successful)
        avg_overlap = sum(r.get("word_overlap", 0) for r in successful) / len(successful)
        citations_rate = sum(r.get("has_citations", False) for r in successful) / len(successful)
        
        print(f"\nAverage report length: {avg_length:.0f} characters")
        print(f"Average sources: {avg_sources:.1f}")
        print(f"Citations present: {citations_rate:.1%}")
        print(f"Average word overlap: {avg_overlap:.1%}")
    
    if failed:
        print(f"\nFailed test cases:")
        for r in failed:
            print(f"  - Test {r.get('test_case_index', '?')}: {r.get('error', 'Unknown error')}")

