"""Evaluation functions for the deep research agent."""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agent import DeepResearchAgent
from config import AgentConfig
from models import ResearchResult
from prompts import JudgePrompt


def evaluate_response(
    agent_result: ResearchResult,
    gold_response: str,
    prompt: str,
    judge_llm: ChatOpenAI = None
) -> Dict[str, Any]:
    """Evaluate a single agent response against a gold standard using LLM judge.
    
    Args:
        agent_result: The ResearchResult from the agent
        gold_response: The expected/gold standard response
        prompt: The original prompt/query
        judge_llm: Optional LLM instance for judging (creates one if not provided)
        
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
    
    # LLM Judge evaluation
    judge_scores = {
        "score": 0,
        "accuracy": 0,
        "completeness": 0,
        "relevance": 0,
        "citation_quality": 0,
        "reasoning": "Evaluation not performed"
    }
    
    if judge_llm is None:
        config = AgentConfig()
        judge_llm = ChatOpenAI(
            model=config.model,
            temperature=0.0,  # Deterministic judging
            api_key=config.openai_api_key
        )
    
    try:
        judge_prompt_class = JudgePrompt()
        messages = [
            SystemMessage(content=judge_prompt_class.get_system_prompt()),
            HumanMessage(content=judge_prompt_class.get_user_prompt(prompt, report, gold_response))
        ]
        
        response = judge_llm.invoke(messages)
        import json
        # Try to extract JSON from response
        content = response.content.strip()
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        judge_scores = json.loads(content)
        
    except Exception as e:
        print(f"  ⚠️  Judge evaluation failed: {str(e)}")
        # Fallback to basic word overlap
        report_words = set(report.lower().split())
        gold_words = set(gold_response.lower().split())
        if gold_words:
            word_overlap = len(report_words & gold_words) / len(gold_words)
            judge_scores["score"] = int(word_overlap * 100)
            judge_scores["reasoning"] = f"Fallback to word overlap: {word_overlap:.1%}"
    
    return {
        "prompt": prompt,
        "report_length": report_length,
        "num_sources": num_sources,
        "has_citations": has_citations,
        "judge_score": judge_scores.get("score", 0),
        "judge_accuracy": judge_scores.get("accuracy", 0),
        "judge_completeness": judge_scores.get("completeness", 0),
        "judge_relevance": judge_scores.get("relevance", 0),
        "judge_citation_quality": judge_scores.get("citation_quality", 0),
        "judge_reasoning": judge_scores.get("reasoning", ""),
        "report": report[:200] + "..." if len(report) > 200 else report  # Preview
    }


def run_evaluation(
    agent: DeepResearchAgent,
    test_cases: List[Dict[str, str]],
    judge_llm: ChatOpenAI = None
) -> List[Dict[str, Any]]:
    """Run evaluation on multiple test cases.
    
    Args:
        agent: Initialized DeepResearchAgent instance
        test_cases: List of dicts with 'prompt' and 'gold_response' keys
        judge_llm: Optional LLM instance for judging (creates one if not provided)
        
    Returns:
        List of evaluation results
    """
    results = []
    
    # Initialize judge LLM if not provided
    if judge_llm is None:
        config = AgentConfig()
        judge_llm = ChatOpenAI(
            model=config.model,
            temperature=0.0,  # Deterministic judging
            api_key=config.openai_api_key
        )
    
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
            
            # Evaluate with LLM judge
            eval_result = evaluate_response(agent_result, gold_response, prompt, judge_llm)
            eval_result["test_case_index"] = i
            eval_result["success"] = True
            
            results.append(eval_result)
            
            print(f"  ✓ Report length: {eval_result['report_length']} chars")
            print(f"  ✓ Sources: {eval_result['num_sources']}")
            print(f"  ✓ Citations: {eval_result['has_citations']}")
            print(f"  ✓ Judge score: {eval_result['judge_score']}/100")
            print(f"    - Accuracy: {eval_result['judge_accuracy']}/100")
            print(f"    - Completeness: {eval_result['judge_completeness']}/100")
            print(f"    - Relevance: {eval_result['judge_relevance']}/100")
            
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
        avg_score = sum(r.get("judge_score", 0) for r in successful) / len(successful)
        avg_accuracy = sum(r.get("judge_accuracy", 0) for r in successful) / len(successful)
        avg_completeness = sum(r.get("judge_completeness", 0) for r in successful) / len(successful)
        avg_relevance = sum(r.get("judge_relevance", 0) for r in successful) / len(successful)
        citations_rate = sum(r.get("has_citations", False) for r in successful) / len(successful)
        
        print(f"\nAverage report length: {avg_length:.0f} characters")
        print(f"Average sources: {avg_sources:.1f}")
        print(f"Citations present: {citations_rate:.1%}")
        print(f"\nJudge Scores (0-100):")
        print(f"  Overall score: {avg_score:.1f}")
        print(f"  Accuracy: {avg_accuracy:.1f}")
        print(f"  Completeness: {avg_completeness:.1f}")
        print(f"  Relevance: {avg_relevance:.1f}")
    
    if failed:
        print(f"\nFailed test cases:")
        for r in failed:
            print(f"  - Test {r.get('test_case_index', '?')}: {r.get('error', 'Unknown error')}")

