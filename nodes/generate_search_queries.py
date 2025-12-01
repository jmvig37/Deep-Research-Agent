"""Node for generating initial search queries."""

from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable

from models import SearchQueries
from prompts import SearchQueryGenerationPrompt


@traceable(name="generate_search_queries")
def generate_search_queries(
    state: TypedDict,
    llm,
    config,
) -> TypedDict:
    """Generate search queries based on the user's question using structured outputs.
    
    Args:
        state: Current agent state
        llm: Language model instance
        config: Agent configuration
        
    Returns:
        Updated agent state
    """
    query = state["query"]
    
    print("🔍 Generating search queries...")
    
    prompt = SearchQueryGenerationPrompt(config.num_searches)
    system_prompt = prompt.get_system_prompt()
    user_prompt = prompt.get_user_prompt(query)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        # Use structured output
        structured_llm = llm.with_structured_output(SearchQueries)
        result = structured_llm.invoke(messages)
        search_queries = result.queries
                    
    except Exception as e:
        # Fallback: use the original query with variations
        search_queries = [f"{query} aspect {i+1}" for i in range(config.num_searches)]
        state["messages"].append(
            AIMessage(content=f"Error generating search queries, using fallback: {str(e)}")
        )
    
    # Store queries in state
    search_queries = search_queries[:config.num_searches]
    state["search_queries"] = search_queries
    
    # Track all queries executed so far for deduplication
    state["all_queries"].extend(search_queries)
    
    print(f"✓ Generated {len(search_queries)} search queries")
    
    # Log query generation output in messages
    state["messages"].append(
        AIMessage(content=f"Generated {len(search_queries)} search queries: {', '.join(search_queries)}")
    )
    
    return state

