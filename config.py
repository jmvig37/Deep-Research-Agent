from typing import Optional, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class AgentConfig(BaseSettings):
    """Configuration for the deep research agent."""

    # --- Model settings ---
    model: str = "gpt-4o-mini"
    temperature: float = 0.2

    # Embedding for dedupe
    embedding_model: str = "text-embedding-ada-002"
    query_similarity_threshold: float = 0.95

    # --- Search settings ---
    num_searches: int = 5
    max_results_per_search: int = 10
    num_refinement_iterations: int = 3

    # --- Report settings ---
    max_report_length_words: int = 2000  # Maximum approximate word count for the generated report
    # Maximum number of search results to include in report
    max_results_for_report: int = 15 

    # --- API keys ---
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None

    # Optional project name for observability (not required)
    langchain_project: str = "pr-enchanted-butter-18"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Setup LangSmith tracing if the key is provided.
        """
        if self.langchain_api_key and self.langchain_api_key.strip():
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "True"
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project