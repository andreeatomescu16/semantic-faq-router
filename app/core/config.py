from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Semantic FAQ Assistant"
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/semantic_faq",
        alias="DATABASE_URL",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    api_auth_token: str | None = Field(default=None, alias="API_AUTH_TOKEN")
    require_auth: bool = True

    similarity_threshold: float = Field(default=0.82, alias="SIMILARITY_THRESHOLD")
    similarity_margin: float = Field(default=0.05, alias="SIMILARITY_MARGIN")
    top_k: int = Field(default=5, alias="TOP_K")
    scoring_strategy: str = Field(default="cosine_threshold", alias="SCORING_STRATEGY")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")
    hybrid_beta: float = Field(default=0.2, alias="HYBRID_BETA")
    hybrid_gamma: float = Field(default=0.1, alias="HYBRID_GAMMA")
    cost_fp_local: float = Field(default=8.0, alias="COST_FP_LOCAL")
    cost_fn_local: float = Field(default=2.0, alias="COST_FN_LOCAL")
    cost_openai_call: float = Field(default=0.3, alias="COST_OPENAI_CALL")
    cost_false_compliance: float = Field(default=3.0, alias="COST_FALSE_COMPLIANCE")
    routing_objective: str = Field(default="cost", alias="ROUTING_OBJECTIVE")
    judge_enabled: bool = Field(default=True, alias="JUDGE_ENABLED")
    judge_model: str = Field(default="gpt-5-mini", alias="JUDGE_MODEL")
    judge_max_cases: int = Field(default=50, alias="JUDGE_MAX_CASES")
    judge_gray_low: float = Field(default=0.55, alias="JUDGE_GRAY_LOW")
    judge_gray_high: float = Field(default=0.70, alias="JUDGE_GRAY_HIGH")

    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_timeout_seconds: int = Field(default=15, alias="OPENAI_TIMEOUT_SECONDS")

    embedding_version: str = Field(default="v1", alias="EMBEDDING_VERSION")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    domain_classifier_enabled: bool = Field(
        default=False,
        alias="DOMAIN_CLASSIFIER_ENABLED",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
