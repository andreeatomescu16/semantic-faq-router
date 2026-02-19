from pydantic import BaseModel, ConfigDict, Field


class AskQuestionRequest(BaseModel):
    user_question: str = Field(min_length=1, max_length=2000, description="End-user question text.")


class AskQuestionResponse(BaseModel):
    source: str = Field(description="Response source: local, openai, or compliance.")
    matched_question: str = Field(description="Best local match question or N/A.")
    category: str = Field(description="Matched category or N/A.")
    confidence: float | None = Field(description="Similarity score for local match.")
    answer: str
    trace_id: str

    model_config = ConfigDict(json_schema_extra={"example": {"source": "local"}})
