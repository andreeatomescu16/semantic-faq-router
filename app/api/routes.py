from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_qa_service, get_token, get_trace_id
from app.schemas.api import AskQuestionRequest, AskQuestionResponse
from app.services.qa_service import QAService

router = APIRouter()


@router.get("/healthz", summary="Liveness probe")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@router.post(
    "/ask-question",
    response_model=AskQuestionResponse,
    summary="Ask FAQ question",
    description="Returns a local FAQ answer, OpenAI fallback, or compliance refusal.",
)
def ask_question(
    payload: AskQuestionRequest,
    _auth: None = Depends(get_token),
    db: Session = Depends(get_db),
    qa_service: QAService = Depends(get_qa_service),
    trace_id: str = Depends(get_trace_id),
) -> AskQuestionResponse:
    result = qa_service.ask(db=db, user_question=payload.user_question, trace_id=trace_id)
    return AskQuestionResponse(
        source=result.source,
        matched_question=result.matched_question,
        category=result.category,
        confidence=result.confidence,
        answer=result.answer,
        trace_id=trace_id,
    )
