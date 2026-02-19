import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.errors import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Semantic FAQ Assistant with pgvector retrieval and safe OpenAI fallback.",
)


@app.middleware("http")
async def add_trace_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return await validation_exception_handler(request, exc)


@app.exception_handler(HTTPException)
async def handled_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return await generic_exception_handler(request, exc)


app.include_router(api_router)
