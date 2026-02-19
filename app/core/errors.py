from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status


def _trace_id_from_request(request: Request) -> str:
    return getattr(request.state, "trace_id", "N/A")


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {"code": "validation_error", "message": "Invalid request payload."},
            "details": exc.errors(),
            "trace_id": _trace_id_from_request(request),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {"code": "internal_error", "message": "Internal server error."},
            "trace_id": _trace_id_from_request(request),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = "auth_error" if exc.status_code == status.HTTP_401_UNAUTHORIZED else "http_error"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": code, "message": str(exc.detail)},
            "trace_id": _trace_id_from_request(request),
        },
    )
