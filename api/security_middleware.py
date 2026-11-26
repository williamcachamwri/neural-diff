"""HTTP security headers and request payload size limit middleware."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

DEFAULT_MAX_API_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB JSON request ceiling


class SecurityHardeningMiddleware(BaseHTTPMiddleware):
    """Enforces request body size limits and standard security headers on every response."""

    def __init__(self, app, *, max_body_bytes: int = DEFAULT_MAX_API_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "PAYLOAD_TOO_LARGE",
                            "message": f"Request body ({content_length} bytes) exceeds maximum {self.max_body_bytes} bytes.",
                            "http_status": 413,
                            "details": {"max_body_bytes": self.max_body_bytes},
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "INVALID_CONTENT_LENGTH",
                        "message": "Malformed Content-Length header.",
                        "http_status": 400,
                        "details": {},
                    },
                )

        body = await request.body()
        if len(body) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "PAYLOAD_TOO_LARGE",
                    "message": f"Request body ({len(body)} bytes) exceeds maximum {self.max_body_bytes} bytes.",
                    "http_status": 413,
                    "details": {"max_body_bytes": self.max_body_bytes},
                },
            )

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
