"""Security middleware for A.E.C.O API.

Provides:
- API key authentication
- Rate limiting (token bucket)
- Request validation
"""

import os
import time
import logging
from collections import defaultdict
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("aeco-security")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware.

    Reads valid keys from API_KEYS env var (comma-separated).
    Public endpoints (health, tiles, dashboard) are exempt.
    """

    PUBLIC_PATHS = {"/health", "/favicon.ico", "/", "/docs", "/openapi.json"}

    def __init__(self, app):
        super().__init__(app)
        keys_raw = os.environ.get("API_KEYS", "")
        self.valid_keys = set(k.strip() for k in keys_raw.split(",") if k.strip())
        if not self.valid_keys:
            logger.warning("No API_KEYS set — all requests will be allowed (dev mode)")

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for tiles (public map tiles)
        if request.url.path.startswith("/tiles/"):
            return await call_next(request)

        # Skip auth if no keys configured (dev mode)
        if not self.valid_keys:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not api_key or api_key not in self.valid_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter.

    Limits requests per IP address. Config via env:
    - RATE_LIMIT_REQUESTS: max requests per window (default: 100)
    - RATE_LIMIT_WINDOW: window in seconds (default: 60)
    """

    def __init__(self, app):
        super().__init__(app)
        self.max_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
        self.window = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < self.window
        ]

        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded for %s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window}s.",
            )

        self.requests[client_ip].append(now)
        return await call_next(request)
