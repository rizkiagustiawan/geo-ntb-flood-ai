"""Prometheus metrics for A.E.C.O API.

Exposes metrics at /metrics endpoint for Prometheus scraping.
Tracks request counts, latency, flood predictions, and system health.

Usage:
    from metrics import setup_metrics
    setup_metrics(app)
"""

import time
import logging
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Prometheus-compatible metrics storage
_metrics = {
    "http_requests_total": {},
    "http_request_duration_seconds": [],
    "flood_predictions_total": 0,
    "flood_pixels_total": 0,
    "model_inference_seconds": [],
    "satellite_sync_total": 0,
    "celery_tasks_total": {"success": 0, "failure": 0},
}


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        # Track request count
        path = request.url.path
        method = request.method
        status = response.status_code
        key = f"{method} {path} {status}"
        _metrics["http_requests_total"][key] = _metrics["http_requests_total"].get(key, 0) + 1

        # Track latency
        _metrics["http_request_duration_seconds"].append(duration)

        return response


def record_flood_prediction(n_pixels: int, duration: float):
    """Record a flood prediction event."""
    _metrics["flood_predictions_total"] += 1
    _metrics["flood_pixels_total"] += n_pixels
    _metrics["model_inference_seconds"].append(duration)


def record_satellite_sync():
    """Record a satellite sync event."""
    _metrics["satellite_sync_total"] += 1


def record_celery_task(success: bool):
    """Record a Celery task completion."""
    if success:
        _metrics["celery_tasks_total"]["success"] += 1
    else:
        _metrics["celery_tasks_total"]["failure"] += 1


def generate_metrics() -> str:
    """Generate Prometheus-compatible metrics text."""
    lines = []

    # HTTP requests
    lines.append("# HELP http_requests_total Total HTTP requests")
    lines.append("# TYPE http_requests_total counter")
    for key, count in _metrics["http_requests_total"].items():
        method, path, status = key.split(" ", 2)
        lines.append(f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}')

    # Request duration
    durations = _metrics["http_request_duration_seconds"]
    if durations:
        lines.append("# HELP http_request_duration_seconds Request duration")
        lines.append("# TYPE http_request_duration_seconds summary")
        lines.append(f"http_request_duration_seconds_count {len(durations)}")
        lines.append(f"http_request_duration_seconds_sum {sum(durations):.6f}")

    # Flood predictions
    lines.append("# HELP flood_predictions_total Total flood predictions")
    lines.append("# TYPE flood_predictions_total counter")
    lines.append(f"flood_predictions_total {_metrics['flood_predictions_total']}")

    lines.append("# HELP flood_pixels_total Total flood pixels detected")
    lines.append("# TYPE flood_pixels_total counter")
    lines.append(f"flood_pixels_total {_metrics['flood_pixels_total']}")

    # Model inference
    infer_durations = _metrics["model_inference_seconds"]
    if infer_durations:
        lines.append("# HELP model_inference_seconds Model inference duration")
        lines.append("# TYPE model_inference_seconds summary")
        lines.append(f"model_inference_seconds_count {len(infer_durations)}")
        lines.append(f"model_inference_seconds_sum {sum(infer_durations):.6f}")

    # Satellite sync
    lines.append("# HELP satellite_sync_total Total satellite syncs")
    lines.append("# TYPE satellite_sync_total counter")
    lines.append(f"satellite_sync_total {_metrics['satellite_sync_total']}")

    # Celery tasks
    lines.append("# HELP celery_tasks_total Celery task completions")
    lines.append("# TYPE celery_tasks_total counter")
    lines.append(f'celery_tasks_total{{status="success"}} {_metrics["celery_tasks_total"]["success"]}')
    lines.append(f'celery_tasks_total{{status="failure"}} {_metrics["celery_tasks_total"]["failure"]}')

    return "\n".join(lines) + "\n"


def setup_metrics(app: FastAPI):
    """Add Prometheus metrics endpoint and middleware to FastAPI app."""
    app.add_middleware(MetricsMiddleware)

    @app.get("/metrics", include_in_schema=False,
             summary="Prometheus metrics",
             description="Exposes metrics in Prometheus exposition format")
    async def metrics():
        return Response(
            content=generate_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    logger.info("Prometheus metrics enabled at /metrics")
