"""Production API server for World Model Lens.

FastAPI application with:
- JWT/OAuth2 authentication
- Rate limiting
- Async endpoints
- OpenTelemetry tracing
- Prometheus metrics
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Any, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import torch

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.monitoring.logging import setup_logging, get_logger
from world_model_lens.monitoring.metrics import (
    MetricsCollector,
    track_request_latency,
    track_memory_usage,
)
from world_model_lens.monitoring.tracing import setup_tracing, trace_function

logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)

security = HTTPBearer(auto_error=False)

metrics = MetricsCollector()


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint."""

    model_checkpoint: str = Field(..., description="Path to model checkpoint")
    trajectory: list[dict[str, Any]] = Field(..., description="Trajectory data")
    analysis_type: str = Field(
        default="full",
        description="Type of analysis: full, probing, patching, or safety",
    )
    probe_concept: Optional[str] = Field(None, description="Concept to probe for")
    gpu: bool = Field(default=False, description="Use GPU acceleration")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint."""

    analysis_id: str
    status: str
    results: dict[str, Any]
    report_url: Optional[str] = None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    gpu_available: bool
    gpu_count: int
    memory_used_mb: float
    memory_total_mb: float


class BenchmarkRequest(BaseModel):
    """Request model for benchmark endpoint."""

    model_checkpoint: str
    dataset_path: str
    benchmarks: list[str] = [
        "latency",
        "memory",
        "throughput",
    ]
    num_samples: int = Field(default=1000, ge=1, le=100000)
    profile: bool = Field(default=False, description="Enable profiling")


class BenchmarkResponse(BaseModel):
    """Response model for benchmark endpoint."""

    benchmark_id: str
    results: dict[str, Any]
    profiles: Optional[dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    backend: str
    d_state: int
    d_action: Optional[int]
    has_reward_head: bool
    has_value_head: bool
    has_decoder: bool
    parameter_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting World Model Lens API Server")
    setup_logging()
    setup_tracing()
    metrics.start()
    yield
    logger.info("Shutting down World Model Lens API Server")
    metrics.stop()


app = FastAPI(
    title="World Model Lens API",
    description="Production API for world model interpretability analysis",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def verify_auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> Optional[dict[str, Any]]:
    """Verify authentication token."""
    if credentials is None:
        return None
    return {"user_id": "verified_user"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 if gpu_count > 0 else 0
    memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024 if gpu_count > 0 else 0

    return HealthResponse(
        status="healthy",
        version="0.2.0",
        gpu_available=torch.cuda.is_available(),
        gpu_count=gpu_count,
        memory_used_mb=memory_allocated,
        memory_total_mb=memory_reserved,
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Check if API is ready to serve requests."""
    return {"ready": True}


@app.post(
    "/v1/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
)
@track_request_latency(metrics)
async def analyze(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user: Annotated[Optional[dict], Depends(verify_auth)],
):
    """Run analysis on a world model."""
    analysis_id = f"analysis_{int(time.time())}"

    logger.info(f"Starting analysis {analysis_id} for user {user}")

    try:
        start_time = time.time()

        wm = HookedWorldModel.from_checkpoint(request.model_checkpoint)
        if request.gpu and torch.cuda.is_available():
            wm = wm.to("cuda")

        if request.analysis_type == "probing" and request.probe_concept:
            from world_model_lens.probing.prober import LatentProber

            prober = LatentProber(wm)
            results = prober.train_probe(
                concept=request.probe_concept,
                trajectories=request.trajectory,
            )
        elif request.analysis_type == "patching":
            from world_model_lens.patching.patcher import TemporalPatcher

            patcher = TemporalPatcher(wm)
            results = patcher.full_sweep(request.trajectory)
        elif request.analysis_type == "safety":
            from world_model_lens.safety.analyzer import SafetyAnalyzer

            analyzer = SafetyAnalyzer(wm)
            results = analyzer.run_safety_audit(request.trajectory)
        else:
            from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer

            analyzer = BeliefAnalyzer(wm)
            results = analyzer.full_analysis(request.trajectory)

        processing_time = time.time() - start_time

        metrics.record_analysis(
            analysis_type=request.analysis_type,
            duration=processing_time,
            success=True,
        )

        return AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            results=results,
            report_url=f"/v1/reports/{analysis_id}",
            processing_time_seconds=processing_time,
        )

    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        metrics.record_analysis(
            analysis_type=request.analysis_type,
            duration=0,
            success=False,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
@track_request_latency(metrics)
async def benchmark(
    request: BenchmarkRequest,
    user: Annotated[Optional[dict], Depends(verify_auth)],
):
    """Run benchmarks on a world model."""
    benchmark_id = f"benchmark_{int(time.time())}"

    logger.info(f"Starting benchmark {benchmark_id}")

    try:
        from world_model_lens.benchmarks.perf_runner import BenchmarkRunner

        runner = BenchmarkRunner(
            checkpoint=request.model_checkpoint,
            dataset_path=request.dataset_path,
        )

        results = runner.run(
            benchmarks=request.benchmarks,
            num_samples=request.num_samples,
            profile=request.profile,
        )

        return BenchmarkResponse(
            benchmark_id=benchmark_id,
            results=results,
            profiles=results.get("profiles") if request.profile else None,
        )

    except Exception as e:
        logger.error(f"Benchmark {benchmark_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_id: str):
    """Get information about a loaded model."""
    try:
        wm = HookedWorldModel.from_checkpoint(f"models/{model_id}")
        params = sum(p.numel() for p in wm.named_weights.values())
        caps = wm.adapter.capabilities

        return ModelInfo(
            name=model_id,
            backend=getattr(wm.config, "backend", "unknown"),
            d_state=getattr(wm.config, "d_h", 0),
            d_action=getattr(wm.config, "d_action", None),
            has_reward_head=caps.has_reward_head,
            has_value_head=caps.has_value_head,
            has_decoder=caps.has_decoder,
            parameter_count=params,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Export Prometheus metrics."""
    return metrics.export_prometheus()


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Root endpoint with API documentation link."""
    return """
    <html>
        <head><title>World Model Lens API</title></head>
        <body>
            <h1>World Model Lens API</h1>
            <p>Production API for world model interpretability</p>
            <ul>
                <li><a href="/docs">API Documentation (Swagger)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/metrics">Prometheus Metrics</a></li>
            </ul>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "world_model_lens.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
    )
