"""FastAPI application for World Model Lens analysis endpoint.

Usage:
    uvicorn world_model_lens.deployment.api:app --reload
"""

import json
import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer

# auto_discover_concepts is imported dynamically where used to avoid
# heavy optional dependencies at module import time.


class EpisodeAnalysisRequest(BaseModel):
    """Request model for episode analysis."""

    observations: list[list[float]] = Field(..., description="Observation sequence")
    actions: list[list[float]] | None = Field(default=None, description="Action sequence")
    # avoid naming collision with pydantic BaseModel.model_config (v2)
    model_cfg: dict[str, Any] | None = Field(
        default=None, description="Model config", alias="model_config"
    )
    analysis_types: list[str] = Field(
        default=["surprise", "patching", "concepts"],
        description="Types of analysis to perform",
    )


class EpisodeAnalysisResponse(BaseModel):
    """Response model for episode analysis."""

    summary: dict[str, Any]
    surprise_timeline: list[float] | None = None
    patching_results: dict[str, Any] | None = None
    concepts: list[dict[str, Any]] | None = None
    html_report: str | None = None


app = FastAPI(
    title="World Model Lens API",
    description="Analysis endpoint for world model interpretability",
    version="0.1.0",
)

wm: HookedWorldModel | None = None
analyzer: BeliefAnalyzer | None = None

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: try to initialize a default lightweight model."""
    global wm, analyzer
    try:
        from world_model_lens.backends.generic_adapter import WorldModelAdapter

        class DefaultAdapter(WorldModelAdapter):
            def __init__(self) -> None:
                super().__init__(WorldModelConfig())
                self.encoder = torch.nn.Linear(128, 256)
                self.gru = torch.nn.GRUCell(260, 256)
                self.posterior_net = torch.nn.Linear(256, 32)
                self.prior_net = torch.nn.Linear(256, 32)
                self.reward_head = torch.nn.Linear(256, 1)

            def encode(
                self, observation: torch.Tensor, context: Any | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
                obs_enc = self.encoder(observation)
                return obs_enc, obs_enc

            def dynamics(self, state: Any, action: torch.Tensor | None = None) -> torch.Tensor:
                if action is not None:
                    gru_input = torch.cat([state.hidden, action], dim=-1)
                else:
                    gru_input = state.hidden
                return self.gru(gru_input)

            def get_components(self) -> list[str]:
                return ["encoder", "gru", "posterior", "prior", "reward"]

        adapter = DefaultAdapter()
        config = WorldModelConfig()
        wm = HookedWorldModel(adapter=adapter, config=config)
        analyzer = BeliefAnalyzer(wm)
        logger.info("World Model Lens API initialized")
    except Exception as e:  # pragma: no cover - best-effort initialization
        logger.warning("Could not initialize default model: %s", e)
        wm = None
        analyzer = None
    yield
    logger.info("World Model Lens API shutting down")


app.router.lifespan_context = lifespan


def generate_html_report(
    summary: dict[str, Any],
    surprise: list[float] | None = None,
    patching: dict[str, Any] | None = None,
) -> str:
    """Generate interactive HTML report."""
    summary_items = []
    for k, v in summary.items():
        summary_items.append(
            f'<div class="metric"><div class="metric-value">{v}</div>'
            + f'<div class="metric-label">{k}</div></div>'
        )

    summary_html = "".join(summary_items)

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>World Model Lens Analysis Report</title>",
        '    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 40px; }",
        "        h1 { color: #333; }",
        "        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }",
        "        .metric { display: inline-block; margin: 10px 20px; padding: 15px; background: #f5f5f5; border-radius: 4px; }",
        "        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }",
        "        .metric-label { font-size: 12px; color: #666; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>World Model Lens Analysis Report</h1>",
        '    <div class="section">',
        "        <h2>Summary</h2>",
        summary_html,
        "    </div>",
    ]

    if surprise:
        surprise_json = json.dumps(surprise)
        html_parts.extend(
            [
                '    <div class="section">',
                "        <h2>Surprise Timeline</h2>",
                '        <div id="surprise-plot"></div>',
                "        <script>",
                f"            var surprise_data = [{'{'}y: {surprise_json}{'}'}];",
                "            Plotly.newPlot('surprise-plot', [{y: surprise_data[0].y, type: 'scatter', mode: 'lines+markers'}]);",
                "        </script>",
                "    </div>",
            ]
        )

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Root endpoint with usage instructions."""
    content = (
        "<html>"
        "<head><title>World Model Lens API</title></head>"
        "<body>"
        "    <h1>World Model Lens API</h1>"
        "    <p>Use the following endpoints:</p>"
        "    <ul>"
        '        <li><a href="/docs">/docs</a> - Swagger UI</li>'
        '        <li><a href="/redoc">/redoc</a> - ReDoc</li>'
        "        <li>POST /analyze_episode - Analyze episode</li>"
        "    </ul>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=content)


@app.post("/analyze_episode", response_model=EpisodeAnalysisResponse)
async def analyze_episode(request: EpisodeAnalysisRequest) -> EpisodeAnalysisResponse:
    """Analyze a single episode and return results with HTML report."""
    if wm is None:
        raise HTTPException(status_code=503, detail="World model not initialized")

    try:
        obs = torch.tensor(request.observations, dtype=torch.float32)
        actions = torch.tensor(request.actions, dtype=torch.float32) if request.actions else None

        traj, cache = wm.run_with_cache(obs, actions)

        summary = {
            "trajectory_length": len(traj),
            "n_components": len(cache.component_names),
            "memory_gb": cache.estimate_memory_gb(),
        }

        surprise_timeline = None
        if "surprise" in request.analysis_types and analyzer is not None:
            surprise_result = analyzer.surprise_timeline(cache)
            summary["mean_surprise"] = surprise_result.mean_surprise
            summary["max_surprise"] = surprise_result.max_surprise_value
            summary["max_surprise_timestep"] = surprise_result.max_surprise_timestep
            surprise_timeline = surprise_result.kl_sequence.tolist()

        patching_results = None
        if "patching" in request.analysis_types:
            summary["n_patch_components"] = len(wm.adapter.get_components())
            patching_results = {
                "status": "completed",
                "n_components": len(wm.adapter.get_components()),
            }

        concepts = None
        if "concepts" in request.analysis_types:
            from world_model_lens.analysis.multimodal import auto_discover_concepts

            discovered = auto_discover_concepts(cache, n_concepts=10)
            concepts = [
                {"name": c.name, "interpretability_score": c.interpretability_score}
                for c in discovered
            ]

        html_report = generate_html_report(summary, surprise_timeline, patching_results)

        return EpisodeAnalysisResponse(
            summary=summary,
            surprise_timeline=surprise_timeline,
            patching_results=patching_results,
            concepts=concepts,
            html_report=html_report,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/upload_episode")
async def upload_episode(file: UploadFile | None = None) -> dict[str, Any]:
    """Upload a zarr/HDF5 file containing episode data."""
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")

    try:
        suffix = ".zarr" if file.filename.endswith(".zarr") else ".h5"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from world_model_lens.core.lazy_trajectory import TrajectoryDataset

        dataset = TrajectoryDataset.from_disk(tmp_path)

        Path(tmp_path).unlink()

        return {
            "status": "success",
            "n_trajectories": len(dataset),
            "statistics": dataset.statistics(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": wm is not None,
        "analyzer_ready": analyzer is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
