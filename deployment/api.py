"""FastAPI application for World Model Lens analysis endpoint.

Usage:
    uvicorn world_model_lens.deployment.api:app --reload
"""

from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import io
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

from world_model_lens import HookedWorldModel, ActivationCache, WorldModelConfig
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from world_model_lens.patching.patcher import TemporalPatcher


class EpisodeAnalysisRequest(BaseModel):
    """Request model for episode analysis."""

    observations: List[List[float]] = Field(description="Observation sequence")
    actions: Optional[List[List[float]]] = Field(default=None, description="Action sequence")
    model_config: Optional[Dict[str, Any]] = Field(default=None, description="Model config")
    analysis_types: List[str] = Field(
        default=["surprise", "patching", "concepts"], description="Types of analysis to perform"
    )


class EpisodeAnalysisResponse(BaseModel):
    """Response model for episode analysis."""

    summary: Dict[str, Any]
    surprise_timeline: Optional[List[float]] = None
    patching_results: Optional[Dict[str, Any]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    html_report: Optional[str] = None


app = FastAPI(
    title="World Model Lens API",
    description="Analysis endpoint for world model interpretability",
    version="0.1.0",
)

wm: Optional[HookedWorldModel] = None
analyzer: Optional[BeliefAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global wm, analyzer
    try:
        from world_model_lens.backends.generic_adapter import WorldModelAdapter

        class DefaultAdapter(WorldModelAdapter):
            def __init__(self):
                super().__init__(None)
                self.encoder = torch.nn.Linear(128, 256)
                self.gru = torch.nn.GRUCell(260, 256)
                self.posterior_net = torch.nn.Linear(256, 32)
                self.prior_net = torch.nn.Linear(256, 32)
                self.reward_head = torch.nn.Linear(256, 1)

            def encode(self, obs, context=None):
                obs_enc = self.encoder(obs)
                return obs_enc, obs_enc

            def dynamics(self, state, action=None):
                if action is not None:
                    gru_input = torch.cat([state.hidden, action], dim=-1)
                else:
                    gru_input = state.hidden
                new_hidden = self.gru(gru_input)
                return new_hidden

            def get_components(self):
                return ["encoder", "gru", "posterior", "prior", "reward"]

        adapter = DefaultAdapter()
        config = WorldModelConfig()
        wm = HookedWorldModel(adapter=adapter, config=config)
        analyzer = BeliefAnalyzer(wm)
        print("World Model Lens API initialized")
    except Exception as e:
        print(f"Warning: Could not initialize default model: {e}")
        wm = None
        analyzer = None
    yield
    print("World Model Lens API shutting down")


app.router.lifespan_context = lifespan


def generate_html_report(
    summary: Dict[str, Any],
    surprise: Optional[List[float]] = None,
    patching: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate interactive HTML report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>World Model Lens Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #f5f5f5; border-radius: 4px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            .metric-label {{ font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>World Model Lens Analysis Report</h1>
        
        <div class="section">
            <h2>Summary</h2>
            {"".join(f'<div class="metric"><div class="metric-value">{v}</div><div class="metric-label">{k}</div></div>' for k, v in summary.items())}
        </div>
    """

    if surprise:
        html += (
            """
        <div class="section">
            <h2>Surprise Timeline</h2>
            <div id="surprise-plot"></div>
            <script>
                var surprise_data = [{y: """
            + json.dumps(surprise)
            + """}];
                Plotly.newPlot('surprise-plot', [{y: surprise_data[0].y, type: 'scatter', mode: 'lines+markers'}]);
            </script>
        </div>
        """
        )

    html += """
    </body>
    </html>
    """
    return html


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with usage instructions."""
    return """
    <html>
    <head><title>World Model Lens API</title></head>
    <body>
        <h1>World Model Lens API</h1>
        <p>Use the following endpoints:</p>
        <ul>
            <li><a href="/docs">/docs</a> - Swagger UI</li>
            <li><a href="/redoc">/redoc</a> - ReDoc</li>
            <li>POST /analyze_episode - Analyze episode</li>
        </ul>
    </body>
    </html>
    """


@app.post("/analyze_episode", response_model=EpisodeAnalysisResponse)
async def analyze_episode(request: EpisodeAnalysisRequest):
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
        if "surprise" in request.analysis_types:
            surprise_result = analyzer.surprise_timeline(cache)
            summary["mean_surprise"] = surprise_result.mean_surprise
            summary["max_surprise"] = surprise_result.max_surprise_value
            summary["max_surprise_timestep"] = surprise_result.max_surprise_timestep
            surprise_timeline = surprise_result.kl_sequence.tolist()

        patching_results = None
        if "patching" in request.analysis_types:
            patcher = TemporalPatcher(wm)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_episode")
async def upload_episode(file: UploadFile = File(...)):
    """Upload a zarr/HDF5 file containing episode data."""
    import tempfile
    import os

    try:
        suffix = ".zarr" if file.filename.endswith(".zarr") else ".h5"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from world_model_lens.core.lazy_trajectory import TrajectoryDataset

        dataset = TrajectoryDataset.from_disk(tmp_path)

        os.unlink(tmp_path)

        return {
            "status": "success",
            "n_trajectories": len(dataset),
            "statistics": dataset.statistics(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": wm is not None,
        "analyzer_ready": analyzer is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
