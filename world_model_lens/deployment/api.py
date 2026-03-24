"""FastAPI server for WorldModelLens analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import torch

from world_model_lens import HookedWorldModel, ActivationCache, WorldModelConfig
from world_model_lens.core.lazy_trajectory import LatentTrajectoryLite, TensorStore
from world_model_lens.probing import LatentProber
from world_model_lens.patching import TemporalPatcher


app = FastAPI(
    title="WorldModelLens API",
    description="Interpretability toolkit for world models",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    observations: List[List[float]]
    actions: Optional[List[List[float]]] = None
    component: str = "h"
    timestep: Optional[int] = None


class ProbingRequest(BaseModel):
    component: str
    labels: List[float]
    concept_name: str
    probe_type: str = "ridge"


class PatchingRequest(BaseModel):
    clean_cache_path: Optional[str] = None
    corrupted_cache_path: Optional[str] = None
    component: str
    timestep: int
    metric_fn: str = "reward"


class AnalysisResponse(BaseModel):
    status: str
    result: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>WorldModelLens API</title>
            <style>
                body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #eee; padding: 2px 5px; }
            </style>
        </head>
        <body>
            <h1>WorldModelLens API</h1>
            <p>Interpretability toolkit for world models</p>
            
            <h2>Endpoints</h2>
            <div class="endpoint">
                <code>POST /analyze</code> - Analyze a trajectory
            </div>
            <div class="endpoint">
                <code>POST /probe</code> - Train a probe on activations
            </div>
            <div class="endpoint">
                <code>POST /patch</code> - Run patching experiment
            </div>
            <div class="endpoint">
                <code>GET /health</code> - Health check
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_episode(request: AnalyzeRequest):
    """Analyze a trajectory and return activations."""
    try:
        obs_tensor = torch.tensor(request.observations)

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        actions_tensor = None
        if request.actions:
            actions_tensor = torch.tensor(request.actions)
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Analysis endpoint ready. Add HookedWorldModel to process.",
                "observations_shape": list(obs_tensor.shape),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/probe", response_model=AnalysisResponse)
async def train_probe(request: ProbingRequest):
    """Train a probe on activations."""
    try:
        prober = LatentProber(seed=42)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Probe training ready. Provide activation cache.",
                "component": request.component,
                "concept": request.concept_name,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/patch", response_model=AnalysisResponse)
async def run_patching(request: PatchingRequest):
    """Run patching experiment."""
    try:
        return AnalysisResponse(
            status="success",
            result={
                "message": "Patching ready. Provide caches for experiment.",
                "component": request.component,
                "timestep": request.timestep,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/trajectory", response_model=AnalysisResponse)
async def upload_trajectory(file: UploadFile = File(...)):
    """Upload trajectory JSON for analysis."""
    try:
        content = await file.read()
        data = json.loads(content)

        return AnalysisResponse(
            status="success",
            result={
                "message": "Trajectory loaded",
                "n_timesteps": len(data.get("observations", [])),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_html_report(
    trajectory_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
) -> str:
    """Generate HTML report from analysis results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WorldModelLens Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            .section { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }
            .metric { display: inline-block; margin: 10px 20px 10px 0; }
            .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #3498db; color: white; }
        </style>
    </head>
    <body>
        <h1>WorldModelLens Analysis Report</h1>
        
        <div class="section">
            <h2>Trajectory Summary</h2>
            <div class="metric">
                <div>Timesteps</div>
                <div class="metric-value">{n_timesteps}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Analysis Results</h2>
            <pre>{results}</pre>
        </div>
    </body>
    </html>
    """.format(
        n_timesteps=trajectory_data.get("n_timesteps", 0),
        results=json.dumps(analysis_results, indent=2),
    )
    return html


@app.get("/report/{analysis_id}")
async def get_report(analysis_id: str):
    """Get HTML report for a completed analysis."""
    return HTMLResponse(content=f"<h1>Analysis {analysis_id}</h1><p>Report placeholder</p>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
