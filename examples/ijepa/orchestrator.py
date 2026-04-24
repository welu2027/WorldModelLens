"""run_suite.py — Unified I-JEPA Interpretability Orchestrator

Aggregates all interpretability experiments into a single call:

    results = run_full_suite(image, checkpoint_path, seed=42)
    # -> JSON-serializable dict + suite_results/suite_summary.png

Experiments covered (model loaded once, shared across all):
  1. Attribution Graph   — attention weights, middle vs. final predictor layer
  2. Circuit Discovery  — energy-threshold circuit patches
  3. Faithfulness       — Top-K ablation/reconstruction AUC (AOPC proxy)
  4. Structural         — predictor layer importance via forward-hook ablation
  5. Formal Circuit     — greedy minimal patch set reaching performance threshold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image


# helpers

def _seeded_masks(
    seed: int,
    num_context: int = 80,
    num_target: int = 3,
    num_patches: int = 196,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_patches)
    context_ids = sorted(indices[:num_context].tolist())
    target_ids = sorted(indices[num_context : num_context + num_target].tolist())
    return context_ids, target_ids


def _load_model(
    checkpoint_path: Optional[str],
    config: WorldModelConfig,
) -> tuple[HookedWorldModel, IJEPAAdapter]:
    adapter = IJEPAAdapter(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        adapter.load_state_dict(sd, strict=False)
        print(f"[Suite] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[Suite] No checkpoint — using random weights")
    adapter.eval()
    wm = HookedWorldModel(adapter, config)
    return wm, adapter


def _predictor_attn(
    adapter: IJEPAAdapter,
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
    layer_idx: int = -1,
) -> np.ndarray:
    """Run one forward pass; return predictor attention for target→context."""
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = [target_id]
    with torch.no_grad():
        wm.run_with_cache(img_tensor)
        raw = adapter.predictor.blocks[layer_idx].attn.last_attn_weights
        weights = raw[0].mean(0)[-1, : len(context_ids)].cpu().numpy()
    return weights


def _target_gt(
    adapter: IJEPAAdapter,
    wm: HookedWorldModel,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
) -> torch.Tensor:
    """Return ground-truth target latent [1, C]."""
    adapter.last_context_ids = context_ids
    adapter.last_target_ids = [target_id]
    with torch.no_grad():
        _, cache = wm.run_with_cache(img_tensor)
        reps = cache.get("target_encoding", 0)
        if reps is not None:
            gt = reps[0, [target_id], :] if reps.dim() == 3 else reps[[target_id], :]
        else:
            gt = adapter.target_encode(img_tensor)[:, [target_id], :]
    return gt


def _predict_mse(
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    active_ids: list[int],
    target_id: int,
    target_gt: torch.Tensor,
) -> float:
    if not active_ids:
        return float("nan")
    adapter.last_context_ids = active_ids
    adapter.last_target_ids = [target_id]
    with torch.no_grad():
        ctx_lat, _ = adapter.encode(img_tensor)
        pred = adapter.dynamics(ctx_lat).squeeze(0)
    return F.mse_loss(pred.flatten(), target_gt.flatten()).item()


# experiment 1: attribution

def _run_attribution(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_ids: list[int],
    k: int = 6,
) -> dict:
    predictor_depth = len(adapter.predictor.blocks)
    layers = {"middle": predictor_depth // 2, "final": predictor_depth - 1}

    per_target: dict[str, dict] = {}
    for target_id in target_ids:
        ctx = [c for c in context_ids if c != target_id]
        per_layer: dict[str, dict] = {}
        for name, layer_idx in layers.items():
            w = _predictor_attn(adapter, wm, img_tensor, ctx, target_id, layer_idx)
            top_idx = np.argsort(w)[::-1][:k]
            per_layer[name] = {
                "top_context_patches": [ctx[i] for i in top_idx],
                "top_weights": [float(w[i]) for i in top_idx],
                "mean_attn": float(w.mean()),
                "entropy": float(-(w * np.log(w + 1e-9)).sum()),
            }
        per_target[str(target_id)] = per_layer

    return {"targets": per_target, "k": k, "layers_analyzed": list(layers.keys())}


# experiment 2: circuit discovery

def _run_circuit(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
    threshold_pct: float = 0.005,
) -> dict:
    ctx = [c for c in context_ids if c != target_id]
    w = _predictor_attn(adapter, wm, img_tensor, ctx, target_id)

    total_energy = w.sum()
    circuit_idx = np.where(w > total_energy * threshold_pct)[0]
    pairs = sorted(
        [(ctx[i], float(w[i])) for i in circuit_idx], key=lambda x: x[1], reverse=True
    )

    return {
        "target_patch": target_id,
        "threshold_pct": threshold_pct,
        "circuit_size": len(pairs),
        "circuit_patches": [p for p, _ in pairs],
        "circuit_weights": [wt for _, wt in pairs],
        "total_context_patches": len(ctx),
        "energy_coverage": float(sum(wt for _, wt in pairs) / (total_energy + 1e-9)),
    }


# experiment 3: faithfulness (AOPC proxy)

def _run_faithfulness(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
    seed: int,
    n_steps: int = 15,
) -> dict:
    rng = np.random.default_rng(seed)
    ctx = [c for c in context_ids if c != target_id]

    w = _predictor_attn(adapter, wm, img_tensor, ctx, target_id)
    sorted_idx = np.argsort(w)[::-1]
    top_ctx = [ctx[i] for i in sorted_idx]
    rand_ctx = rng.permutation(ctx).tolist()

    gt = _target_gt(adapter, wm, img_tensor, ctx, target_id)

    step = max(1, len(ctx) // n_steps)
    k_values = list(range(1, len(ctx), step))[:n_steps]

    ablation: dict[str, list] = {"top": [], "random": [], "k_values": k_values}
    reconstruction: dict[str, list] = {"top": [], "random": [], "k_values": k_values}

    for k in k_values:
        # ablation: remove k patches
        ablation["top"].append(_predict_mse(adapter, img_tensor, [c for c in ctx if c not in top_ctx[:k]], target_id, gt))
        ablation["random"].append(_predict_mse(adapter, img_tensor, [c for c in ctx if c not in rand_ctx[:k]], target_id, gt))
        # reconstruction: use only k patches
        reconstruction["top"].append(_predict_mse(adapter, img_tensor, top_ctx[:k], target_id, gt))
        reconstruction["random"].append(_predict_mse(adapter, img_tensor, rand_ctx[:k], target_id, gt))

    def _auc(vals: list) -> float:
        clean = [v for v in vals if v == v]  # drop NaN
        return float(np.trapz(clean) / max(1, len(clean))) if clean else 0.0

    top_auc = _auc(ablation["top"])
    rand_auc = _auc(ablation["random"])

    return {
        "target_patch": target_id,
        "ablation": ablation,
        "reconstruction": reconstruction,
        "aopc": top_auc - rand_auc,
        "top_ablation_auc": top_auc,
        "random_ablation_auc": rand_auc,
    }


# experiment 4: structural (layer importance)

def _run_structural(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
) -> dict:
    ctx = [c for c in context_ids if c != target_id]
    gt = _target_gt(adapter, wm, img_tensor, ctx, target_id)

    # Baseline MSE with all context
    clean_mse = _predict_mse(adapter, img_tensor, ctx, target_id, gt)

    layer_scores: list[dict] = []
    for layer_idx, block in enumerate(adapter.predictor.blocks):
        # Zero the output of this block via a temporary PyTorch hook
        handle = block.register_forward_hook(
            lambda m, inp, out: torch.zeros_like(out)
        )
        try:
            ablated_mse = _predict_mse(adapter, img_tensor, ctx, target_id, gt)
        finally:
            handle.remove()

        delta = ablated_mse - clean_mse
        layer_scores.append(
            {
                "layer": layer_idx,
                "clean_mse": clean_mse,
                "ablated_mse": float(ablated_mse),
                "mse_delta": float(delta),
                "importance": float(delta) / (clean_mse + 1e-9),
            }
        )

    best_layer = max(layer_scores, key=lambda x: x["mse_delta"])["layer"]
    return {
        "target_patch": target_id,
        "clean_mse": clean_mse,
        "layer_importance": layer_scores,
        "most_important_layer": best_layer,
    }


# experiment 5: formal minimal circuit

def _run_formal_circuit(
    wm: HookedWorldModel,
    adapter: IJEPAAdapter,
    img_tensor: torch.Tensor,
    context_ids: list[int],
    target_id: int,
    threshold: float = 0.85,
) -> dict:
    ctx = [c for c in context_ids if c != target_id]

    # Rank patches by final-layer predictor attention
    adapter.last_context_ids = ctx
    adapter.last_target_ids = [target_id]
    with torch.no_grad():
        _, cache = wm.run_with_cache(img_tensor)
        attn = adapter.predictor.blocks[-1].attn.last_attn_weights
        attributions = attn[0].mean(0)[-1, : len(ctx)].cpu().numpy()
        reps = cache.get("target_encoding", 0)
        if reps is not None:
            gt = reps[0, [target_id], :] if reps.dim() == 3 else reps[[target_id], :]
        else:
            gt = adapter.target_encode(img_tensor)[:, [target_id], :]

    ranked_ctx = [ctx[i] for i in np.argsort(attributions)[::-1]]
    baseline_mse = torch.mean(gt**2).item()

    def _perf(active: list[int]) -> float:
        if not active:
            return 0.0
        mse = _predict_mse(adapter, img_tensor, active, target_id, gt)
        return 1.0 - mse / (baseline_mse + 1e-6)

    # Greedy build until threshold is met
    circuit: list[int] = []
    perf_curve: list[float] = []
    for patch in ranked_ctx:
        circuit.append(patch)
        perf = _perf(circuit)
        perf_curve.append(float(perf))
        if perf >= threshold:
            break

    return {
        "target_patch": target_id,
        "threshold": threshold,
        "circuit_patches": circuit,
        "circuit_size": len(circuit),
        "total_context": len(ctx),
        "sparsity": 1.0 - len(circuit) / max(1, len(ctx)),
        "achieved_performance": perf_curve[-1] if perf_curve else 0.0,
        "full_context_performance": float(_perf(ranked_ctx)),
        "performance_curve": perf_curve,
    }


# summary figure

def _plot_summary(results: dict, output_dir: str) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("I-JEPA Interpretability Suite — Summary", fontsize=14, fontweight="bold")

    # Panel 1: attribution weights (first target, final layer)
    ax = axes[0, 0]
    try:
        tid = next(iter(results["attribution"]["targets"]))
        final = results["attribution"]["targets"][tid].get("final", {})
        weights = final.get("top_weights", [])
        patch_ids = final.get("top_context_patches", [])
        ax.bar(range(len(weights)), weights, color="#2196F3")
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels([f"p{p}" for p in patch_ids], rotation=45, fontsize=8)
        ax.set_title(f"Attribution — Target {tid} (Final Layer)")
    except Exception:
        ax.set_title("Attribution — no data")
    ax.set_ylabel("Attention Weight")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: circuit size per target
    ax = axes[0, 1]
    try:
        tids = list(results["circuit"]["per_target"].keys())
        sizes = [results["circuit"]["per_target"][t]["circuit_size"] for t in tids]
        ax.bar(tids, sizes, color="#4CAF50")
        ax.set_title("Circuit Size per Target Patch")
        ax.set_ylabel("# Circuit Patches")
    except Exception:
        ax.set_title("Circuit Discovery — no data")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: faithfulness ablation curves (first target)
    ax = axes[1, 0]
    try:
        first_key = next(iter(results["faithfulness"]["per_target"]))
        f = results["faithfulness"]["per_target"][first_key]
        k_vals = f["ablation"]["k_values"]
        ax.plot(k_vals[: len(f["ablation"]["top"])], f["ablation"]["top"], "r-o", markersize=4, label="Top-K removed")
        ax.plot(k_vals[: len(f["ablation"]["random"])], f["ablation"]["random"], color="gray", linestyle="--", label="Random removed")
        ax.legend(fontsize=8)
        ax.set_title(f"Faithfulness Ablation — Target {first_key}")
    except Exception:
        ax.set_title("Faithfulness — no data")
    ax.set_xlabel("Patches Removed")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)

    # Panel 4: predictor layer importance (first target)
    ax = axes[1, 1]
    try:
        first_key = next(iter(results["structural"]["per_target"]))
        s = results["structural"]["per_target"][first_key]
        layers = [d["layer"] for d in s["layer_importance"]]
        deltas = [d["mse_delta"] for d in s["layer_importance"]]
        ax.bar(layers, deltas, color="#FF5722")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_title(f"Layer Importance — Target {first_key}")
    except Exception:
        ax.set_title("Structural — no data")
    ax.set_xlabel("Predictor Layer")
    ax.set_ylabel("MSE Delta on Ablation")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "suite_summary.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# main orchestrator

def run_full_suite(
    image: Union[str, Image.Image, None] = None,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
    output_dir: str = "suite_results",
    target_ids: Optional[List[int]] = None,
    num_context: int = 80,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all I-JEPA interpretability experiments and return JSON-serializable results.

    Args:
        image: PIL image, path to image file, or None (downloads sample dog image).
        checkpoint_path: Path to .pth checkpoint. None uses random weights.
        seed: Random seed for reproducible masking and sampling.
        output_dir: Directory where figures and suite_results.json are saved.
        target_ids: Explicit list of target patch IDs. None samples from seed.
        num_context: Number of context patches.
        config_overrides: Dict of WorldModelConfig field overrides (e.g. {"n_layers": 12}).

    Returns:
        Dict with keys: metadata, attribution, circuit, faithfulness,
        structural, formal_circuit, figures.
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load image
    if image is None:
        raw_img = get_sample_image()
    elif isinstance(image, str):
        raw_img = Image.open(image).convert("RGB")
    else:
        raw_img = image
    img_tensor = preprocess_image(raw_img)

    # Build config (mini ViT defaults)
    cfg_kwargs: Dict[str, Any] = dict(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384
    )
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    config = WorldModelConfig(**cfg_kwargs)

    wm, adapter = _load_model(checkpoint_path, config)

    # Seeded masks
    context_ids, auto_targets = _seeded_masks(seed, num_context=num_context)
    if target_ids is None:
        target_ids = auto_targets[:2]

    print(
        f"[Suite] seed={seed} | targets={target_ids} | "
        f"context={len(context_ids)} patches | output={output_dir}/"
    )

    results: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "checkpoint": checkpoint_path,
            "target_ids": target_ids,
            "context_size": len(context_ids),
            "config": cfg_kwargs,
        }
    }

    # 1: Attribution
    print("[Suite] 1/5 Attribution Graph...")
    results["attribution"] = _run_attribution(wm, adapter, img_tensor, context_ids, target_ids)

    # 2: Circuit discovery
    print("[Suite] 2/5 Circuit Discovery...")
    results["circuit"] = {
        "per_target": {
            str(tid): _run_circuit(wm, adapter, img_tensor, context_ids, tid)
            for tid in target_ids
        }
    }

    # 3: Faithfulness
    print("[Suite] 3/5 Faithfulness Evaluation...")
    per_faith = {
        str(tid): _run_faithfulness(wm, adapter, img_tensor, context_ids, tid, seed)
        for tid in target_ids
    }
    results["faithfulness"] = {
        "per_target": per_faith,
        "mean_aopc": float(np.mean([v["aopc"] for v in per_faith.values()])),
    }

    # 4: Structural
    print("[Suite] 4/5 Structural Circuits...")
    results["structural"] = {
        "per_target": {
            str(tid): _run_structural(wm, adapter, img_tensor, context_ids, tid)
            for tid in target_ids
        }
    }

    # 5: Formal circuit
    print("[Suite] 5/5 Formal Minimal Circuit...")
    per_formal = {
        str(tid): _run_formal_circuit(wm, adapter, img_tensor, context_ids, tid)
        for tid in target_ids
    }
    results["formal_circuit"] = {
        "per_target": per_formal,
        "mean_circuit_size": float(np.mean([v["circuit_size"] for v in per_formal.values()])),
        "mean_sparsity": float(np.mean([v["sparsity"] for v in per_formal.values()])),
    }

    # Summary figure
    print("[Suite] Generating summary figure...")
    figures: Dict[str, str] = {}
    try:
        figures["summary"] = _plot_summary(results, output_dir)
    except Exception as exc:
        print(f"[Suite] Warning: summary figure failed — {exc}")

    results["figures"] = figures
    results["metadata"]["elapsed_seconds"] = round(time.time() - t0, 2)

    json_path = os.path.join(output_dir, "suite_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"[Suite] Done in {results['metadata']['elapsed_seconds']}s → {json_path}")
    return results


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full I-JEPA interpretability suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", default=None, help="Path to input image (None = sample dog image)")
    parser.add_argument("--checkpoint", default="ijepa_mini.pth", help="Path to .pth checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="suite_results", help="Output directory")
    parser.add_argument("--targets", type=int, nargs="+", default=None, help="Target patch IDs")
    parser.add_argument("--num-context", type=int, default=80, help="Number of context patches")
    args = parser.parse_args()

    res = run_full_suite(
        image=args.image,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        target_ids=args.targets,
        num_context=args.num_context,
    )

    print("\n=== Suite Summary ===")
    print(f"  Targets         : {res['metadata']['target_ids']}")
    print(f"  AOPC (mean)     : {res['faithfulness']['mean_aopc']:.4f}")
    print(f"  Circuit size    : {res['formal_circuit']['mean_circuit_size']:.1f} / {res['metadata']['context_size']}")
    print(f"  Sparsity        : {res['formal_circuit']['mean_sparsity']:.1%}")
    print(f"  Output dir      : {args.output_dir}/")
    sys.exit(0)
