import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
from matplotlib import cm
import os

# ---------------------------------------------------------------------------
# Integrated Gradients for I-JEPA patch attribution
# ---------------------------------------------------------------------------

class IJEPAIntegratedGradients:
    """
    Computes Integrated Gradients (IG) attributions for context patches
    with respect to the prediction quality for a given target patch.

    Theory:
        IG_i(x) = (x_i - x'_i) * ∫₀¹ [∂F/∂x_i](x' + α(x - x')) dα

    Where:
        x   = actual patch embeddings (the input)
        x'  = baseline embeddings (mean embedding — on-manifold)
        F   = scalar prediction score (negative MSE: higher = better prediction)
        i   = index over (patch, embedding_dim) space

    Summing |IG_i| over the embedding dimension gives a scalar attribution per patch.

    Axioms satisfied:
        - Completeness: sum of attributions = F(x) - F(x') (verifiable via assertion)
        - Sensitivity: if x_i ≠ x'_i and the model depends on x_i, IG_i ≠ 0
        - Implementation invariance: output is independent of model internals

    We differentiate through the context encoder + predictor jointly.
    The target encoder runs without gradients (as in the original I-JEPA loss).
    """

    def __init__(self, adapter: IJEPAAdapter, n_steps: int = 50):
        """
        Args:
            adapter: Loaded IJEPAAdapter (in eval mode).
            n_steps: Number of Riemann sum steps. More steps = more accurate.
                     50 is sufficient in practice; 100 for publication-quality.
        """
        self.adapter = adapter
        self.n_steps = n_steps

        # Put model in eval to avoid batchnorm/dropout stochasticity
        self.adapter.eval()

    def _forward_score(
        self,
        patch_emb: torch.Tensor,   # [1, N_all, C] — raw patch embeddings (NO pos added yet)
        context_ids: list,
        target_id: int,
        target_gt: torch.Tensor,   # [1, C] — ground truth target embedding
    ) -> torch.Tensor:
        """Full forward from post-patchify embeddings to scalar score.

        We inject BEFORE positional embeddings so that the interpolation path is:
            baseline_raw_emb → actual_raw_emb
        and the positional encoding is applied consistently at each interpolation step.
        This is correct: IG should attribute importance of the raw patch content,
        not of position (which is fixed and identical for all αs).

        forward_blocks() expects input AFTER positional embedding + dropout,
        so we replicate those steps here manually.
        """
        # Select context patches and add positional embeddings
        ctx_ids_t = torch.tensor(context_ids, device=patch_emb.device)
        ctx_emb = patch_emb[:, ctx_ids_t, :]                          # [1, N_ctx, C]
        pos = self.adapter.context_encoder.pos_embed[:, ctx_ids_t, :] # [1, N_ctx, C]
        ctx_with_pos = ctx_emb + pos                                   # [1, N_ctx, C]

        # forward_blocks applies pos_drop first (identity in eval mode), then blocks+norm
        ctx_latents = self.adapter.context_encoder.forward_blocks(ctx_with_pos)

        # Run predictor
        pred = self.adapter.predictor(ctx_latents, context_ids, [target_id])
        # pred: [1, 1, C]

        # Score = negative MSE (more negative = worse prediction)
        score = -F.mse_loss(pred.squeeze(1), target_gt)
        return score

    @torch.no_grad()
    def _get_target_gt(self, img_tensor: torch.Tensor, target_id: int) -> torch.Tensor:
        """Get ground-truth target embedding from the (frozen) target encoder."""
        target_reps = self.adapter.target_encoder(img_tensor)  # [1, N, C]
        return target_reps[:, [target_id], :]  # [1, 1, C]

    def compute(
        self,
        img_tensor: torch.Tensor,   # [1, 3, H, W]
        context_ids: list,
        target_id: int,
    ) -> np.ndarray:
        """Compute IG attributions for each context patch.

        Returns:
            attributions: np.ndarray of shape [N_context], scalar per patch.
                          Higher value = patch is more causally important.
        """
        img_tensor = img_tensor.float()
        device = next(self.adapter.parameters()).device
        img_tensor = img_tensor.to(device)

        # 1. Compute actual and baseline patch embeddings
        with torch.no_grad():
            actual_emb = self.adapter.context_encoder.patch_embed(img_tensor)  # [1, N, C]
            # Baseline: mean embedding over all patches (on-manifold)
            baseline_emb = actual_emb.mean(dim=1, keepdim=True).expand_as(actual_emb)

        # 2. Pre-compute target ground truth (no grad needed)
        target_gt = self._get_target_gt(img_tensor, target_id)          # [1, 1, C]
        target_gt_flat = target_gt.squeeze(0)                             # [1, C]

        # 3. Riemann sum: integrate gradients from baseline to actual
        accumulated_grads = torch.zeros_like(actual_emb)  # [1, N, C]

        for step in range(self.n_steps):
            alpha = step / self.n_steps  # α ∈ [0, 1)

            # Interpolated embeddings: x' + α(x - x')
            interp_emb = (baseline_emb + alpha * (actual_emb - baseline_emb)).detach()
            interp_emb.requires_grad_(True)

            # Forward score at interpolated point
            score = self._forward_score(interp_emb, context_ids, target_id, target_gt_flat)

            # Backward to get ∂F/∂x at this interpolation point
            score.backward()

            if interp_emb.grad is not None:
                accumulated_grads += interp_emb.grad.detach()
            
            # Zero any lingering grads (interp_emb is a leaf, so this is safe)
            interp_emb.grad = None

        # 4. IG formula: (actual - baseline) * (1/n_steps) * Σ gradients
        delta = actual_emb.detach() - baseline_emb.detach()          # [1, N, C]
        ig_per_dim = delta * (accumulated_grads / self.n_steps)       # [1, N, C]

        # 5. Sum over embedding dim → scalar attribution per patch
        ig_per_patch = ig_per_dim.abs().sum(dim=-1).squeeze(0)       # [N_all]

        # Only return attributions for context patches
        ctx_attrs = ig_per_patch[context_ids].cpu().numpy()           # [N_ctx]

        # Verify completeness (approximate): sum of IG ≈ F(x) - F(x')
        # We compute this for logging but don't enforce it as an assert
        with torch.no_grad():
            actual_score = self._forward_score(
                actual_emb.detach(), context_ids, target_id, target_gt_flat
            ).item()
            baseline_score = self._forward_score(
                baseline_emb.detach(), context_ids, target_id, target_gt_flat
            ).item()
        ig_sum = ig_per_dim.sum().item()
        expected_diff = actual_score - baseline_score
        completeness_error = abs(ig_sum - expected_diff) / (abs(expected_diff) + 1e-8)
        print(
            f"  IG completeness: sum(IG)={ig_sum:.4f}, F(x)-F(x')={expected_diff:.4f}, "
            f"error={completeness_error:.2%} (< 5% is good)"
        )

        return ctx_attrs


# ---------------------------------------------------------------------------
# Visualization helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_patch_rect(patch_id, grid_size=14, img_size=224):
    """Calculates [x, y, w, h] for a patch ID."""
    row = patch_id // grid_size
    col = patch_id % grid_size
    p_size = img_size // grid_size
    return col * p_size, row * p_size, p_size, p_size


def compute_structured_layout(G, target_node, mode="importance", grid_size=14):
    """Computes research-grade positions."""
    pos = {}
    context_nodes = [n for n in G.nodes() if "target" not in n]

    if mode == "importance":
        pos[target_node] = np.array([0, 0])
        weights = {n: G.get_edge_data(n, target_node)['weight'] for n in context_nodes}
        sorted_context = sorted(context_nodes, key=lambda n: weights[n], reverse=True)
        num_context = len(sorted_context)
        for i, node in enumerate(sorted_context):
            angle = np.pi/2 - (2 * np.pi * i / num_context)
            w = weights[node]
            radius = 1.0 - (w * 0.4)
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    elif mode == "spatial":
        pos[target_node] = np.array([int(target_node.split("_")[1]) % grid_size,
                                     grid_size - (int(target_node.split("_")[1]) // grid_size)])
        for node in context_nodes:
            pid = int(node.split("_")[1])
            pos[node] = np.array([pid % grid_size, grid_size - (pid // grid_size)])
    else:  # Bipartite
        pos[target_node] = np.array([1, 0.5])
        sorted_context = sorted(context_nodes,
                                key=lambda n: G.get_edge_data(n, target_node)['weight'],
                                reverse=True)
        for i, node in enumerate(sorted_context):
            pos[node] = np.array([0, i / (max(1, len(sorted_context)-1))])
    return pos


def draw_attribution_viz(ax, G, pos, target_node, top_n_labels=3):
    context_nodes = [n for n in G.nodes() if "target" not in n]
    edges = list(G.edges())
    if not edges:
        return
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    edge_ranks = np.argsort(weights)[::-1]

    for i in edge_ranks:
        u, v = edges[i]
        cmap = plt.get_cmap('viridis')
        color = cmap(norm_weights[i])
        is_strongest = (i == edge_ranks[0])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=2 + 10 * norm_weights[i],
                               edge_color=color if not is_strongest else "#FFD700",
                               arrowsize=20, ax=ax, alpha=0.8,
                               connectionstyle="arc3,rad=0.1")
        if i in edge_ranks[:top_n_labels]:
            rank_str = f"#{i+1}: {weights[i]:.3f}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): rank_str},
                                         font_size=8, ax=ax, label_pos=0.6)

    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="#F44336",
                           node_size=1800, ax=ax, edgecolors="white", linewidths=3)
    nx.draw_networkx_nodes(G, pos, nodelist=context_nodes, node_color="#2196F3",
                           node_size=1200, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold",
                            ax=ax, font_color="white")


def plot_image_overlay(ax, img, target_id, top_context_ids, label="", grid_size=14):
    ax.imshow(img.resize((224, 224)))
    tx, ty, tw, th = get_patch_rect(target_id, grid_size)
    ax.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=4,
                                   edgecolor='#F44336', facecolor='none', label='Target'))
    for i, pid in enumerate(top_context_ids):
        cx, cy, cw, ch = get_patch_rect(pid, grid_size)
        is_strongest = (i == 0)
        color = "#FFD700" if is_strongest else "#2196F3"
        ax.add_patch(patches.Rectangle((cx, cy), cw, ch,
                                       linewidth=4 if is_strongest else 2,
                                       edgecolor=color, facecolor='none',
                                       alpha=1.0 if is_strongest else 0.7))
        ax.text(cx, cy, f"#{i+1}", color="white", fontsize=8, weight='bold',
                bbox=dict(facecolor=color, alpha=0.5, pad=0))
    if label:
        ax.set_title(label, fontsize=10, pad=4)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Comparison visualization: Attention vs Integrated Gradients
# ---------------------------------------------------------------------------

def visualize_ig_vs_attention(target_ids=None, k=6, n_ig_steps=50):
    """
    Runs both attention-based and IG-based patch ranking for each target,
    then plots them side-by-side for comparison.

    Comparison helps researchers understand:
    - WHERE the two methods agree (strong signal — likely truly important patches)
    - WHERE they disagree (attention may be spurious; IG attribution is more trustworthy)

    An overlap score is computed: |intersection(top-K_attn, top-K_IG)| / K.
    High overlap = attention is a faithful proxy. Low overlap = IG reveals patches
    that attention-based visualization misses.
    """
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)

    config = WorldModelConfig(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384
    )
    adapter = IJEPAAdapter(config)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "ijepa_mini.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        adapter.load_state_dict(
            torch.load(checkpoint_path, weights_only=True), strict=False
        )

    adapter.eval()
    wm = HookedWorldModel(adapter, config)

    if target_ids is None:
        target_ids = [42, 114]

    ig_computer = IJEPAIntegratedGradients(adapter, n_steps=n_ig_steps)

    num_targets = len(target_ids)
    # 3 columns: attention graph | IG image overlay | attention image overlay
    fig, axes = plt.subplots(num_targets, 3, figsize=(20, 7 * num_targets))
    if num_targets == 1:
        axes = axes[np.newaxis, :]

    for row, target_id in enumerate(target_ids):
        context_ids, _ = get_ijepa_masks(num_context=80)
        if target_id in context_ids:
            context_ids.remove(target_id)

        adapter.last_context_ids = context_ids
        adapter.last_target_ids = [target_id]

        # ---- Attention-based ranking (uses last layer of predictor) ----
        with torch.no_grad():
            wm.run_with_cache(img_tensor)
            attn = adapter.predictor.blocks[-1].attn.last_attn_weights
            target_to_context_attn = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()

        attn_sorted = np.argsort(target_to_context_attn)[::-1]
        top_attn_ids = [context_ids[i] for i in attn_sorted[:k]]
        top_attn_weights = [target_to_context_attn[i] for i in attn_sorted[:k]]

        # ---- Integrated Gradients ranking ----
        print(f"\nComputing IG for target {target_id} ({n_ig_steps} steps)...")
        ig_attrs = ig_computer.compute(img_tensor, context_ids, target_id)

        ig_sorted = np.argsort(ig_attrs)[::-1]
        top_ig_ids = [context_ids[i] for i in ig_sorted[:k]]
        top_ig_weights = [ig_attrs[i] for i in ig_sorted[:k]]

        # ---- Overlap score ----
        overlap = len(set(top_attn_ids) & set(top_ig_ids)) / k
        print(f"  Overlap (top-{k} Attention & IG) = {overlap:.0%}")

        # ---- Column 0: Attribution graph (attention weights, same as before) ----
        ax_graph = axes[row, 0]
        G = nx.DiGraph()
        target_node = f"target_{target_id}"
        for pid, w in zip(top_attn_ids, top_attn_weights):
            G.add_edge(f"patch_{pid}", target_node, weight=float(w))
        pos = compute_structured_layout(G, target_node, mode="importance")
        draw_attribution_viz(ax_graph, G, pos, target_node)
        ax_graph.set_title(
            f"Attention Flow | Target {target_id}\n(top-{k} by predictor attention)",
            fontsize=11, loc='left'
        )
        ax_graph.axis('off')

        # ---- Column 1: IG image overlay ----
        ax_ig = axes[row, 1]
        plot_image_overlay(
            ax_ig, raw_img, target_id, top_ig_ids,
            label=f"IG Attribution | Target {target_id}\n(top-{k}, overlap={overlap:.0%})"
        )

        # ---- Column 2: Attention image overlay ----
        ax_attn = axes[row, 2]
        plot_image_overlay(
            ax_attn, raw_img, target_id, top_attn_ids,
            label=f"Attention Ranking | Target {target_id}\n(top-{k})"
        )

    plt.suptitle(
        "Patch Attribution Comparison: Integrated Gradients vs Attention Weights\n"
        "Gold box = #1 ranked patch. Blue boxes = top-K. "
        "High overlap → attention is faithful; Low overlap → IG reveals hidden dependencies.",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if os.environ.get("SAVE_PLOT"):
        plt.savefig("attribution_ig_vs_attention.png", dpi=150, bbox_inches='tight')
        print("Saved to attribution_ig_vs_attention.png")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Legacy entry point (kept for backward compatibility)
# ---------------------------------------------------------------------------

def visualize_research_ijepa(target_ids=None, k=6, layout_mode="importance"):
    """Original attention-only visualization — retained for backward compatibility."""
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)

    config = WorldModelConfig(
        backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384
    )
    adapter = IJEPAAdapter(config)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "ijepa_mini.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        adapter.load_state_dict(
            torch.load(checkpoint_path, weights_only=True), strict=False
        )

    wm = HookedWorldModel(adapter, config)
    wm.adapter.eval()

    if target_ids is None:
        target_ids = [42, 114]

    predictor_depth = len(adapter.predictor.blocks)
    layers_to_compare = [predictor_depth // 2, predictor_depth - 1]
    layer_names = ["Middle", "Final"]

    num_targets = len(target_ids)
    num_layers = len(layers_to_compare)
    fig = plt.figure(figsize=(18, 6 * num_targets * num_layers))

    for t_idx, target_id in enumerate(target_ids):
        context_ids, _ = get_ijepa_masks(num_context=80)
        if target_id in context_ids:
            context_ids.remove(target_id)

        adapter.last_context_ids = context_ids
        adapter.last_target_ids = [target_id]

        for l_idx, (layer_idx, name) in enumerate(zip(layers_to_compare, layer_names)):
            with torch.no_grad():
                wm.run_with_cache(img_tensor)
                attn = adapter.predictor.blocks[layer_idx].attn.last_attn_weights
                target_to_context = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()

            sorted_indices = np.argsort(target_to_context)[::-1][:k]
            top_context_ids = [context_ids[i] for i in sorted_indices]
            top_weights = [target_to_context[i] for i in sorted_indices]

            row_idx = t_idx * num_layers + l_idx
            gs = fig.add_gridspec(num_targets * num_layers, 2, width_ratios=[1.2, 1])
            ax_graph = fig.add_subplot(gs[row_idx, 0])
            ax_img = fig.add_subplot(gs[row_idx, 1])

            G = nx.DiGraph()
            target_node = f"target_{target_id}"
            for pid, w in zip(top_context_ids, top_weights):
                G.add_edge(f"patch_{pid}", target_node, weight=float(w))

            pos = compute_structured_layout(G, target_node, mode=layout_mode)
            draw_attribution_viz(ax_graph, G, pos, target_node)
            ax_graph.set_title(
                f"I-JEPA Flow | Layer: {name} ({layer_idx})\nTarget: {target_id}",
                fontsize=12, loc='left'
            )
            ax_graph.axis('off')

            plot_image_overlay(ax_img, raw_img, target_id, top_context_ids)
            ax_img.set_title(f"Grounding | {name} Layer")

    plt.tight_layout()
    if os.environ.get("SAVE_PLOT"):
        plt.savefig("attribution_comparison.png")
        print("Comparison plot saved to attribution_comparison.png")
    else:
        plt.show()


if __name__ == "__main__":
    # New default: run the IG vs Attention comparison
    # Pass n_ig_steps=20 for a faster (lower accuracy) run during testing
    visualize_ig_vs_attention(target_ids=[42, 114], k=6, n_ig_steps=50)
