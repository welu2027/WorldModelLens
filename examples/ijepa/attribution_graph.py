import torch
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
    else: # Bipartite
        pos[target_node] = np.array([1, 0.5])
        sorted_context = sorted(context_nodes, key=lambda n: G.get_edge_data(n, target_node)['weight'], reverse=True)
        for i, node in enumerate(sorted_context):
            pos[node] = np.array([0, i / (max(1, len(sorted_context)-1))])
    return pos

def draw_attribution_viz(ax, G, pos, target_node, top_n_labels=3):
    context_nodes = [n for n in G.nodes() if "target" not in n]
    edges = list(G.edges())
    if not edges: return
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    edge_ranks = np.argsort(weights)[::-1]
    
    for i in edge_ranks:
        u, v = edges[i]
        color = plt.cm.viridis(norm_weights[i])
        is_strongest = (i == edge_ranks[0])
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2 + 10 * norm_weights[i], 
                               edge_color=[color] if not is_strongest else ["#FFD700"], 
                               arrowsize=20, ax=ax, alpha=0.8, connectionstyle="arc3,rad=0.1")
        if i in edge_ranks[:top_n_labels]:
            rank_str = f"#{i+1}: {weights[i]:.3f}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): rank_str}, font_size=8, ax=ax, label_pos=0.6)

    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="#F44336", node_size=1800, ax=ax, edgecolors="white", linewidths=3)
    nx.draw_networkx_nodes(G, pos, nodelist=context_nodes, node_color="#2196F3", node_size=1200, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax, font_color="white")

def plot_image_overlay(ax, img, target_id, top_context_ids, grid_size=14):
    ax.imshow(img.resize((224, 224)))
    tx, ty, tw, th = get_patch_rect(target_id, grid_size)
    ax.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=4, edgecolor='#F44336', facecolor='none', label='Target'))
    for i, pid in enumerate(top_context_ids):
        cx, cy, cw, ch = get_patch_rect(pid, grid_size)
        is_strongest = (i == 0)
        color = "#FFD700" if is_strongest else "#2196F3"
        ax.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=4 if is_strongest else 2, edgecolor=color, facecolor='none', alpha=1.0 if is_strongest else 0.7))
        ax.text(cx, cy, f"#{i+1}", color="white", fontsize=8, weight='bold', bbox=dict(facecolor=color, alpha=0.5, pad=0))
    ax.axis('off')

def visualize_research_ijepa(target_ids=None, k=6, layout_mode="importance"):
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    
    config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3)
    adapter = IJEPAAdapter(config)
    
    import os
    checkpoint_path = "ijepa_mini.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
        
    wm = HookedWorldModel(adapter, config)
    wm.adapter.eval()
    
    if target_ids is None:
        target_ids = [42, 114]
    
    # We'll compare middle and final layers of the predictor
    predictor_depth = len(adapter.predictor.blocks)
    layers_to_compare = [predictor_depth // 2, predictor_depth - 1]
    layer_names = ["Middle", "Final"]
    
    num_targets = len(target_ids)
    num_layers = len(layers_to_compare)
    fig = plt.figure(figsize=(18, 6 * num_targets * num_layers))
    
    for t_idx, target_id in enumerate(target_ids):
        # Image overlay only needs to be drawn once per target
        context_ids, _ = get_ijepa_masks(num_context=80) 
        if target_id in context_ids: context_ids.remove(target_id)
        
        adapter.last_context_ids = context_ids
        adapter.last_target_ids = [target_id]

        for l_idx, (layer_idx, name) in enumerate(zip(layers_to_compare, layer_names)):
            with torch.no_grad():
                wm.run_with_cache(img_tensor)
                # Tap the selected layer
                attn = adapter.predictor.blocks[layer_idx].attn.last_attn_weights
                target_to_context = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()

            sorted_indices = np.argsort(target_to_context)[::-1][:k]
            top_context_ids = [context_ids[i] for i in sorted_indices]
            top_weights = [target_to_context[i] for i in sorted_indices]

            # Subplots: Target Row, Column 1=Graph, Column 2=Image
            row_idx = t_idx * num_layers + l_idx
            gs = fig.add_gridspec(num_targets * num_layers, 2, width_ratios=[1.2, 1])
            ax_graph = fig.add_subplot(gs[row_idx, 0])
            ax_img = fig.add_subplot(gs[row_idx, 1])
            
            # Build Graph
            G = nx.DiGraph()
            target_node = f"target_{target_id}"
            for pid, w in zip(top_context_ids, top_weights):
                G.add_edge(f"patch_{pid}", target_node, weight=float(w))

            pos = compute_structured_layout(G, target_node, mode=layout_mode)
            draw_attribution_viz(ax_graph, G, pos, target_node)
            ax_graph.set_title(f"I-JEPA Flow | Layer: {name} ({layer_idx})\nTarget: {target_id}", fontsize=12, loc='left')
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
    visualize_research_ijepa(target_ids=[42, 114], k=6, layout_mode="importance")
