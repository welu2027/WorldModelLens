import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from ijepa_model import IJEPAModel
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
        # Radial layout: strongest at top, closer to center
        pos[target_node] = np.array([0, 0])
        weights = {n: G.get_edge_data(n, target_node)['weight'] for n in context_nodes}
        sorted_context = sorted(context_nodes, key=lambda n: weights[n], reverse=True)
        
        num_context = len(sorted_context)
        for i, node in enumerate(sorted_context):
            # i=0 (strongest) -> angle=pi/2 (top)
            angle = np.pi/2 - (2 * np.pi * i / num_context)
            # Normalize weight for distance (stronger = closer)
            w = weights[node]
            radius = 1.0 - (w * 0.4) 
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
    elif mode == "spatial":
        # Position nodes based on image grid coordinates
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
    """Draws the nodes and edges with research-grade encoding."""
    context_nodes = [n for n in G.nodes() if "target" not in n]
    
    # Edges
    edges = list(G.edges())
    weights = np.array([G[u][v]['weight'] for u, v in edges])
    norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    
    # Sort edges for ranking
    edge_ranks = np.argsort(weights)[::-1]
    
    for i in edge_ranks:
        u, v = edges[i]
        # Scientific colormap
        color = plt.cm.viridis(norm_weights[i])
        is_strongest = (i == edge_ranks[0])
        
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            width=2 + 10 * norm_weights[i], 
            edge_color=[color] if not is_strongest else ["#FFD700"], # Golden for #1
            arrowsize=20, ax=ax, alpha=0.8,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Reduced Clutter labels
        if i in edge_ranks[:top_n_labels]:
            rank_str = f"#{i+1}: {weights[i]:.3f}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): rank_str}, font_size=8, ax=ax, label_pos=0.6)

    # Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="#F44336", node_size=1800, ax=ax, edgecolors="white", linewidths=3)
    nx.draw_networkx_nodes(G, pos, nodelist=context_nodes, node_color="#2196F3", node_size=1200, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax, font_color="white")

def plot_image_overlay(ax, img, target_id, top_context_ids, grid_size=14):
    """Plots image with structured bounding boxes and highlighting."""
    ax.imshow(img.resize((224, 224)))
    
    # Target Box
    tx, ty, tw, th = get_patch_rect(target_id, grid_size)
    ax.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=4, edgecolor='#F44336', facecolor='none', label='Target'))
    
    # Context Boxes
    for i, pid in enumerate(top_context_ids):
        cx, cy, cw, ch = get_patch_rect(pid, grid_size)
        is_strongest = (i == 0)
        color = "#FFD700" if is_strongest else "#2196F3"
        alpha = 1.0 if is_strongest else 0.7
        lw = 4 if is_strongest else 2
        
        ax.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha))
        # Optional rank label on image
        ax.text(cx, cy, f"#{i+1}", color="white", fontsize=8, weight='bold', bbox=dict(facecolor=color, alpha=0.5, pad=0))

    ax.axis('off')
    ax.set_title(f"Image Patch Analysis (Target {target_id})", fontsize=12)

def add_legend(fig):
    """Adds a clear research legend to the figure."""
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Patch', markerfacecolor='#F44336', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Context Patch', markerfacecolor='#2196F3', markersize=12),
        Line2D([0], [0], color='#FFD700', lw=4, label='Strongest Attribution (#1)'),
        Line2D([0], [0], color='#2196F3', lw=2, label='Contributing Edges'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True, shadow=True)

def visualize_research_ijepa(target_ids=None, k=6, layout_mode="importance"):
    """Research-grade attribution visualization suite."""
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    model = IJEPAModel(img_size=224, patch_size=16, embed_dim=192)
    model.eval()
    
    num_patches = 196
    if target_ids is None:
        _, target_ids = get_ijepa_masks(num_patches=num_patches, num_context=50, num_target=1)
    
    num_targets = len(target_ids)
    fig = plt.figure(figsize=(16, 8 * num_targets))
    
    for idx, target_id in enumerate(target_ids):
        # 1. Run Analysis
        context_ids, _ = get_ijepa_masks(num_patches=num_patches, num_context=80) 
        if target_id in context_ids: context_ids.remove(target_id)
        
        with torch.no_grad():
            model.predict(img_tensor, context_ids, [target_id])
            attn = model.predictor.get_last_self_attention()
            target_to_context = attn[0].mean(0)[-1, :len(context_ids)].cpu().numpy()

        sorted_indices = np.argsort(target_to_context)[::-1][:k]
        top_context_ids = [context_ids[i] for i in sorted_indices]
        top_weights = [target_to_context[i] for i in sorted_indices]

        # 2. Build Graph
        G = nx.DiGraph()
        target_node = f"target_{target_id}"
        for pid, w in zip(top_context_ids, top_weights):
            G.add_edge(f"patch_{pid}", target_node, weight=float(w))

        # 3. Subplots
        gs = fig.add_gridspec(num_targets, 2, width_ratios=[1.2, 1])
        ax_graph = fig.add_subplot(gs[idx, 0])
        ax_img = fig.add_subplot(gs[idx, 1])
        
        pos = compute_structured_layout(G, target_node, mode=layout_mode)
        draw_attribution_viz(ax_graph, G, pos, target_node)
        
        ax_graph.set_title(f"I-JEPA Attribution Structure ({layout_mode.capitalize()} Layout)\nTarget: {target_id}", fontsize=14, loc='left')
        ax_graph.axis('off')
        
        plot_image_overlay(ax_img, raw_img, target_id, top_context_ids)

    add_legend(fig)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

if __name__ == "__main__":
    # Example: Visualize two targets using the research-grade spatial layout
    visualize_research_ijepa(target_ids=[42, 114], k=6, layout_mode="importance")
