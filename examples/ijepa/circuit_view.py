import networkx as nx
import matplotlib.pyplot as plt
from dummy_data import generate_ijepa_data

def visualize_circuit(
    circuit_patches: list = ["patch_1", "patch_4", "patch_7"],
    seed: int = 42
):
    """
    Shows a 'Circuit View' highlighting a specific subset of patches
    that represent a concept or predictive mechanism.
    """
    data = generate_ijepa_data(num_context_patches=25, seed=seed)
    target_patch = data["target_patch"]
    coords = data["patch_coords"]
    attributions = data["attributions"]
    
    # 1. Filter edges for the circuit
    circuit_edges = [(pid, target_patch) for pid in circuit_patches if pid in attributions]
    
    if not circuit_edges:
        print("No valid circuit patches found in data.")
        return

    # 2. Build Graph
    G = nx.DiGraph()
    G.add_edges_from(circuit_edges)
    
    # Add other nodes as background (low alpha)
    background_nodes = [pid for pid in data["context_patches"] if pid not in circuit_patches]
    
    # 3. Layout (Fixed Grid for circuit context)
    pos = coords
    
    plt.figure(figsize=(10, 8))
    
    # 4. Draw Background
    nx.draw_networkx_nodes(
        G.to_undirected(), 
        pos, 
        nodelist=background_nodes,
        node_color="#e0e0e0", 
        node_size=600, 
        alpha=0.3
    )
    
    # 5. Draw Circuit Nodes
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist=circuit_patches,
        node_color="#4CAF50", # Material Green
        node_size=1200, 
        edgecolors="white",
        linewidths=3
    )
    
    # 6. Draw Target Node
    nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist=[target_patch],
        node_color="#F44336", # Material Red
        node_size=1500, 
        edgecolors="white",
        linewidths=3
    )
    
    # 7. Draw Circuit Edges
    weights = [attributions[u] for u, v in G.edges()]
    norm_weights = [w * 10 + 2 for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, 
        width=norm_weights, 
        edge_color="#4CAF50",
        alpha=0.8,
        arrowsize=25,
        connectionstyle="arc3,rad=0.2"
    )
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title(f"I-JEPA Predictive Circuit View\nNodes: {', '.join(circuit_patches)}", fontsize=14, pad=20)
    plt.axis('off')
    
    # Add a custom legend or text box
    plt.text(0.05, 0.05, "Green: Predictive Circuit\nRed: Target Prediction", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: Visualizing a 'Top-3' circuit
    visualize_circuit(circuit_patches=["patch_3", "patch_8", "patch_15"])
