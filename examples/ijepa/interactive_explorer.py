import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, RadioButtons
import numpy as np
from PIL import Image
from ijepa_model import IJEPAModel
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks

class AttributionExplorer:
    def __init__(self, k=6, threshold=0.01, layout="radial"):
        print("Initializing Research-Grade Interactive Explorer...")
        self.k = k
        self.threshold = threshold
        self.layout_mode = layout
        self.target_id = 97 # Default target
        self.hovered_node = None
        
        # 1. Load Model & Data
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)
        self.model = IJEPAModel(img_size=224, patch_size=16, embed_dim=192)
        self.model.eval()
        
        # 2. Pre-compute All-to-All Attention for speed
        print("Pre-computing full context attention matrix...")
        self.num_patches = 196
        self.all_indices = list(range(self.num_patches))
        
        # Simulating a full-image context pass
        with torch.no_grad():
            # In this prototype, we simulate context via the full encoder
            gt_latents = self.model.context_encoder(self.img_tensor)
            # Full sequence is 196 patches. Predictor takes context + target.
            # We'll use all patches as potential context/target for the matrix.
            # Note: For simplicity in the interactive tool, we use the self-attention of the full image.
            # In a real I-JEPA use case, you'd slice specifically, but this matrix gives the "importance" flow.
            # We capture attention from the last encoder block for a global importance map.
            attn = self.model.context_encoder.blocks[-1].attn.last_attn_weights[0]
            self.attn_matrix = attn.mean(0).cpu().numpy() # [196, 196]

        # 3. Setup Figure
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1.2, 1])
        
        self.ax_graph = self.fig.add_subplot(self.gs[0, 0])
        self.ax_image = self.fig.add_subplot(self.gs[0, 1])
        self.ax_ctrl = self.fig.add_subplot(self.gs[1, :])
        self.ax_ctrl.axis('off')
        
        self.setup_widgets()
        self.setup_hover()
        self.update()

    def get_patch_rect(self, patch_id):
        row, col = patch_id // 14, patch_id % 14
        p_size = 224 // 14
        return col * p_size, row * p_size, p_size, p_size

    def setup_widgets(self):
        # Slider positions
        ax_target = self.fig.add_axes([0.15, 0.15, 0.3, 0.03])
        self.s_target = Slider(ax_target, 'Target Patch', 0, 195, valinit=self.target_id, valstep=1)
        
        ax_k = self.fig.add_axes([0.15, 0.1, 0.3, 0.03])
        self.s_k = Slider(ax_k, 'Top-K', 1, 25, valinit=self.k, valstep=1)
        
        ax_thresh = self.fig.add_axes([0.15, 0.05, 0.3, 0.03])
        self.s_thresh = Slider(ax_thresh, 'Threshold', 0.0, 0.1, valinit=self.threshold)
        
        ax_layout = self.fig.add_axes([0.6, 0.05, 0.1, 0.1])
        self.r_layout = RadioButtons(ax_layout, ('radial', 'spatial', 'bipartite'), active=0)
        
        self.s_target.on_changed(self.on_widget_change)
        self.s_k.on_changed(self.on_widget_change)
        self.s_thresh.on_changed(self.on_widget_change)
        self.r_layout.on_clicked(self.on_widget_change)

    def setup_hover(self):
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_widget_change(self, val):
        self.target_id = int(self.s_target.val)
        self.k = int(self.s_k.val)
        self.threshold = self.s_thresh.val
        self.layout_mode = self.r_layout.value_selected
        self.update()

    def update(self):
        self.ax_graph.clear()
        self.ax_image.clear()
        
        # 1. Fetch Attributions for selected target
        attributions = self.attn_matrix[self.target_id]
        indices = np.argsort(attributions)[::-1]
        
        # Filter (excluding self)
        top_indices = [i for i in indices if i != self.target_id and attributions[i] > self.threshold][:self.k]
        top_weights = [attributions[i] for i in top_indices]
        
        # 2. Build Subgraph
        self.G = nx.DiGraph()
        target_node = f"patch_{self.target_id}"
        for i, weight in zip(top_indices, top_weights):
            self.G.add_edge(f"patch_{i}", target_node, weight=float(weight))
            
        # 3. Compute Layout
        self.pos = self.compute_layout(target_node)
        
        # 4. Draw Graph
        self.draw_graph(target_node, top_indices, top_weights)
        
        # 5. Draw Image Overlay
        self.draw_image(top_indices)
        
        self.fig.canvas.draw_idle()

    def compute_layout(self, target_node):
        pos = {}
        context_nodes = [n for n in self.G.nodes() if n != target_node]
        if self.layout_mode == "radial":
            pos[target_node] = np.array([0, 0])
            for i, node in enumerate(context_nodes):
                w = self.G.get_edge_data(node, target_node)['weight']
                angle = np.pi/2 - (2 * np.pi * i / len(context_nodes))
                radius = 1.2 - (w * 10) # Fixed rad for prototype stability
                pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        elif self.layout_mode == "spatial":
            for node in self.G.nodes():
                pid = int(node.split("_")[1])
                pos[node] = np.array([pid % 14, 14 - (pid // 14)])
        else: # bipartite
            pos[target_node] = np.array([1, 0.5])
            for i, node in enumerate(context_nodes):
                pos[node] = np.array([0, i / max(1, len(context_nodes)-1)])
        return pos

    def draw_graph(self, target_node, top_indices, top_weights):
        # Edges
        edges = list(self.G.edges())
        if edges:
            weights = [self.G[u][v]['weight'] for u, v in edges]
            max_w = max(weights)
            for i, (u, v) in enumerate(edges):
                w = weights[i]
                is_hovered = (u == self.hovered_node)
                color = "#FFD700" if is_hovered else plt.cm.viridis(w / max_w)
                nx.draw_networkx_edges(self.G, self.pos, edgelist=[(u,v)], width=1+8*(w/max_w), 
                                       edge_color=[color], ax=self.ax_graph, alpha=0.9 if is_hovered else 0.6)

        # Nodes
        context_nodes = [n for n in self.G.nodes() if n != target_node]
        target_color = "#F44336"
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[target_node], node_color=target_color, node_size=1000, ax=self.ax_graph)
        
        for n in context_nodes:
            color = "#FFD700" if n == self.hovered_node else "#2196F3"
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[n], node_color=color, node_size=600, ax=self.ax_graph)
        
        nx.draw_networkx_labels(self.G, self.pos, font_size=7, font_weight="bold", ax=self.ax_graph, font_color="white")
        self.ax_graph.set_title(f"Attribution Topology ({self.layout_mode})")
        self.ax_graph.axis('off')

    def draw_image(self, top_indices):
        self.ax_image.imshow(self.raw_img.resize((224, 224)))
        # Target rect
        tx, ty, tw, th = self.get_patch_rect(self.target_id)
        self.ax_image.add_patch(patches.Rectangle((tx, ty), tw, th, linewidth=3, edgecolor='#F44336', facecolor='none'))
        
        # Top-K context rects
        for pid in top_indices:
            cx, cy, cw, ch = self.get_patch_rect(pid)
            is_hovered = (f"patch_{pid}" == self.hovered_node)
            color = "#FFD700" if is_hovered else "#2196F3"
            lw = 3 if is_hovered else 1.5
            alpha = 1.0 if is_hovered else 0.6
            self.ax_image.add_patch(patches.Rectangle((cx, cy), cw, ch, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha))
            
        self.ax_image.set_title("Spatial Grounding")
        self.ax_image.axis('off')

    def on_hover(self, event):
        if event.inaxes != self.ax_graph and event.inaxes != self.ax_image:
            return
            
        new_hover = None
        
        # 1. Check if hovering over graph node
        if event.inaxes == self.ax_graph:
            for node, (nx_x, nx_y) in self.pos.items():
                dist = np.sqrt((event.xdata - nx_x)**2 + (event.ydata - nx_y)**2)
                if dist < 0.15: # Node radius
                    new_hover = node
                    break
                    
        # 2. Check if hovering over image patch
        elif event.inaxes == self.ax_image:
            px, py = event.xdata, event.ydata
            grid_row, grid_col = int(py // (224/14)), int(px // (224/14))
            hover_pid = grid_row * 14 + grid_col
            # Only highlight if it's in our top-k graph
            if f"patch_{hover_pid}" in self.G.nodes():
                new_hover = f"patch_{hover_pid}"

        if new_hover != self.hovered_node:
            self.hovered_node = new_hover
            self.update()

if __name__ == "__main__":
    explorer = AttributionExplorer()
    plt.show()
