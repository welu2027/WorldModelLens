import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from world_model_lens import HookedWorldModel
from world_model_lens.backends.ijepa_adapter import IJEPAAdapter
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookContext, HookPoint
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import os

class IJEPAStructuralTracer:
    def __init__(self, target_id=114):
        self.target_id = target_id
        self.raw_img = get_sample_image()
        self.img_tensor = preprocess_image(self.raw_img)
        
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3)
        self.adapter = IJEPAAdapter(config)
        
        checkpoint_path = "ijepa_mini.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            self.adapter.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=False)
            
        self.wm = HookedWorldModel(self.adapter, config)
        self.wm.adapter.eval()
        
        # Determine context patches
        self.context_ids, _ = get_ijepa_masks(num_context=80)
        if target_id in self.context_ids: self.context_ids.remove(target_id)
        
        # Sync masks to adapter once
        self.adapter.last_context_ids = self.context_ids
        self.adapter.last_target_ids = [self.target_id]

    def run_eval(self):
        """Standard high-level run via HookedWorldModel to ensure hook injection."""
        with torch.no_grad():
            traj, cache = self.wm.run_with_cache(self.img_tensor)
            # Find the actual target prediction in the cache or traj
            # In I-JEPA, dynamics() returns the target_preds
            pred = traj.states[0].metadata.get("target_preds")
            if pred is None:
                # Fallback if metadata not set
                pred = self.adapter.dynamics(self.adapter.encode(self.img_tensor)[0])
            
            target_gt = cache["target_encoding", 0][[self.target_id], :]
            mse = F.mse_loss(pred.squeeze(0), target_gt).item()
        return mse, target_gt

    def run_layer_ablation(self):
        print(f"Phase 3.1: Layer-by-layer causal tracing for Target {self.target_id}...")
        clean_mse, target_gt = self.run_eval()
        predictor_depth = len(self.adapter.predictor.blocks)
        results = []

        for layer_idx in range(predictor_depth):
            def ablate_hook(x, ctx):
                return torch.zeros_like(x)

            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=f"predictor.block_{layer_idx}", fn=ablate_hook))
            
            corrupted_mse, _ = self.run_eval()
            importance = (corrupted_mse - clean_mse) / (clean_mse + 1e-8)
            results.append(importance)
            print(f"  Layer {layer_idx} Importance: {importance:.4f}")

        return results, clean_mse

    def run_head_zoom(self, layer_idx):
        print(f"Phase 3.2: Zooming into Layer {layer_idx} (Greedy Head Ablation)...")
        self.wm.clear_hooks() # VERY IMPORTANT: Clear any previous layer-ablation hooks
        clean_mse, _ = self.run_eval()
        num_heads = self.adapter.predictor.blocks[layer_idx].attn.num_heads
        head_results = []

        for head_idx in range(num_heads):
            def ablate_head_hook(out_heads, ctx):
                out_heads = out_heads.clone()
                out_heads[:, head_idx, :, :] = 0
                return out_heads

            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=f"predictor.block_{layer_idx}.attn.heads", fn=ablate_head_hook))
            
            corrupted_mse, _ = self.run_eval()
            importance = (corrupted_mse - clean_mse) / (clean_mse + 1e-8)
            head_results.append(importance)
            print(f"    Head {head_idx} Importance: {importance:.4f}")

        return head_results

    def visualize(self, layer_scores, head_scores_map):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Layer Importance
        layers = range(len(layer_scores))
        ax1.bar(layers, layer_scores, color='#2196F3', alpha=0.8)
        ax1.set_xlabel("Predictor Block Index")
        ax1.set_ylabel("Causal Importance (Relative MSE Increase)")
        ax1.set_title("Structural Causal Tracing (Layer-by-Layer)")
        ax1.grid(True, axis='y', alpha=0.3)

        # 2. Head Importance (Zoom)
        for layer_idx, head_scores in head_scores_map.items():
            heads = range(len(head_scores))
            ax2.bar(heads, head_scores, color='#4CAF50', alpha=0.8)
            
        ax2.set_xlabel("Head Index")
        ax2.set_ylabel("Causal Importance")
        ax2.set_title(f"Target-Specific Head Discovery (Layer {list(head_scores_map.keys())[0]})")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"I-JEPA Structural Circuit Discovery | Target {self.target_id}", fontsize=16)
        plt.tight_layout()
        
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("structural_circuits.png")
            print("Structural circuit analysis saved to structural_circuits.png")
        else:
            plt.show()

if __name__ == "__main__":
    tracer = IJEPAStructuralTracer(target_id=114)
    layer_scores, clean_mse = tracer.run_layer_ablation()
    
    # Identify the most critical layer
    critical_layer = np.argmax(layer_scores)
    
    # Zoom into the critical layer
    head_scores = tracer.run_head_zoom(critical_layer)
    
    tracer.visualize(layer_scores, {critical_layer: head_scores})
