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
        
        config = WorldModelConfig(backend="ijepa", d_embed=192, n_layers=6, n_heads=3, predictor_embed_dim=384, predictor_depth=4, predictor_heads=6)
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
        
        self.hook_fire_count = 0

    def run_eval(self, expect_hook=False):
        """Verified run that ensures hooks fire by using a side-effect counter."""
        self.hook_fire_count = 0
        
        with torch.no_grad():
            # Mandatory sync for internal component routing
            if hasattr(self.wm.adapter, "hooks"):
                self.wm.adapter.hooks = self.wm.hook_registry
            self.wm.adapter.current_timestep = 0
            
            traj, cache = self.wm.run_with_cache(self.img_tensor)
            
            # Extract from cache to ensure we use post-hook values
            pred = cache.get("z_prior", 0)
            assert pred is not None, "Error: 'z_prior' (prediction) missing from cache."
            
            # PROOF OF LIFE: Ensure at least one hook fired if we expected one
            if expect_hook:
                assert self.hook_fire_count > 0, (
                    "CRITICAL ERROR: No hooks fired during intervention! "
                    "The forward pass is bypassing the HookRegistry. Results are invalid."
                )
            
            target_reps = cache.get("target_encoding", 0)
            if target_reps is not None:
                # Dimension-agnostic indexing [B, N, C] or [N, C]
                target_gt = target_reps[0, [self.target_id], :] if target_reps.dim() == 3 else target_reps[[self.target_id], :]
            else:
                target_gt = self.adapter.target_encode(self.img_tensor)[:, [self.target_id], :]
            
            mse = F.mse_loss(pred.squeeze(0), target_gt.squeeze(0)).item()
        return mse, target_gt

    def run_layer_ablation(self):
        print(f"Phase 3.1: Layer-by-layer causal tracing for Target {self.target_id}...")
        
        self.wm.clear_hooks()
        clean_mse, _ = self.run_eval(expect_hook=False)
        
        predictor_depth = len(self.adapter.predictor.blocks)
        results = []

        def make_layer_hook():
            def ablate_hook(x, ctx):
                self.hook_fire_count += 1
                return torch.zeros_like(x)
            return ablate_hook

        for layer_idx in range(predictor_depth):
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=f"predictor.block_{layer_idx}", fn=make_layer_hook()))
            
            corrupted_mse, _ = self.run_eval(expect_hook=True)
            importance = (corrupted_mse - clean_mse) / (clean_mse + 1e-8)
            results.append(importance)
            print(f"  Layer {layer_idx} Importance: {importance:.4f} [Verified: {self.hook_fire_count} fire(s)]")

        return results, clean_mse

    def run_head_zoom(self, layer_idx):
        print(f"Phase 3.2: Zooming into Layer {layer_idx} (Greedy Head Ablation)...")
        
        self.wm.clear_hooks()
        clean_mse, _ = self.run_eval(expect_hook=False)
        
        num_heads = self.adapter.predictor.blocks[layer_idx].attn.num_heads
        head_results = []

        def make_head_hook(h):
            def ablate_head_hook(out_heads, ctx):
                # SHAPE SAFETY: Verify we are intercepting the pre-projection split tensor
                assert out_heads.dim() == 4, (
                    f"Structural Error: Expected [B, heads, N, head_dim], got {out_heads.shape}. "
                    "Hook is intercepting the wrong tensor."
                )
                self.hook_fire_count += 1
                out_heads = out_heads.clone()
                out_heads[:, h, :, :] = 0
                return out_heads
            return ablate_head_hook

        for head_idx in range(num_heads):
            self.wm.clear_hooks()
            self.wm.add_hook(HookPoint(name=f"predictor.block_{layer_idx}.attn.heads", fn=make_head_hook(head_idx)))
            
            corrupted_mse, _ = self.run_eval(expect_hook=True)
            importance = (corrupted_mse - clean_mse) / (clean_mse + 1e-8)
            head_results.append(importance)
            print(f"    Head {head_idx} Importance: {importance:.4f} [Verified: {self.hook_fire_count} fire(s)]")

        return head_results

    def visualize(self, layer_scores, head_scores_map):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        layers = range(len(layer_scores))
        ax1.bar(layers, layer_scores, color='#2196F3', alpha=0.8)
        ax1.set_xlabel("Predictor Block Index")
        ax1.set_ylabel("Causal Importance")
        ax1.set_title("Layer-by-Layer Zoom")
        ax1.grid(True, axis='y', alpha=0.3)

        ax2.clear()
        colors = ['#4CAF50', '#8BC34A', '#CDDC39']
        for i, (l_idx, scores) in enumerate(head_scores_map.items()):
            h_indices = range(len(scores))
            ax2.bar(h_indices, scores, color=colors[i % len(colors)], alpha=0.8, label=f"Layer {l_idx}")
            
        ax2.set_xlabel("Head Index")
        ax2.set_title("Target-Specific Head Discovery")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"I-JEPA Verified Structural Circuits | Target {self.target_id}", fontsize=14)
        plt.tight_layout()
        
        if os.environ.get("SAVE_PLOT"):
            plt.savefig("structural_circuits_verified.png")
            print("Verified structural circuits saved to structural_circuits_verified.png")
        else:
            plt.show()

if __name__ == "__main__":
    tracer = IJEPAStructuralTracer(target_id=114)
    layer_scores, _ = tracer.run_layer_ablation()
    critical_layer = np.argmax(layer_scores)
    head_scores = tracer.run_head_zoom(critical_layer)
    tracer.visualize(layer_scores, {critical_layer: head_scores})
