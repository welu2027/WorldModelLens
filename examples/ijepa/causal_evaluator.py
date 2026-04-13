import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from ijepa_model import IJEPAModel
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks

class IJEPAFaithfulnessEvaluator:
    def __init__(self, model: IJEPAModel, img_tensor: torch.Tensor):
        self.model = model
        self.img_tensor = img_tensor
        self.patch_size = 16
        self.num_patches = 196
        
        # Pre-compute ground truth latents and attention
        with torch.no_grad():
            self.full_latents = self.model.context_encoder(self.img_tensor)
            self.attn_matrix = self.model.context_encoder.blocks[-1].attn.last_attn_weights[0].mean(0).cpu().numpy()
            
            # Pre-compute patch embeddings before positional encoding
            self.patch_embeddings = self.model.context_encoder.patch_embed(self.img_tensor)
            # Global mean patch embedding for ablation
            self.mean_patch_emb = self.patch_embeddings.mean(dim=1, keepdim=True) # [1, 1, C]
            self.zero_patch_emb = torch.zeros_like(self.mean_patch_emb)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Computes MSE, L2, and Cosine Similarity."""
        mse = F.mse_loss(pred, target).item()
        l2 = torch.norm(pred - target).item()
        cos = F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).item()
        return {"mse": mse, "l2": l2, "cos": cos}

    def run_intervention(
        self, 
        target_id: int, 
        active_context_ids: List[int], 
        ablated_ids: List[int], 
        baseline="mean"
    ) -> Dict[str, float]:
        """Runs the model with a subset of context patches and a subset of ablated patches."""
        B = self.img_tensor.shape[0]
        emb_dim = self.patch_embeddings.shape[-1]
        
        # 1. Prepare intervened context embeddings
        # We start with the actual patch embeddings
        x = self.patch_embeddings.clone()
        
        # Apply ablation to specific IDs before adding positional embeddings and encoding
        baseline_emb = self.mean_patch_emb if baseline == "mean" else self.zero_patch_emb
        for pid in ablated_ids:
            x[:, pid, :] = baseline_emb
            
        # 2. Add encoder's positional info and extract context subset
        x = x + self.model.context_encoder.pos_embed
        
        context_ids = sorted(list(active_context_ids) + list(ablated_ids))
        context_latents_shallow = x[:, context_ids, :]
        
        # 3. Process through context encoder blocks (THE DEEP STEP)
        with torch.no_grad():
            context_latents = self.model.context_encoder.forward_blocks(context_latents_shallow)
            
            # 4. Predict
            # Add predictor's pos embed inside predictor.forward
            pred = self.model.predictor(context_latents, context_ids, [target_id])
            target_gt = self.full_latents[:, [target_id], :]
            
        return self.compute_metrics(pred, target_gt)

    def evaluate_faithfulness(self, target_id: int, context_ids: List[int], n_steps=20, n_random=5):
        """Runs ablation and reconstruction tests for a target."""
        # Get attributions from target's perspective (using predictor attn on DEEP latents)
        with torch.no_grad():
            # Get deep context latents for attribution extraction
            x_shallow = self.patch_embeddings + self.model.context_encoder.pos_embed
            ctx_latents_shallow = x_shallow[:, context_ids, :]
            ctx_latents = self.model.context_encoder.forward_blocks(ctx_latents_shallow)
            
            _ = self.model.predictor(ctx_latents, context_ids, [target_id])
            attn = self.model.predictor.get_last_self_attention()[0].mean(0)
            # Index of target in the combined [context + target] seq is the last one
            target_to_context_attn = attn[-1, :len(context_ids)].cpu().numpy()

        sorted_indices = np.argsort(target_to_context_attn)[::-1]
        top_context_ids = [context_ids[i] for i in sorted_indices]
        random_orders = [np.random.permutation(context_ids) for _ in range(n_random)]
        bottom_context_ids = top_context_ids[::-1]

        results = {
            "ablation": {"top": [], "random": [], "bottom": []},
            "reconstruction": {"top": [], "random": []}
        }

        # Ablation Test: Start with full context, remove patches
        for k in range(min(n_steps, len(context_ids))):
            # Top-K removal
            ablated = top_context_ids[:k]
            active = [pid for pid in context_ids if pid not in ablated]
            results["ablation"]["top"].append(self.run_intervention(target_id, active, ablated))
            
            # Bottom-K removal
            ablated_b = bottom_context_ids[:k]
            active_b = [pid for pid in context_ids if pid not in ablated_b]
            results["ablation"]["bottom"].append(self.run_intervention(target_id, active_b, ablated_b))
            
            # Random removal (avg over n_random)
            rand_metrics = []
            for order in random_orders:
                ablated_r = order[:k]
                active_r = [pid for pid in context_ids if pid not in ablated_r]
                rand_metrics.append(self.run_intervention(target_id, active_r, ablated_r))
            
            avg_rand = {m: np.mean([r[m] for r in rand_metrics]) for m in ["mse", "l2", "cos"]}
            results["ablation"]["random"].append(avg_rand)

        # Reconstruction Test: Start with empty context, add patches
        for k in range(1, min(n_steps, len(context_ids))):
            # Top-K addition
            active = top_context_ids[:k]
            results["reconstruction"]["top"].append(self.run_intervention(target_id, active, []))
            
            # Random addition
            rand_metrics = []
            for order in random_orders:
                active_r = order[:k]
                rand_metrics.append(self.run_intervention(target_id, active_r, []))
            
            avg_rand = {m: np.mean([r[m] for r in rand_metrics]) for m in ["mse", "l2", "cos"]}
            results["reconstruction"]["random"].append(avg_rand)

        return results

def calculate_auc(values):
    """Calculates Area Under Curve using trapezoidal rule."""
    return np.trapz(values) / len(values)

def compute_monotonicity(values):
    """Measures how strictly monotonic the sequence is (0 to 1)."""
    diffs = np.diff(values)
    # For MSE addition, we expect it to decrease. Check % of negative diffs.
    increasing = np.sum(diffs > 0) / len(diffs)
    decreasing = np.sum(diffs < 0) / len(diffs)
    return max(increasing, decreasing)

def find_predictive_circuit(metric_values, threshold_ratio=0.9):
    """Finds the step where metric reaches threshold_ratio of final performance."""
    final_val = metric_values[-1]
    best_val = min(metric_values) # For MSE/L2
    target_val = best_val + (1 - threshold_ratio) * (metric_values[0] - best_val)
    
    for i, v in enumerate(metric_values):
        if v <= target_val:
            return i + 1
    return len(metric_values)

def plot_faithfulness(results_list, metric="mse", target_ids=None):
    """Plots multi-target averaged faithfulness curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Average across all targets
    types = ["top", "random", "bottom"]
    
    # Ablation
    for t_type in types:
        all_vals = []
        for res in results_list:
            all_vals.append([x[metric] for x in res["ablation"][t_type]])
        
        avg_vals = np.mean(all_vals, axis=0)
        std_vals = np.std(all_vals, axis=0)
        
        color = "red" if t_type == "top" else "blue" if t_type == "bottom" else "gray"
        ls = "--" if t_type == "random" else "-"
        ax1.plot(avg_vals, label=f"{t_type.capitalize()}-K", color=color, linestyle=ls, lw=2)
        ax1.fill_between(range(len(avg_vals)), avg_vals - std_vals, avg_vals + std_vals, color=color, alpha=0.1)

    ax1.set_title(f"Ablation Suite ({metric.upper()})\nAverage over {len(results_list)} targets")
    ax1.set_xlabel("Patches Intervened")
    ax1.set_ylabel(metric.upper())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction
    for t_type in ["top", "random"]:
        all_vals = []
        for res in results_list:
            all_vals.append([x[metric] for x in res["reconstruction"][t_type]])
            
        avg_vals = np.mean(all_vals, axis=0)
        std_vals = np.std(all_vals, axis=0)
        
        color = "green" if t_type == "top" else "gray"
        ls = "--" if t_type == "random" else "-"
        ax2.plot(range(1, len(avg_vals)+1), avg_vals, label=f"{t_type.capitalize()}-K", color=color, linestyle=ls, lw=2)
        ax2.fill_between(range(1, len(avg_vals)+1), avg_vals - std_vals, avg_vals + std_vals, color=color, alpha=0.1)

    ax2.set_title(f"Reconstruction Suite ({metric.upper()})\nFaster decay = Stronger Causal Evidence")
    ax2.set_xlabel("Patches Provided")
    ax2.set_ylabel(metric.upper())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
    import os
    
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    model = IJEPAModel()
    model.eval()
    
    # Load trained weights if available
    checkpoint_path = "ijepa_mini.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    else:
        print("No trained weights found. Running with random initialization.")
    
    evaluator = IJEPAFaithfulnessEvaluator(model, img_tensor)
    
    # Multi-target validation
    context_ids, target_ids = get_ijepa_masks(num_context=80, num_target=5)
    
    print(f"Starting Multi-Target Causal Validation ({len(target_ids)} targets)...")
    all_results = []
    
    for tid in target_ids:
        print(f"  Validating Target {tid}...")
        res = evaluator.evaluate_faithfulness(tid, context_ids, n_steps=40)
        all_results.append(res)
        
        # Quantitative scoring for this target
        top_mse = [x['mse'] for x in res["reconstruction"]["top"]]
        rand_mse = [x['mse'] for x in res["reconstruction"]["random"]]
        
        auc_top = calculate_auc(top_mse)
        auc_rand = calculate_auc(rand_mse)
        mono = compute_monotonicity(top_mse)
        circuit_size = find_predictive_circuit(top_mse)
        
        print(f"    Target {tid} -> AUC Gap: {auc_rand - auc_top:.4f} | Monotonicity: {mono:.2%} | Circuit Size: {circuit_size} patches")

    plot_faithfulness(all_results, metric="mse", target_ids=target_ids)
