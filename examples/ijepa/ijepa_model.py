import torch
import torch.nn as nn
from vit import VisionTransformer, Block

class IJEPAPredictor(nn.Module):
    """
    Predictor transformer that maps context embeddings to target embeddings.
    """
    def __init__(self, embed_dim=192, depth=4, num_heads=3, num_patches=196):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Mask token to represent the target patches we want to predict
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Blocks for prediction
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Positional embedding for the full grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, context_latents, context_ids, target_ids):
        """
        Args:
            context_latents: [B, N_context, C]
            context_ids: [N_context] indices of visible patches
            target_ids: [N_target] indices of patches to predict
        """
        B = context_latents.shape[0]
        
        # 1. Prepare inputs: Context latents + Mask tokens for targets
        # Add positional info to context
        context_inputs = context_latents + self.pos_embed[:, context_ids, :]
        
        # Prepare target inputs (mask tokens + pos info)
        target_tokens = self.mask_token.expand(B, len(target_ids), -1)
        target_inputs = target_tokens + self.pos_embed[:, target_ids, :]
        
        # Combined sequence
        x = torch.cat([context_inputs, target_inputs], dim=1)
        
        # 2. Process through blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 3. Return only the predicted target latents
        return x[:, len(context_ids):, :]

    def get_last_self_attention(self):
        """Returns the attention weights from the last block."""
        return self.blocks[-1].attn.last_attn_weights

class IJEPAModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.context_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=6
        )
        
        self.predictor = IJEPAPredictor(
            embed_dim=embed_dim, depth=4, num_patches=num_patches
        )
        
    def predict(self, img, context_ids, target_ids):
        """
        Simulates the I-JEPA prediction flow.
        """
        # 1. Patchify and get full latents (simulating the target encoder for GT)
        with torch.no_grad():
            gt_latents = self.context_encoder(img)
            target_gt = gt_latents[:, target_ids, :]
            
        # 2. Get context latents (only visible patches)
        # In real I-JEPA, context encoder runs full depth on context patches.
        x = self.context_encoder.patch_embed(img)
        x = x + self.context_encoder.pos_embed
        context_latents_shallow = x[:, context_ids, :]
        
        # Run through context encoder blocks
        context_latents = self.context_encoder.forward_blocks(context_latents_shallow)
        
        # 3. Predict targets
        pred_latents = self.predictor(context_latents, context_ids, target_ids)
        
        return pred_latents, target_gt
