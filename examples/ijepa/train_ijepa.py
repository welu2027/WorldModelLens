import torch
import torch.nn.functional as F
import torch.optim as optim
from ijepa_model import IJEPAModel
from image_utils import get_sample_image, preprocess_image, get_ijepa_masks
import time
import os

def train_ijepa_mini(epochs=50, steps_per_epoch=4, lr=1e-4):
    print("Starting I-JEPA Mini-Training...")
    model = IJEPAModel()
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # We'll use a few "scenes" for training
    # For this prototype, we'll use the same image with different masks 
    # and potentially another sample if available.
    raw_img = get_sample_image()
    img_tensor = preprocess_image(raw_img)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            # Generate random masks for training
            # context ~50-80 patches, target ~10-20 patches
            context_ids, target_ids = get_ijepa_masks(num_context=70, num_target=10)
            
            optimizer.zero_grad()
            
            # Forward pass: Predict target latents from context
            pred_latents, target_gt = model.predict(img_tensor, context_ids, target_ids)
            
            # Loss: MSE in latent space
            loss = F.mse_loss(pred_latents, target_gt)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / steps_per_epoch
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Elapsed: {time.time()-start_time:.1f}s")

    # Save weights
    checkpoint_path = "ijepa_mini.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining Complete. Model saved to {checkpoint_path}")
    
    return losses

if __name__ == "__main__":
    train_ijepa_mini(epochs=100)
