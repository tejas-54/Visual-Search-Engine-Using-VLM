import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from tqdm import tqdm
import gc
from torchvision import transforms

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 15
CAPTION_CSV = "/home/shivaprasad/shiva/image_captions.csv"
MODEL_NAME = "openai/clip-vit-base-patch16"
OUTPUT_MODEL = "clip_vit_b16_finetuned.pth"

# Custom Dataset with Augmentation
class CLIPDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Loading captions from {csv_path}")
        self.data = pd.read_csv(csv_path)
        print(f"Found {len(self.data)} caption entries")
       
        print(f"Loading CLIP processor from {MODEL_NAME}")
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
       
        # Advanced data augmentation pipeline
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
        ])
       
        # Verify all image paths exist
        valid_rows = []
        for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Validating images"):
            if os.path.isfile(row["image_path"]):
                valid_rows.append(i)
            else:
                print(f"Warning: Image not found: {row['image_path']}")
       
        self.data = self.data.iloc[valid_rows].reset_index(drop=True)
        print(f"Dataset contains {len(self.data)} valid images with captions")
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
           
            # Apply augmentation
            image = self.augmentation(image)
           
            # Process image and text
            inputs = self.processor(
                text=[item["caption"]],
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
           
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
               
            return inputs
        except Exception as e:
            print(f"Error processing item {idx}, image: {item['image_path']}: {e}")
            return self[min(idx + 1, len(self.data) - 1)]

def train_clip():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
   
    if device == "cuda":
        torch.cuda.empty_cache()
   
    # Initialize CLIP
    print(f"Loading CLIP model {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
   
    # Freeze the vision encoder except for the LayerNorm layers
    for name, param in model.named_parameters():
        if 'vision_model' in name:
            if 'layernorm' not in name.lower():
                param.requires_grad = False
   
    # Make sure text encoder is trainable
    for name, param in model.text_model.named_parameters():
        param.requires_grad = True
   
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
   
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
        eps=1e-7
    )
   
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-7
    )
   
    # Dataset and Dataloader
    dataset = CLIPDataset(CAPTION_CSV)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
   
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler()
   
    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_loss = float('inf')
   
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
       
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
       
        for i, batch in enumerate(progress_bar):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)
           
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(**batch)
                logits_per_image = outputs.logits_per_image
               
                # Contrastive loss
                targets = torch.arange(len(batch["pixel_values"])).to(device)
                loss = torch.nn.functional.cross_entropy(logits_per_image, targets)
               
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
           
            # Backward pass
            scaler.scale(loss).backward()
           
            # Update weights every GRADIENT_ACCUMULATION_STEPS batches
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
                # Clip gradients to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
               
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
           
            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_loss += current_loss
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.8f}"
            })
           
            del outputs, loss
            if i % 10 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
       
        # Update learning rate at the end of each epoch
        scheduler.step()
       
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.8f}")
       
        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = "clip_best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with loss: {best_loss:.4f}")
       
        # Regular checkpoint
        checkpoint_path = f"clip_checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
   
    # Save final model
    torch.save(model.state_dict(), OUTPUT_MODEL)
    torch.save(model, "clip_vit_b16_finetuned_full.pth")
    print(f"Fine-tuning complete! Models saved to {OUTPUT_MODEL} and clip_vit_b16_finetuned_full.pth")

if __name__ == "__main__":
    # Verify required packages
    try:
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("Torchvision not found. Please install it with: pip install torchvision")
        exit(1)
       
    torch.set_num_threads(2)
    train_clip()

