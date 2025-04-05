import os
import torch
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import glob
import gc
import json

# Fix OpenMP runtime initialization issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Memory optimization for GPU with limited VRAM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# Reduced batch size for 4GB VRAM
BATCH_SIZE = 16

# Path to your image dataset and output directory
IMAGE_DIR = r"ENTER-YOUR-IMAGE-DIR"
OUTPUT_DIR = "index_files"

# Path to your fine-tuned model
MODEL_PATH = r"ENTER-YOUR-FINETUNED-MODEL-PATH"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    # Load base CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Load fine-tuned weights if they exist
    if os.path.exists(MODEL_PATH):
        print(f"Loading fine-tuned CLIP model from {MODEL_PATH}...")
        # Use weights_only=True to address security warning
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, processor, device

def get_image_paths():
    # Get all image paths (adjust extensions as needed)
    print(f"Finding images in {IMAGE_DIR}...")
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, "**", ext), recursive=True))
    return image_paths

def process_batch(model, processor, image_paths, device):
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            # Open and convert image
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if not images:
        return [], []
        
    # Process images with CLIP
    with torch.no_grad():
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # Move back to CPU to free up GPU memory
            embeddings = image_features.cpu().numpy()
            
            # Explicit cleanup to free GPU memory
            del inputs, image_features
            torch.cuda.empty_cache()
            
            return embeddings, valid_paths
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM error: {e}. Try reducing batch size further.")
                # Try with half the batch size as a fallback
                half_size = len(images) // 2
                if half_size == 0:
                    return [], []
                    
                print(f"Retrying with batch size {half_size}")
                emb1, paths1 = process_batch(model, processor, image_paths[:half_size], device)
                emb2, paths2 = process_batch(model, processor, image_paths[half_size:], device)
                
                if len(emb1) > 0 and len(emb2) > 0:
                    return np.vstack([emb1, emb2]), paths1 + paths2
                elif len(emb1) > 0:
                    return emb1, paths1
                elif len(emb2) > 0:
                    return emb2, paths2
                else:
                    return [], []
            else:
                print(f"Error: {e}")
                return [], []

def build_index():
    print("Loading model...")
    model, processor, device = load_model()
    
    print("Getting image paths...")
    image_paths = get_image_paths()
    print(f"Found {len(image_paths)} images")
    
    all_embeddings = []
    all_valid_paths = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Indexing images"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        embeddings, valid_paths = process_batch(model, processor, batch_paths, device)
        
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_valid_paths.extend(valid_paths)
        
        # Periodically clear CUDA cache to prevent memory fragmentation
        if i % (BATCH_SIZE * 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Stack all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings with shape: {all_embeddings.shape}")
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        index.add(all_embeddings.astype(np.float32))
        
        # Save index and metadata
        print("Saving index and metadata...")
        faiss.write_index(index, os.path.join(OUTPUT_DIR, "image_index.faiss"))
        np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), all_embeddings)
        
        # Save image paths as JSON
        with open(os.path.join(OUTPUT_DIR, "image_paths.json"), "w") as f:
            json.dump(all_valid_paths, f)
                
        print(f"Index built successfully with {len(all_valid_paths)} images.")
    else:
        print("No valid embeddings were created. Check your images and paths.")

if __name__ == "__main__":
    build_index()
