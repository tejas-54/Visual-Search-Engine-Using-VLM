import os
import torch
import numpy as np
from PIL import Image
import faiss
import json
from transformers import CLIPProcessor, CLIPModel

class VisualSearchEngine:
    def __init__(self, index_folder, model_path=None):
        self.index_folder = r"ADD-YOUR-INDEX-FOLDER-PATH"
        self.model_path = r"ADD-YOUR-FINETUNED-MODEL-PATH"
        
        # Paths to index files
        self.index_path = os.path.join(index_folder, "image_index.faiss")
        self.embeddings_path = os.path.join(index_folder, "embeddings.npy")
        self.paths_path = os.path.join(index_folder, "image_paths.json")
        
        # Check if index files exist
        if not os.path.exists(self.index_path) or not os.path.exists(self.paths_path):
            raise FileNotFoundError(f"Index files not found in {index_folder}. Run build_index.py first.")
        
        # Load model
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load base CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Load fine-tuned weights if provided
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading fine-tuned model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load index and image paths
        print("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)
        
        with open(self.paths_path, 'r') as f:
            self.image_paths = json.load(f)
        
        print(f"Visual search engine initialized with {len(self.image_paths)} images")
    
    def encode_image(self, image_path):
        """Encode an image to a feature vector using CLIP"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
            return image_features.cpu().numpy()
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")
    
    def encode_text(self, text):
        """Encode text to a feature vector using CLIP"""
        try:
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                # Normalize features
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
            return text_features.cpu().numpy()
        except Exception as e:
            raise Exception(f"Error encoding text: {str(e)}")
    
    def search_by_image(self, image_path, k=5):
        """Search for similar images given an image path"""
        try:
            # Encode query image
            query_features = self.encode_image(image_path)
            
            # Search in the index
            distances, indices = self.index.search(query_features, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = distances[0][i]
                
                if idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(score)
                    })
            
            return results
        except Exception as e:
            raise Exception(f"Error in image search: {str(e)}")
    
    def search_by_text(self, text, k=5):
        """Search for images that match the text description"""
        try:
            # Encode query text
            query_features = self.encode_text(text)
            
            # Search in the index
            distances, indices = self.index.search(query_features, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = distances[0][i]
                
                if idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(score)
                    })
            
            return results
        except Exception as e:
            raise Exception(f"Error in text search: {str(e)}")
