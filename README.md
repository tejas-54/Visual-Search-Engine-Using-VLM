# Visual-Search-Engine-Using-VLM
This project implements a visual search engine leveraging state-of-the-art vision-language models (VLMs) to retrieve relevant images based on textual queries or sample images. The system embeds both text and images into a shared representation space, allowing for semantic search across modalities.

# Uses in the Apparel Domain  
This visual search engine enables apparel shoppers to find clothing items by submitting either text descriptions ("red floral maxi dress") or uploading reference images of desired styles. The system analyzes the query using vision-language models to understand visual and textual features, then retrieves the most visually similar items from the product catalog. This approach eliminates the frustration of text-only searches that miss visual nuances, reduces search time by showing highly relevant results, and helps customers discover products that precisely match their style preferences even when they struggle to describe fashion details in words.  

## Features:  
__Multi-Modal Search:__ Query using either text descriptions or example images  
__Semantic Understanding:__ Find images based on conceptual meaning, not just keywords  
__Efficient Retrieval:__ Fast similarity search using approximate nearest neighbors  
__Intuitive Web Interface:__ Simple UI for uploading images or entering text queries    


## Setup:  
Clone this repository:  
```
git clone https://github.com/tejas-54/Visual-Search-Engine-Using-VLM.git
cd visual-search-vlm
```
Install dependencies::  
```
pip install -r requirements.txt
```
### IMPORTANT: 
Change file paths in the configuration sections of each script to match your system setup.  

# File Structure:  
**image/** - Directory for storing your image dataset  

**clip_vit_b16_finetuned.pth** - Fine-tuned CLIP ViT-B/16 model checkpoint  

**generate_image_descriptions.py** - Script to generate text descriptions for images  

**newindex.py** - Creates embeddings and builds the search index  

**visual_search.py** - Core search functionality implementation  

**webapp.py** - Web application for the search interface  

# Usage:  
**1. Run the requirements.txt**  
```
pip install -r requirements.txt
```
**2. Generate Image Descriptions**  
Generate descriptive captions for images in your dataset:  
```
python generate_image_descriptions.py
```
**Note:** Edit the input and output paths in this file before running.  

**3. Run the finetuning code**  
Generate a finetuned state-of-art-clip-model:  
```
python finetune_vlm.py
```
**Note:** Change the required paths in this file.  

**4. Run the udate path file**  
Update the paths of the images file in the fine tuned model:  
```
python update_paths.py
```
**Note:** Change the required paths in this file  

**5. Create Search Index**
Build the search index from your image dataset:
```
python newindex.py
```
**Note:** Change the image directory path and index output location in this file.  

**6. Create Visual Search for the model**  
Run the visual_search.py to create visual_search:  
```
python visual_search.py
```
**Note:** Edit the index and model paths in this file before running.    

**7. Run the Web Application**  
Start the web interface:  
```
python webapp.py
```
**Note:** Modify the model path and index location in this file before running.    

**Perform Visual Search**   
Navigate to **http://localhost:5000** in your browser   

Enter a text query or upload a sample image  

View search results sorted by relevance  

**Example Queries**  
**Text queries:** "red floral dress", "white nike shoes", "brown pants for men" etc  

**Image queries:** Upload any apparel image to find visually similar ones  

**Performance Evaluation**  
The system evaluates search performance using:  

Precision@k  

Recall@k  

Mean Average Precision (MAP)  

**Technical Details**  
This implementation uses:  

CLIP (Contrastive Language-Image Pre-training) for the shared embedding space  

FAISS for efficient similarity search and indexing  

Flask for the web interface  

 




