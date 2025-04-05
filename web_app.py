import os
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory
import uuid
import shutil
from visual_search import VisualSearchEngine
from flask_cors import CORS
import gc
import torch
import numpy as np
import logging
from pathlib import Path

# Configure logging to avoid showing full paths
class PathFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.replace(os.path.expanduser('~'), '~')
        return True

# Set up logger
logger = logging.getLogger(__name__)
logger.addFilter(PathFilter())

# Create Flask app with explicit template folder
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

# Set OpenMP environment variable to avoid duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define folders for uploads and results
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize search engine
search_engine = None

# Metrics storage
search_metrics = {
    'queries': [],
    'precision': [],
    'recall': []
}

def calculate_precision_at_k(relevant_results, k):
    """Calculate precision@k metric"""
    if k == 0:
        return 0
    return len(relevant_results[:k]) / k

def calculate_recall_at_k(relevant_results, total_relevant, k):
    """Calculate recall@k metric"""
    if total_relevant == 0:
        return 0
    return len(relevant_results[:k]) / total_relevant

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests (text or image)"""
    global search_engine
   
    try:
        # Initialize search engine if not already done
        if search_engine is None:
            logger.info("Initializing search engine...")
            search_engine = VisualSearchEngine(
                index_folder="index_files",
                model_path="clip_vit_b16_finetuned.pth"
            )
            logger.info("Search engine initialized successfully")

        # Get query parameters
        query_type = request.form.get('query_type')
        k = int(request.form.get('k', 5))
       
        logger.info(f"Received {query_type} query for {k} results")
        results = []

        if query_type == 'text':
            # Process text query
            text_query = request.form.get('text_query', '')
            if not text_query:
                return jsonify({"error": "Text query is required"}), 400

            logger.info(f"Searching for text: {text_query}")
            results = search_engine.search_by_text(text_query, k)

        elif query_type == 'image':
            # Process image query
            if 'image_query' not in request.files:
                return jsonify({"error": "Image file is required"}), 400

            file = request.files['image_query']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            # Generate unique filename and save
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            logger.info(f"Searching for image: {filename}")  # Don't show full path
            results = search_engine.search_by_image(file_path, k)

        # Format results for display
        formatted_results = []
        logger.info(f"Found {len(results)} results")
       
        for i, result in enumerate(results):
            img_path = result["image_path"]
            # Extract just the filename to avoid exposing full path
            img_filename = Path(img_path).name
           
            # Create a copy in static folder for web access
            result_filename = f"result_{i}_{img_filename}"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)

            # Copy file to results folder
            try:
                shutil.copy2(img_path, result_path)
                logger.info(f"Processed result {i+1}")
            except Exception as e:
                logger.error(f"Error processing result {i+1}: {str(e)}")
                continue

            formatted_results.append({
                "image_url": f"/static/results/{result_filename}",
                "similarity": f"{result['similarity']:.4f}",
                "filename": img_filename  # Only show filename, not full path
            })

        # Calculate metrics if ground truth is provided
        if 'ground_truth' in request.form:
            try:
                ground_truth = request.form.get('ground_truth').split(',')
                # Determine relevant results (those in ground truth)
                relevant_results = [r for r in results if Path(r["image_path"]).name in ground_truth]
               
                # Calculate metrics
                precision_at_k = calculate_precision_at_k(relevant_results, k)
                recall_at_k = calculate_recall_at_k(relevant_results, len(ground_truth), k)
               
                # Store metrics
                query = text_query if query_type == 'text' else filename
                search_metrics['queries'].append(query)
                search_metrics['precision'].append(precision_at_k)
                search_metrics['recall'].append(recall_at_k)
               
                # Add metrics to response
                formatted_results = {
                    "results": formatted_results,
                    "metrics": {
                        "precision_at_k": precision_at_k,
                        "recall_at_k": recall_at_k
                    }
                }
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
       
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
           
        logger.info(f"Returning {len(formatted_results)} formatted results")
        return jsonify({"results": formatted_results})
       
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/metrics')
def get_metrics():
    """Return all stored metrics"""
    avg_precision = np.mean(search_metrics['precision']) if search_metrics['precision'] else 0
    avg_recall = np.mean(search_metrics['recall']) if search_metrics['recall'] else 0
   
    return jsonify({
        "overall": {
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "num_queries": len(search_metrics['queries'])
        },
        "per_query": [
            {
                "query": q,
                "precision": p,
                "recall": r
            }
            for q, p, r in zip(
                search_metrics['queries'],
                search_metrics['precision'],
                search_metrics['recall']
            )
        ]
    })

@app.route('/clear_metrics', methods=['POST'])
def clear_metrics():
    """Clear all stored metrics"""
    global search_metrics
    search_metrics = {
        'queries': [],
        'precision': [],
        'recall': []
    }
    return jsonify({"status": "success", "message": "Metrics cleared"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

