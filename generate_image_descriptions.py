import os
import time
import pandas as pd
from PIL import Image
import google.generativeai as genai

# Function to get all image files from a directory
def get_image_files(directory):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
   
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
   
    return image_files

# Function to get image description using Gemini API
def get_image_description(image_path, api_key):
    try:
        genai.configure(api_key=api_key)
       
        # Initialize Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-pro-vision')
       
        # Load the image
        img = Image.open(image_path)
       
        # Get description from Gemini
        response = model.generate_content(["Describe what's in this image in detail", img])
       
        return response.text
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return f"Error: {str(e)}"

# Main function
def main():
    # MODIFY THESE PARAMETERS
    # =======================
    # Path to your images folder
    images_folder = "/path/to/your/images"
   
    # Your Gemini API key
    api_key = "YOUR_GEMINI_API_KEY_HERE"
   
    # Output CSV filename
    output_csv = "image_descriptions.csv"
    # =======================
   
    # Get all image files from the directory
    print(f"Scanning {images_folder} for images...")
    image_files = get_image_files(images_folder)
    print(f"Found {len(image_files)} images")
   
    # Prepare data for DataFrame
    data = []
   
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{len(image_files)}: {image_path}")
            description = get_image_description(image_path, api_key)
            data.append({'image_path': image_path, 'description': description})
           
            # Add delay to avoid rate limiting
            if i < len(image_files) - 1:
                time.sleep(0.5)
               
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            data.append({'image_path': image_path, 'description': f"Error: {str(e)}"})
   
    # Create DataFrame
    df = pd.DataFrame(data)
   
    # Save to CSV
    df.to_csv(output_csv, index=False)
   
    print(f"Successfully saved descriptions to {output_csv}")

if __name__ == "__main__":
    main()

