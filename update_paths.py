import json
import os

# Load existing paths
with open('search_index/image_paths.json', 'r') as f:
    paths = json.load(f)

# Update paths to reference files within the repository
new_paths = []
for path in paths:
    # Extract just the filename from the old path
    filename = os.path.basename(path)
    # Create new path relative to the repository root
    new_path = os.path.join('images', filename)
    new_paths.append(new_path)

# Save updated paths
with open('search_index/image_paths.json', 'w') as f:
    json.dump(new_paths, f)

print(f"Updated {len(new_paths)} image paths")

