from pathlib import Path
from PIL import Image
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Specify the folder containing subfolders with photos
photos_root_folder = Path("mgr_data/IJBC")  # Update with the root folder path

# Target size for resizing
target_size = (112, 112)

# Function to resize images
def resize_images_in_folder(folder_path: Path, target_size: tuple):
    for image_path in folder_path.rglob("*"):  # Recursively find all files in subfolders
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            try:
                with Image.open(image_path) as img:
                    # Resize and save the image
                    img_resized = img.resize(target_size)
                    img_resized.save(image_path)
                    print(f"Resized: {image_path}")
            except Exception as e:
                print(f"Error resizing {image_path}: {e}")

# Call the function
resize_images_in_folder(photos_root_folder, target_size)
