import os
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
print(parent_dir)
# Specify the paths
txt_file = "mgr_data/ijbc_face_tid_mid.txt"  # Update with the path to your txt file
photos_folder = "mgr_data/loose_crop"  # Update with the folder containing photos
destination_folder = "mgr_data/IJBC"  # Update with
# Read the txt file and process it
with open(txt_file, "r") as file:
    for line in file:
        parts = line.strip().split()  # Split the line into components
        if len(parts) < 2:
            continue  # Skip malformed lines
        
        image_name = parts[0]
        folder_name = parts[1]
        
        # Create the folder if it doesn't exist
        folder_path = os.path.join(destination_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Move the image to the corresponding folder
        image_path = os.path.join(photos_folder, image_name)
        if os.path.exists(image_path):  # Check if the image exists
            shutil.move(image_path, folder_path)
        else:
            print(f"Warning: {image_name} not found.")
