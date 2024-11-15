import os
import shutil
import math
from tqdm import tqdm

# Define the base directory and the new folder paths
base_folder = 'mgr_data/MS1M'
new_folder_1 = 'mgr_data/R_MS1M'
new_folder_2 = 'mgr_data/L_MS1M'

# Create new folders if they don't exist
os.makedirs(new_folder_1, exist_ok=True)
os.makedirs(new_folder_2, exist_ok=True)

# Iterate through the folders in the base folder
for folder_name in tqdm(os.listdir(base_folder)):
    folder_path = os.path.join(base_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # Get the list of image files in the folder
        images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        total_images = len(images)
        
        # Calculate the number of images for each folder
        split_index = math.ceil(total_images * 2 / 3)  # First 2/3
        
        # Create corresponding folders in the new folders
        os.makedirs(os.path.join(new_folder_1, folder_name), exist_ok=True)
        os.makedirs(os.path.join(new_folder_2, folder_name), exist_ok=True)
        
        # Move the first 2/3 of images to Newfolder_1
        for image in images[:split_index]:
            shutil.copy2(os.path.join(folder_path, image), os.path.join(new_folder_1, folder_name, image))
        
        # Copy the remaining 1/3 of images to Newfolder_2
        for image in images[split_index:]:
            shutil.copy2(os.path.join(folder_path, image), os.path.join(new_folder_2, folder_name, image))

print("Images have been split and moved successfully!")
