from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import argparse

def delete_small_folders(root_folder, min_images=5):
    # Convert root folder to a Path object
    root_path = Path(root_folder)

    # Iterate through all subfolders in the root folder
    for subfolder in root_path.iterdir():
        # Check if the path is a directory
        if subfolder.is_dir():
            # List all image files in the subfolder
            images = list(subfolder.glob('*.*'))
            
            # Filter images based on common image extensions
            image_files = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']]
            
            # Delete the subfolder if it has fewer than min_images
            if len(image_files) < min_images:
                for file in subfolder.iterdir():
                    file.unlink()  # Delete files inside the folder
                subfolder.rmdir()  # Delete the empty folder
                print(f"Deleted folder: {subfolder}")

def crop_faces(folder_path):
    mtcnn = MTCNN(image_size=112, margin=20)
    root_path = Path(folder_path)

    # Iterate through all subfolders in the root folder
    for subfolder in root_path.iterdir():
        # Check if the path is a directory
        if subfolder.is_dir():
            # Process each image in the subfolder
            for image_path in subfolder.glob('*.*'):
                # Filter only common image files
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    img = Image.open(image_path)
                    img_cropped = mtcnn(img, save_path=[image_path])

# Replace 'your_folder_path' with the path to your folder

def to_bgr(folder_path):
    root_path = Path(folder_path)

    # Iterate through all subfolders in the root folder
    for subfolder in root_path.iterdir():
        # Check if the path is a directory
        if subfolder.is_dir():
            # Process each image in the subfolder
            for image_path in subfolder.glob('*.*'):
                # Filter only common image files
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    img = cv2.imread(str(image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(image_path), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to the image dataset to preprocess")
    parser.add_argument("-m", "--min_videos", help="minimum number of images to keep a folder", type=int, default=10)
    args = parser.parse_args()

    delete_small_folders(args.path, args.min_videos)
    # crop_faces(args.path)
    to_bgr(args.path)