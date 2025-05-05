import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import torchvision
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from face_models.model import QualityFaceNet, QualityAdaFace
torch.manual_seed(42)
np.random.seed(42)

# change net
# model = QualityFaceNet()
model = QualityAdaFace()
# change checpoint
checkpoint = torch.load('method/new_SDD/checkpoints/adaface/checkpoint_epoch_20.pth')
# change image folder
image_folder = Path("mgr_data/IJBC_cut")
# image_folder = Path("mgr_data/LFW")

model.load_state_dict(checkpoint)
model.eval()

output_csv = "method/new_SDD/scores/test/adaface_IJBC.csv"

results = []

for img_path in tqdm(image_folder.rglob("*.*")): 
    try:
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = torchvision.io.read_image(img_path).float().cuda()
            image = image.unsqueeze(0)
            # print(image.shape)
            
            with torch.no_grad():
                output = model(image)
                output_score = output.item()
        
            results.append({"img_path": img_path, "output_score": output_score})
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Step 5: Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)