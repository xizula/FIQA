import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

from dataset.dataset import get_dataset
from face_models.model import load_model
import yaml
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle
import torch

np.random.seed(42)

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

data_loader = get_dataset()
model_names = config['model']

data_name = config['data']['path']
data_name = data_name.split('/')[-1]
print(data_name)

for model_name in model_names:
    embeddings = []
    paths = []
    labels = []
    wrong_paths =[]
    print(model_name)
    model = load_model(model_name)
    for image, label, path in tqdm(data_loader):
        try:
            embedding = model(image.cuda())
            embedding = embedding.squeeze()
            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)
            # print(embedding.shape)
            embeddings.append(embedding)
            paths.extend(path)
            labels.extend(list(label.numpy()))
        except Exception as e:
            wrong_paths.extend(path)
            print(f"Error: {e}")
    

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.vstack(embeddings)
    embeddings = embeddings.tolist()
    df = {'embedding': embeddings, 'label': labels, 'path': paths}
    embeddings_df = pd.DataFrame(df)

    embeddings_df.to_csv(f'mgr_data/embeddings/{model_name}_{data_name}_embeddings.csv', index=False)
    print(f"Finished calculating for {model_name}")
