import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(parent_dir)
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

import itertools
import pandas as pd
import yaml
import ast
from tqdm import tqdm
import torch
from face_models.model import load_model
import numpy as np

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

file_path = config['data']['test_embeddings']
model_name = config['model']['embeddings']

model = load_model(model_name)
df = pd.read_csv(file_path)

n_rows = len(df)
labels = df['label'].values
embeddings = df['embedding'].values
embeddings = np.array([np.array(ast.literal_eval(e)) for e in embeddings])
score = model.compute_similarities(embeddings, embeddings)

labels_matrix = np.equal(labels[:, None], labels)
np.fill_diagonal(labels_matrix, False)
upper_triangle_indices = np.triu_indices_from(score, k=1)
class_labels = labels_matrix[upper_triangle_indices]
scores = score[upper_triangle_indices]

df = pd.DataFrame({
    'label': class_labels.astype(int),
    'score': scores
})

df.to_csv(f'mgr_data/reco_scores/{model_name}_base_scores.csv', index=False)