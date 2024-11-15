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
# Sample list
file_path = config['data']['test_embeddings']
model_name = config['model']['embeddings']

model = load_model(model_name)
df = pd.read_csv(file_path)
labels = []
scores = []

n_rows = len(df)
# Iterate through each pair of rows without repetition
# for i in tqdm(range(n_rows)):
#     for j in range(i + 1, n_rows):
#         row_i = df.iloc[i]
#         row_j = df.iloc[j]
#         label = 1 if row_i['label'] == row_j['label'] else 0
#         # print(label)
#         e_i = np.array(ast.literal_eval(row_i['embedding']))
#         e_j = np.array(ast.literal_eval(row_j['embedding']))
        
#         score = model.compute_similarities(e_i, e_j)
#         # print(score)
#         labels.append(label)
#         scores.append(score)

del df
data = {'label': labels, 'score': scores}
# data = data.detach().cpu().numpy()
data = pd.DataFrame(data)
data.to_csv(f'mgr_data/reco_scores/{model_name}_base_scores.csv', index=False)