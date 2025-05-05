import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
import random
import ast
np.random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from face_models.model import AdaFace, FaceNet
from dataset.dataset import get_dataset
import torch
import torch.nn.functional as F
from itertools import combinations
from sklearn.metrics.pairwise import euclidean_distances


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def generate_stochastic_embeddings(model, image, m=100):
    model.model.eval()
    enable_dropout(model) 

    embeddings = []
    for _ in range(m):
        with torch.no_grad():
            emb = model(image.cuda())
            embeddings.append(emb.cpu())

    return torch.stack(embeddings).squeeze(1)


def calculate_serfiq_score(embeddings, m: int = 100, alpha: float = 130.0, r: float = 0.88):
    norm = F.normalize(embeddings, dim=1)

    eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(m, k=1)]

    score = 2*(1/(1+np.exp(np.mean(eucl_dist))))

    return 1 / (1+np.exp(-(alpha * (score - r))))


# def ser_fiq(model, image, m=100):
#     stochastic_embeddings = generate_stochastic_embeddings(model, image, m)
#     score = calculate_serfiq_score(stochastic_embeddings, m)
#     return score


dataset = get_dataset('mgr_data/IJBC_cut')
output_path = "method/SER_FIQ/scores/test/adaface_IJBC.csv"

model = AdaFace()
all_scores = []
for images, labels, paths in tqdm(dataset):
    for img, label, path in zip(images, labels, paths):
        img = img.unsqueeze(0).cuda()  # add batch dimension
        with torch.no_grad():
            emb_set = generate_stochastic_embeddings(model, img, m=100)
            score = calculate_serfiq_score(emb_set, m=100)

        all_scores.append({
            "img_path": path,
            "output_score": score
        })

scores = pd.DataFrame(all_scores)
scores.to_csv(output_path, index=False)
