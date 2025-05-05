import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import wasserstein_distance
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

def cos(e1: np.ndarray, e2: np.ndarray) -> float:
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))


def get_class_centers(data: pd.DataFrame) -> pd.DataFrame:
    center = data.groupby('label')['embedding'].apply(lambda x: np.mean([ast.literal_eval(i) for i in x], axis=0)).reset_index()
    center = center.rename(columns={'embedding': 'class_center'})
    return center


def calculate_distribution(embedding: np.ndarray, samples: pd.DataFrame, eps: float = 1^(-9)) -> float:
    embedding = embedding.reshape(1, -1)
    cos_sim = cosine_similarity(embedding, samples)
    return cos_sim.squeeze()



def get_quality_score(data: pd.DataFrame, class_centers: pd.DataFrame, eps: float = 1^(-9)) -> pd.DataFrame:
    quality_data = {"path": [], "quality": []}

    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Calculating quality scores"):
        embedding = np.array(ast.literal_eval(row['embedding']))
        path = row['path']
        label = row['label']
        neg_label = class_centers[class_centers['label'] == label]['nearest_label'].values[0]

        pos_samples = np.array(data[(data['label'] == label) & (data['path'] != path)]['embedding'].apply(ast.literal_eval).to_list())
        neg_samples = np.array(data[data['label'] == neg_label]['embedding'].apply(ast.literal_eval).to_list())

        pos_dist = calculate_distribution(embedding, pos_samples)
        neg_dist = calculate_distribution(embedding, neg_samples)

        ws_distance = wasserstein_distance(pos_dist, neg_dist)
        quality_data["path"].append(path)
        quality_data["quality"].append(ws_distance)

    quality_df = pd.DataFrame(quality_data)
    return quality_df


def get_nearest_class_centers(class_centers: pd.DataFrame) -> pd.DataFrame:
    embeddings = np.array(class_centers['class_center'].to_list())
    labels = class_centers['label'].to_numpy()

    similarity_matrix = cosine_similarity(embeddings)

    np.fill_diagonal(similarity_matrix, -np.inf)

    nearest_indices = np.argmax(similarity_matrix, axis=1)
    nearest_labels = labels[nearest_indices]

    class_centers['nearest_label'] = nearest_labels
    return class_centers


if __name__ == '__main__':
    data = pd.read_csv('mgr_data/embeddings/facenet_CasiaWebFace_small_embeddings.csv')
    class_centers = get_class_centers(data)
    class_centers = get_nearest_class_centers(class_centers)
    quality = get_quality_score(data, class_centers)
    quality.to_csv('method/new_SDD/scores/facenet_CasiaWebFace_small_quality.csv', index=False)
    print(np.max(quality['quality']), np.min(quality['quality']), np.mean(quality['quality']))
