import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

def cos(e1: np.ndarray, e2: np.ndarray) -> float:
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))


def get_class_centers(data: pd.DataFrame) -> pd.DataFrame:
    center = data.groupby('label')['embedding'].apply(lambda x: np.mean([ast.literal_eval(i) for i in x], axis=0)).reset_index()
    center = center.rename(columns={'embedding': 'class_center'})
    return center

# def get_nearest_negative_center(class_centers: pd.DataFrame) -> pd.DataFrame:
#     n_classes = len(class_centers)
#     centers = np.vstack(class_centers['class_center'].values)
#     cosine_sim = cosine_similarity(centers, centers)

#     nearest_neg_centers =[]

#     for i in range(n_classes):
#         similarities = cosine_sim[i]
#         similarities[i] = -np.inf
#         nearest_label = class_centers['label'][np.argmax(similarities)].item()
#         nearest_neg_centers.append(nearest_label)

#     class_centers['nearest_neg_center'] = nearest_neg_centers
#     return class_centers

def get_quality_score(data: pd.DataFrame, class_centers: pd.DataFrame, eps: float = 1^(-9)) -> pd.DataFrame:
    quality_data = {"path": [], "quality": []}

    for _, row in data.iterrows():
        embedding = ast.literal_eval(row['embedding'])
        path = row['path']
        label = row['label']
        center = class_centers[class_centers['label'] == label]['class_center'].values[0]
        # nearest_neg_center = class_centers[class_centers['label'] == label]['nearest_neg_center'].values[0]
        # neg_center = class_centers[class_centers['label'] == nearest_neg_center]['class_center'].values[0]

        ccs = cos(embedding, center)
        # nnccs = cos(embedding, neg_center)
        # qc = ccs / (nnccs + (1+eps))
        quality_data['path'].append(path)
        quality_data['quality'].append(ccs)

    quality_df = pd.DataFrame(quality_data)
    return quality_df



if __name__ == '__main__':
    data = pd.read_csv('mgr_data/embeddings/facenet_CasiaWebFace_small_embeddings.csv')
    # data = pd.read_csv('mgr_data/embeddings/facenet_data_sample_embeddings.csv')
    class_centers = get_class_centers(data)
    # class_centers = get_nearest_negative_center(class_centers)
    quality = get_quality_score(data, class_centers)
    quality.to_csv('method/CR_FIQA/scores/facenet_CasiaWebFace_small_quality.csv', index=False)
    print(np.max(quality['quality']), np.min(quality['quality']), np.mean(quality['quality']))
