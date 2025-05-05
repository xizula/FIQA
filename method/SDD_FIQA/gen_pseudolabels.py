import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import sys
import os
from tqdm import tqdm
import random
import ast
np.random.seed(42)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)


FIXED_NUM = 24
REPEATS = 3

def cos(e1: np.ndarray, e2: np.ndarray) -> float:
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

def same_ppl_sim(data: pd.DataFrame, ids: np.ndarray) -> (dict, int):
    sim = {}
    print("*** Calculating positive similarity ***")
    for label in tqdm(ids):
        id_data = data[data['label'] == label]  # dataframe for a single id
        full_paths = id_data['path'].values
        for i, image in id_data.iterrows():   # for every image
            img_path = image['path']
            paths = np.delete(full_paths, np.where(full_paths == img_path))
            compare_paths = random.sample(sorted(paths), FIXED_NUM)
            img_sim = []
            for path in compare_paths:
                e1 = ast.literal_eval(image['embedding'])
                e2 = ast.literal_eval(id_data[id_data['path'] == path]['embedding'].values[0])
                img_sim.append(cos(e1, e2))
            sim[img_path] = img_sim
    pair_num = FIXED_NUM * len(sim.keys())
    return sim, pair_num     

def diff_ppl_sim(data: pd.DataFrame) -> dict:
    sim = {}
    print("*** Calculating negative similarity ***")
    for label in tqdm(ids):
        id_data = data[data['label'] == label]  # dataframe for a single id
        not_id_data = data[data['label'] != label]
        neg_paths = not_id_data['path'].values
        for i, image in id_data.iterrows():   # for every image
            img_path = image['path']
            compare_paths = random.sample(sorted(neg_paths), FIXED_NUM)
            img_sim = []
            for path in compare_paths:
                e1 = ast.literal_eval(image['embedding'])
                e2 = ast.literal_eval(not_id_data[not_id_data['path'] == path]['embedding'].values[0])
                img_sim.append(cos(e1, e2))
            sim[img_path] = img_sim
    return sim

def gen_distance(same_dict: dict, diff_dict: dict) -> dict:
    imgs = list(same_dict.keys())
    distance = {}
    for img in imgs:
        same = same_dict[img]
        diff = diff_dict[img]
        dist = wasserstein_distance(same, diff)
        distance[img] = dist
    dist_df = pd.DataFrame(distance.items(), columns=['path', 'distance'])
    return distance

def calculate_id_score(distances: dict) -> dict:
    idsocre_dist = {}
    idnames = set()
    keys = list(distances.keys())
    quality_scores = list(distances.values())
    for i in keys: idsocre_dist[i.split("\\")[-2]] = [0,0]
    
    quality_scores = (quality_scores - np.min(quality_scores)) / \
                    (np.max(quality_scores) - np.min(quality_scores)) * 100
    for i, img in enumerate(keys):
        idname = img.split('\\')[-2]
        idnames.add(idname)
        idsocre_dist[idname][0] += quality_scores[i]
        idsocre_dist[idname][1] += 1
    idnames = list(idnames)    
    id_score = {}
    for i in tqdm(idnames): id_score[i] = (idsocre_dist[i][0] / idsocre_dist[i][1])
    return id_score

def norm_labels(distances: dict, id_score: dict) -> dict:
    keys = list(distances.keys())
    quality_scores = list(distances.values())
    quality_scores = (quality_scores-np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100
    return quality_scores

if __name__ == '__main__':
    data = pd.read_csv('mgr_data/embeddings/adaface_CasiaWebFace_small_embeddings.csv')
    qc = []
    ids = sorted(data['label'].unique())
    for i in range(REPEATS):
        same_sim, num_pairs = same_ppl_sim(data, ids)
        diff_sim = diff_ppl_sim(data)
        distances = gen_distance(same_sim, diff_sim)
        id_score = calculate_id_score(distances)
        quality_scores = norm_labels(distances, id_score)
        qc.append(quality_scores)
        qc_mean = np.mean(qc, axis=0)
        qc_data = pd.DataFrame({'path': list(distances.keys()), 'quality': qc_mean})
        qc_data.to_csv(f'adaface_CasiaWebFace_small_quality_{i}.csv', index=False)

    qc = np.mean(qc, axis=0)
    qc_data = pd.DataFrame({'path': list(distances.keys()), 'quality': qc})
    qc_data.to_csv('adaface_CasiaWebFace_small_quality.csv', index=False)




    

