from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.model_zoo import get_model
import torch
from insightface.app import FaceAnalysis
import cv2
import onnx
import onnxruntime
from insightface.model_zoo import get_model
import numpy as np
import torch
from ellzaf_ml.models import GhostFaceNetsV2
from torchvision.io import read_image



        # return (e_i - e_j).norm().item()


class FaceNet:
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

    def __call__(self, images):
        embedding = self.resnet(images)
        return embedding.detach().cpu()

    def compute_similarities(self, e_i, e_j):
        return np.linalg.norm(e_i - e_j) # 


class ArcFace:
    def __init__(self):
        self.arcface_model = get_model('buffalo_l', allow_download=True, download_zip=True)
        self.arcface_model.prepare(ctx_id=0)

    def __call__(self, images):
        emb = []
        for img in images:
            cv2.imwrite('temp.jpg', (img.detach().cpu().numpy().transpose(1, 2, 0)*255))
            img = img.detach().cpu().numpy().transpose(1, 2, 0)*255
            
            embeddings = self.arcface_model.get_feat(img)
            emb.append(embeddings)
        return torch.tensor(emb)

    def compute_similarities(self, e_i, e_j):
        return self.arcface_model.compute_sim(e_i, e_j)

class AdaFace:
    def __init__(self):
        self.adaface_model = onnxruntime.InferenceSession("face_models/models/adaface.onnx")
        self.input_name = self.adaface_model.get_inputs()[0].name
        self.output_name = self.adaface_model.get_outputs()[0].name
    
    def __call__(self, images):
        images = images.detach().cpu().numpy().transpose(0,2,3,1)
        norm = ((images) - 0.5) / 0.5
        tensor = norm.transpose(0,3,1,2).astype(np.float32)
        embeddings = self.adaface_model.run([self.output_name], {self.input_name: tensor})[0]

        return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T


class GhostFaceNet:
    def __init__(self):
        self.model = GhostFaceNetsV2(image_size=112, width=1, dropout=0.)
        self.model.eval()
        self.model.cuda()

    def __call__(self, images):
        return self.model(images).detach().cpu().numpy()

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j) / (np.linalg.norm(e_i) * np.linalg.norm(e_j))


def load_model(name: str):
    if name == 'facenet':
        return FaceNet()
    elif name == 'arcface':
        return ArcFace()
    elif name == 'adaface':
        return AdaFace()
    elif name == 'ghostfacenet':
        return GhostFaceNet()
    else:
        raise ValueError(f'Model {name} not supported')




# img1 = read_image('mgr_data/data_sample/0/0.jpg').float().cuda()
# img2 = read_image('mgr_data/data_sample/0/3.jpg').float().cuda()
# img3 = read_image('mgr_data/data_sample/1/110.jpg').float().cuda()
# img4 = read_image('mgr_data/data_sample/1/111.jpg').float().cuda()


# model = load_model('ghostfacenet')
# e1 = model(img2.unsqueeze(0)).detach().cpu().numpy().squeeze()
# e2 = model(img1.unsqueeze(0)).detach().cpu().numpy().squeeze()


# print(model.compute_similarities(e1, e2))