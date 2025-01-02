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
from ellzaf_ml.models import GhostFaceNetsV2, GhostFaceNetsV1
from torchvision.io import read_image
from numpy.linalg import norm
import onnx
from onnx2torch import convert
from torch import nn
torch.manual_seed(42)
np.random.seed(42)


class FaceNet:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')

    def __call__(self, images):
        embedding = self.model(images)
        return embedding.detach().cpu()

    def compute_similarities(self, e_i, e_j):
        squared_norms = np.sum(e_i**2, axis=1) 
        dist_squared = squared_norms[:, None] + squared_norms[None, :] - 2 * (e_i @ e_i.T)
        dist_squared = np.maximum(dist_squared, 0)
        distances = np.sqrt(dist_squared)
        return distances


class ArcFace:
    def __init__(self):
        self.model = get_model('buffalo_l', allow_download=True, download_zip=True)
        self.model.prepare(ctx_id=0)

    def __call__(self, images):
        emb = []
        for img in images:
            cv2.imwrite('temp.jpg', (img.detach().cpu().numpy().transpose(1, 2, 0)*255))
            img = img.detach().cpu().numpy().transpose(1, 2, 0)*255
            
            embeddings = self.model.get_feat(img)
            emb.append(embeddings)
        return torch.tensor(emb)

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (norm(e_i) * norm(e_j)) *100

class AdaFace:
    def __init__(self):
        self.model_path = "face_models/models/adaface.onnx"
        self.model = onnxruntime.InferenceSession(self.model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
    
    def __call__(self, images):
        images = images.detach().cpu().numpy().transpose(0,2,3,1)
        norm = ((images) - 0.5) / 0.5
        tensor = norm.transpose(0,3,1,2).astype(np.float32)
        embeddings = self.model.run([self.output_name], {self.input_name: tensor})[0]

        return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T


class GhostFaceNet:
    def __init__(self):
        self.model = GhostFaceNetsV1(image_size=112, width=1, dropout=0.)
        self.model.eval()
        self.model.cuda()
    def __call__(self, images):
        return self.model(images.float()).detach().cpu().numpy()
    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j) / (np.linalg.norm(e_i) * np.linalg.norm(e_j))


class QualityFaceNet(nn.Module):
    def __init__(self):
        super(QualityFaceNet, self).__init__()
        self.model = load_model('facenet')
        self.features = nn.Sequential(*list(self.model.model.children())[:-4]).cuda()
        self.dropout = nn.Dropout(p=0.5).cuda()
        self.fc = nn.Linear(1792, 1).cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten before the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

def load_model(name: str):
    if name == 'facenet':
        return FaceNet()
    elif name == 'arcface':
        return ArcFace()
    elif name == 'adaface':
        return AdaFace()
    elif name == 'adaface_onnx':
        model = AdaFace()
        return onnx.load(model.model_path)
    elif name == 'ghostfacenet':
        return GhostFaceNet()
    else:
        raise ValueError(f'Model {name} not supported')

# model = load_model('ghostfacenet')
# for i, layer in enumerate(model.model.children()):
#     print(f"Layer {i}: {layer}")

# model = load_model('facenet')
# for i, layer in enumerate(model.model.children()):
#     print(f"Layer {i}: {layer}")

# model = load_model('adaface_onnx')
# for node in model.graph.node:
#     print(f"Name: {node.name}, OpType: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

# onnx_model = load_model('adaface_onnx')
# pytorch_model = convert(onnx_model)
# print(pytorch_model)


