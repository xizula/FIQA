import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['PYTHONPATH'] = parent_dir
sys.path.append(parent_dir)

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
import onnx
import onnx.helper as helper
import onnx.shape_inference as shape_inference
from onnx2pytorch import ConvertModel
import onnxruntime as ort
from face_models.models.AdaFace.inference import load_pretrained_model

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
    def __init__(self, train=False):
        self.model = load_pretrained_model('ir_101')
        self.model.eval()
        self.model.cuda()
        self.train = train

    def __call__(self, images):
        images = images.detach().cpu().numpy().transpose(0,2,3,1)
        norm = ((images) - 0.5) / 0.5
        tensor = torch.tensor(norm.transpose(0,3,1,2)).float()
        if not self.train:
          with torch.no_grad():
            embeddings, _ = self.model(tensor.cuda())
        else:
            embeddings, _ = self.model(tensor.cuda())
        return embeddings.detach().cpu()
    
    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T

class AdaFaceQuality:
    def __init__(self, adaface_model, train_data_embeddings):
        self.adaface = adaface_model
        self.train_mean = train_data_embeddings.mean(dim=0)
        self.train_var = train_data_embeddings.var(dim=0)
        self.criterion = torch.nn.MSELoss()
    
    def compute_quality(self, images):
        embeddings = self.adaface(images)
        embeddings.requires_grad_(True)  # Enable gradient tracking

        batch_mean = embeddings.mean(dim=0)
        batch_var = embeddings.var(dim=0)

        mse_loss = self.criterion(batch_mean, self.train_mean) + self.criterion(batch_var, self.train_var)

        mse_loss.backward()
        gradients = embeddings.grad.abs().sum(dim=1)

        return gradients.detach().cpu().numpy()


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
    

class QualityAdaFace(nn.Module):
    def __init__(self, **kwargs):
        super(QualityAdaFace, self).__init__()
        original_model = load_model('adaface', **kwargs)
        self.input_layer = original_model.model.input_layer
        self.body = original_model.model.body
        
        # Add new layers: dropout and a fully connected layer
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1).cuda()  # Assuming 512 features from the body
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = x.mean(dim=(2, 3))  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def load_model(name: str, **kwargs):

    if name == 'facenet':
        return FaceNet()
    elif name == 'arcface':
        return ArcFace()
    elif name == 'adaface':
        return AdaFace(train=kwargs.get('train', False))
    elif name == 'adaface_onnx':
        model = AdaFace()
        return onnx.load(model.model_path)
    elif name == 'ghostfacenet':
        return GhostFaceNet()
    else:
        raise ValueError(f'Model {name} not supported')


# model = load_model('adaface')
# print(model.model)




