import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from dataset.dataset import load_quality_data
from face_models.model import QualityFaceNet

sys.path.append(parent_dir)
torch.manual_seed(42)
np.random.seed(42)


class TrainQualityTask():

    def __init__(self):
        super(TrainQualityTask, self).__init__()

    def dataSet(self, img_list):
        trainloader, val_loader = load_quality_data(img_list)
        return trainloader, val_loader

    def backboneSet(self):
       net = QualityFaceNet()
       return net

    def trainSet(self, net):
        criterion = nn.SmoothL1Loss()

        optimizer = optim.Adam(net.parameters(),
                                lr = 0.0001, 
                                betas=(0.9, 0.99), 
                                eps=1e-06,
                                weight_decay=0.0005)
        scheduler_gamma = 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=scheduler_gamma)
        return criterion, optimizer, scheduler

    def train(self, trainloader, valloader, net, epoch, model_name):
        net.train()
        itersNum = 1
        os.makedirs('method/SDD_FIQA/checkpoints', exist_ok=True)
        logfile = open(os.path.join('method/SDD_FIQA/checkpoints', "log"), 'w')
        writer = SummaryWriter(log_dir=os.path.join('method/SDD_FIQA', 'tensorboard_logs'))
        for e in range(epoch):
            net.train()
            loss_sum = 0
            for _, data, labels in tqdm(trainloader, desc=f"Epoch {e+1}/{epoch}", total=len(trainloader)):
                data = data.to('cuda')
                labels = labels.to('cuda').float()
                preds = net(data).squeeze()
                loss = criterion(preds, labels)
                loss_sum += np.mean(loss.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itersNum % 100==0:
                    logfile = open(os.path.join("method/SDD_FIQA/checkpoints", "log"), 'a')
                    logfile.write(f"Epoch {e+1} / {epoch} | {itersNum} Loss=" + '\t' + f"{loss}" + '\n')
                    writer.add_scalar('Train/Loss', loss.item(), itersNum)
                itersNum += 1
            mean_train_loss  = loss_sum / len(trainloader)
            print(f"LR = {optimizer.param_groups[0]['lr']} | Mean_Train_Loss  = {mean_train_loss}")
            logfile.write(f"LR = {optimizer.param_groups[0]['lr']} | Mean_Train_Loss  = {mean_train_loss}" + '\n')
            writer.add_scalar('Train/Mean_Loss', mean_train_loss, e + 1)
                    # Validation phase
            net.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for _, data, labels in tqdm(valloader, desc=f"Validation Epoch {e+1}/{epoch}", total=len(valloader)):
                    data = data.to('cuda')
                    labels = labels.to('cuda').float()
                    preds = net(data).squeeze()
                    val_loss = criterion(preds, labels)
                    val_loss_sum += np.mean(val_loss.cpu().detach().numpy())

            mean_val_loss = val_loss_sum / len(valloader)
            print(f"Mean_Validation_Loss = {mean_val_loss}")
            logfile.write(f"Mean_Validation_Loss = {mean_val_loss}" + '\n')
            writer.add_scalar('Validation/Mean_Loss', mean_val_loss, e + 1)
            if (e+1) % 1 == 0:   # save model
                os.makedirs('method/SDD_FIQA/checkpoints', exist_ok=True)
                savePath = os.path.join('method/SDD_FIQA/checkpoints', f"{model_name}_net_{e+1}epoch.pth")
                torch.save(net.state_dict(), savePath)
                print(f"SAVE MODEL: {savePath}")
            scheduler.step()
        return net

if __name__ == "__main__":
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train_task = TrainQualityTask()
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    net = train_task.backboneSet()
    model = config['quality_model']
    csv_path = model + "_CasiaWebFace_small_quality.csv"
    img_list = f"method/SDD_FIQA/scores/{csv_path}"
    trainloader, valloader = train_task.dataSet(img_list)
    criterion, optimizer, scheduler = train_task.trainSet(net)
    net = train_task.train(trainloader, valloader, net, epoch=20, model_name=model)
    