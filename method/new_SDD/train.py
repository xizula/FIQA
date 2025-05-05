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
from face_models.model import QualityAdaFace as QualityModel

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
       net = QualityModel(train=True)
       return net

    def trainSet(self, net):
        criterion = nn.HuberLoss()

        optimizer = optim.Adam(net.parameters(),
                                lr = 0.001, 
                                betas=(0.9, 0.99), 
                                eps=1e-06,
                                weight_decay=0.0005)
        scheduler_gamma = 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=scheduler_gamma)
        return criterion, optimizer, scheduler

    def train(self, trainloader, valloader, net, epoch, model_name,ckpt_num=None):
        net.train()
        itersNum = 1
        os.makedirs('method/new_SDD/checkpoints/adaface', exist_ok=True)
        logfile = open(os.path.join('method/new_SDD/checkpoints/adaface', "log"), 'w')
        writer = SummaryWriter(log_dir=os.path.join('method/new_SDD/adaface', 'tensorboard_logs'))
        if ckpt_num is not None:
            checkpoint = torch.load(f'method/new_SDD/checkpoints/adaface/checkpoint_epoch_{ckpt_num}.pth')
            net.load_state_dict(checkpoint)
            start_step = ckpt_num
        else:
            start_step = 0
        for e in range(start_step, epoch):
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
                    # logfile = open(os.path.join("method/CR_FIQA/checkpoints/facenet", "log"), 'a')
                    # logfile.write(f"Epoch {e+1} / {epoch} | {itersNum} Loss=" + '\t' + f"{loss}" + '\n')
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
                checkpoint_path = f"method/new_SDD/checkpoints/{model_name}/checkpoint_epoch_{e+1}.pth"
                torch.save(net.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
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
    # model = config['quality_model']
    model = 'adaface'
    csv_path = model + "_CasiaWebFace_small_quality.csv"
    img_list = f"method/new_SDD/scores/{csv_path}"
    trainloader, valloader = train_task.dataSet(img_list)
    criterion, optimizer, scheduler = train_task.trainSet(net)
    net = train_task.train(trainloader, valloader, net, epoch=20, model_name=model, ckpt_num=6)
    