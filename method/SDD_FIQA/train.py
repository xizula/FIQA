import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from generate_pseudo_labels.extract_embedding.model import model_mobilefaceNet, model
from dataset.dataset import load_quality_data
from train_config import config as conf
import numpy as np


class TrainQualityTask():

    def __init__(self):
        super(TrainQualityTask, self).__init__()

    def dataSet(self):
        trainloader, val_loader = load_quality_data(self.config, label=True, train=True)
        return trainloader, val_loader

    def backboneSet(self):
        # Network Setup
        device = self.config.device
        multi_GPUs = self.config.multi_GPUs
        if conf.backbone == 'MFN':         # MobileFaceNet
            net = model_mobilefaceNet.MobileFaceNet([112, 112], 512, \
                    output_name = 'GDC', use_type = "Qua").to(device)
        else:                                    # ResNet50
            net = model.R50([112, 112], use_type = "Qua").to(device)
        # Transfer learning from recognition model
        if self.config.finetuning_model is not None:
            print('='*20 + "FINE-TUNING" + '='*20)
            net_dict = net.state_dict()
            print('='*20 + "LOADING NETWROK PARAMETERS" + '='*20)
            pretrained_dict = torch.load(conf.finetuning_model, map_location=device)
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            same_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            diff_dict = {k: v for k, v in net_dict.items() if k not in pretrained_dict}
            net_dict.update(same_dict)
            net.load_state_dict(net_dict)
            print('='*20 + f"LOADING DONE {len(same_dict)}/{len(pretrained_dict)} LAYERS" + '='*20)
            ignore_dictName = list(diff_dict.keys())
            print ('='*20 + 'INGNORING LAYERS:' + '='*20)
            print (ignore_dictName)
        if device != 'cpu' and len(multi_GPUs) > 1:
            net = nn.DataParallel(net, device_ids = multi_GPUs)
        return net

    def trainSet(self, net):
        criterion = nn.SmoothL1Loss()

        optimizer = optim.Adam(net.parameters(),
                                lr = self.config.lr, 
                                betas=(0.9, 0.99), 
                                eps=1e-06,
                                weight_decay=self.config.weight_decay)
        scheduler_gamma = 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config.stepLR, gamma=scheduler_gamma)
        return criterion, optimizer, scheduler

    def train(self, trainloader, valloader, net, epoch, model_name):
        net.train()
        itersNum = 1
        os.makedirs(self.config.checkpoints, exist_ok=True)
        logfile = open(os.path.join(self.config.checkpoints, "log"), 'w')
        for e in range(epoch):
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
                    logfile = open(os.path.join("method/SDD/checkpoints", "log"), 'a')
                    logfile.write(f"Epoch {e+1} / {epoch} | {itersNum} Loss=" + '\t' + f"{loss}" + '\n')
                itersNum += 1
            mean_loss = loss_sum / len(trainloader)
            print(f"LR = {optimizer.param_groups[0]['lr']} | Mean_Loss = {mean_loss}")
            logfile.write(f"LR = {optimizer.param_groups[0]['lr']} | Mean_Loss = {mean_loss}" + '\n')
            if (e+1) % 1 == 0:   # save model
                os.makedirs('method/SDD/checkpoints', exist_ok=True)
                savePath = os.path.join('method/SDD/checkpoints', f"{model_name}_net_{e+1}epoch.pth")
                torch.save(net.state_dict(), savePath)
                print(f"SAVE MODEL: {savePath}")
            scheduler.step()
        return net

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(42)
    train_task = TrainQualityTask(conf)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    net = train_task.backboneSet()
    trainloader, valloader = train_task.dataSet()
    criterion, optimizer, scheduler = train_task.trainSet(net)
    net = train_task.train(trainloader, valloader, net, epoch=20)
    