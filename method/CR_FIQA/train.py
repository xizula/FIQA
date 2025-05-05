import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from dataset.dataset import load_quality_data  # Twój loader
from face_models.model import QualityAdaFace  # Twój model
from torch.utils.tensorboard import SummaryWriter

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Wczytanie danych
    train_loader, val_loader = load_quality_data(args.data_csv)
    
    # Inicjalizacja modelu
    model = QualityAdaFace(train=True).cuda()
    
    # Funkcja straty i optymalizator
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    writer = SummaryWriter(log_dir=os.path.join('method/CR_FIQA/adaface', 'tensorboard_logs'))

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for img_paths, images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader)):
            images, targets = images.to(device), targets.to(device).float()
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # print(running_loss)
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        
        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_paths, images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).float()
                outputs = model(images).squeeze()
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"method/CR_FIQA/checkpoints/adaface/checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    # Zapis modelu
    torch.save(model.state_dict(), "method/CR_FIQA/checkpoints/adaface/quality_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default='method/CR_FIQA/scores/adaface_CasiaWebFace_small_quality.csv', help="Ścieżka do CSV z danymi")
    parser.add_argument("--batch_size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok")
    args = parser.parse_args()
    
    train(args)
