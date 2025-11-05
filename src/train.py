
import torch
import torch.nn as nn
import torch.optim as optim

from loguru import logger
from rich.progress import track
from rich.console import Console

import mlflow
import yaml
import os
import numpy as np

from data.gtsrb_dataset import get_dataloaders
from models.TrafficSignCNN import TrafficSignCNN

import dagshub

from dotenv import load_dotenv

# Get train parameters
params = yaml.safe_load(open('params.yaml'))['hyperparams']
dagshub_params = yaml.safe_load(open('params.yaml'))['dagshub']

# Set tracking username and password for MLFlow
load_dotenv()



def set_seed():
    seed = yaml.safe_load(open('params.yaml'))['train']['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in track(dataloader, description='Training'):
        images, labels = images.to(device), labels.to(device)

        out_p = model(images)
        loss = criterion(out_p, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = out_p.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in track(dataloader, description='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            out_p = model(images)
            loss = criterion(out_p, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = out_p.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

def main():
    set_seed()

    console = Console()

    console.rule("Step 2. Training the model")
    # Connect dagshub
    dagshub.init(repo_owner='RePlay-h', repo_name='TrafficSignNet', mlflow=True)

    # define device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader = get_dataloaders()

    # model
    model = TrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(params['lr']))

    # MLflow setup
    mlflow.set_experiment("TrafficSignNet")

    with mlflow.start_run():
        mlflow.log_params({
            "lr": params['lr'],
            "batch_size": params['batch_size'],
            "optimizer": "Adam",
            "dropout": params['dropout'],
            'epochs': params['epochs']
        })

        best_acc = 0.0
        os.makedirs("models", exist_ok=True)

        for epoch in range(1, int(params['epochs']) + 1):
            logger.info(f"Epoch {epoch}/10")

            train_loss, train_acc = train_one_epoch(model, train_loader, 
                                                    criterion, optimizer, device)
            
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                
                torch.save(model.state_dict(), "models/best_model.pth")
                
                logger.info(f"New best model saved (acc={best_acc:.4f})")
            
        mlflow.log_metric("best_val_acc", best_acc)
        mlflow.log_artifact("models/best_model.pth")

        logger.success("Training has finished!")

if __name__ == '__main__':
    main()


