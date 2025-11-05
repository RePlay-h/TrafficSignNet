import torch
import mlflow 
import json
import os

from sklearn.metrics import classification_report

from data.gtsrb_dataset import get_dataloaders
from models.TrafficSignCNN import TrafficSignCNN

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_dl = get_dataloaders()

    model = TrafficSignCNN(num_classes=43)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_dl:
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    
    report = classification_report(y_true, y_pred, output_dict=True)

    os.makedirs('reports', exist_ok=True)

    with open('reports/metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

    mlflow.log_metrics({
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"] 
    })

if __name__ == '__main__':
    evaluate()


