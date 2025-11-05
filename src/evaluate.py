import torch
import mlflow 
import json
import os
import dagshub
import yaml
from dotenv import load_dotenv

from loguru import logger
from rich.console import Console
from rich.progress import track

from sklearn.metrics import classification_report

from data.gtsrb_dataset import get_dataloaders
from models.TrafficSignCNN import TrafficSignCNN

def evaluate():
    console = Console()
    console.rule("Step 2. Evaluate the model")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_dl = get_dataloaders()

    load_dotenv()  
    params = yaml.safe_load(open('params.yaml'))['dagshub']
    dagshub.init(repo_owner=params['repo_owner'], repo_name=params['repo_name'], mlflow=True)

    mlflow.set_experiment('TrafficSignNet')

    model = TrafficSignCNN(num_classes=43)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in track(val_dl, description="Evaluating"):
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    
    report = classification_report(y_true, y_pred, output_dict=True)

    os.makedirs('reports', exist_ok=True)

    with open('reports/metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

    logger.info("Save metrics with the help of MLflow")
    with mlflow.start_run():
        mlflow.log_metrics({
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"] 
        })
        mlflow.log_artifact('reports/metrics.json')

    logger.info(f"precision: {report["weighted avg"]["precision"]}")
    logger.info(f"recall: {report["weighted avg"]["recall"]}")
    logger.info(f"f1-score: {report["weighted avg"]["f1-score"]}")

    logger.success("Evaluating has finished!")
    
if __name__ == '__main__':
    evaluate()


