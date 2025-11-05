from fastapi import FastAPI, UploadFile, File

from fastapi.responses import JSONResponse

import torch
import torchvision.transforms as T
from PIL import Image
import io
import mlflow
from models.TrafficSignCNN import TrafficSignCNN

app = FastAPI(title="TrafficSignNet API")

model_path = 'models/best_model.pth'
model = TrafficSignCNN(num_classes=43)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

transform = T.Compose([
    T.ReSize((64,64)),
    T.ToTensor(),
    T.Normilize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return JSONResponse({'predicted_class': int(pred)})