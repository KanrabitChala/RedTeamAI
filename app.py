import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from model import SimpleCNN

app = FastAPI(title="AI Security Testing API")
model = None

@app.on_event("startup")
def load_model():
    global model
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("[*] API loaded model successfully.")

@app.get("/")
def home():
    return {"status": "online", "vulnerability": "Metadata leakage enabled"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: 1.0 - x)
    ]) #you can remove'transforms.Lambda(lambda x: 1.0 - x)' I added it to invert the image I will upload
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    return {"predicted_class": pred, "pipeline_info": "Raw model output exposed"}