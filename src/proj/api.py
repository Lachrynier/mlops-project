from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

import torch
from proj.model import create_model
from proj.data import TRANSFORM

from google.cloud import storage
import os

storage_client = storage.Client()
bucket_name = os.environ['BUCKET_NAME']
print(f'type(bucket_name): {type(bucket_name)}')
print(f'bucket_name: {bucket_name}')

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob('models/model.pth')

model_path = 'model.pth'
blob.download_to_filename(model_path)


DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# If the whole model is saved:
# model = torch.load(model_path)

# If state_dict is saved:
# Ideally this is not hardcoded but read from a config, or whole model is saved.
model = create_model(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()


app = FastAPI()

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        if not image.filename.endswith(('.jpg', 'jpeg', '.png')) :
            raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = output.argmax(dim=1)

        return JSONResponse(content={'predicted_class': predicted_class.item()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Local debugging
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)