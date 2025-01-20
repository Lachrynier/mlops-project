import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

import torch
from proj.model import create_model
from proj.data import TRANSFORM

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

model_path = "./models/model.pth"

# If the whole model is saved:
# model = torch.load(model_path)

# If state_dict is saved:
# Ideally this is not hardcoded but read from a config, or whole model is saved.
model = create_model(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        if not image.filename.endswith(('.jpg', 'jpeg', '.png')) :
            raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        input_tensor = TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = output.argmax(dim=1)

        return JSONResponse(content={'predicted_class': predicted_class.item()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Local debugging
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)