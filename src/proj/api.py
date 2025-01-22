from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

import torch
from proj.model import create_model
from proj.data import TRANSFORM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATHS = ["/gcs/models/model.pth", "models/model.pth"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    state_dict = None

    for path in MODEL_PATHS:
        try:
            state_dict = torch.load(path, weights_only=True)
            break
        except FileNotFoundError:
            continue

    if state_dict is None:
        raise RuntimeError(f"No model found. Searched {MODEL_PATHS}.")

    model = create_model(num_classes=10).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    yield

    del model


app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        if not image.filename.endswith((".jpg", "jpeg", ".png")):
            raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image.")

        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))

        input_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = output.argmax(dim=1)
        probabilities = output.softmax(dim=1)

        return JSONResponse(
            content={"predicted_class": predicted_class.item(), "probabilities": probabilities.tolist()}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Local debugging
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
