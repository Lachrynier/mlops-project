from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

import torch
from proj.model import create_model
from proj.data import TRANSFORM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
ROOTS = [Path("/gcs"), Path("")]
MODEL_PATH = "models/model.pth"
CLASS_NAMES_PATH = "models/class_names.txt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global class_names

    for root in ROOTS:
        try:
            state_dict = torch.load(root / MODEL_PATH, weights_only=True)

            model = create_model(num_classes=10).to(DEVICE)
            model.load_state_dict(state_dict)
            model.eval()

            with open(root / CLASS_NAMES_PATH) as class_names_file:
                class_names = class_names_file.readlines()
                class_names = class_names[:NUM_CLASSES]
                class_names = [name.strip() for name in class_names]

        except FileNotFoundError:
            continue

    if model is None or class_names is None:
        raise RuntimeError(f"No model found. Searched {[root / MODEL_PATH for root in ROOTS]}.")

    yield

    del model
    del class_names


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
            output = model(input_tensor).squeeze(dim=0)

        prediction = output.argmax(dim=0)
        probabilities = output.softmax(dim=0)

        return JSONResponse(content={"prediction": prediction.item(), "probabilities": probabilities.tolist()})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/class_names/")
async def get_class_names():
    return JSONResponse(content={"class_names": class_names})


# Local debugging
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
