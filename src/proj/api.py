import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image

import hydra
import torch
from omegaconf import DictConfig
from proj.model import create_model
from proj.data import TRANSFORM


@hydra.main(config_path="../../configs/hydra", config_name="config", version_base=None)
def create_app(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg.model.num_classes

    model = create_model(num_classes=num_classes).to(device)

    model = None
    class_names = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal model, class_names

        root = Path("./")

        if Path("/gcs").exists():
            root = Path("/gcs")

        state_dict = torch.load(root / f"models/{cfg.model_name}.pt", map_location=device, weights_only=True)

        model = create_model(num_classes=num_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        with open(root / "models/class_names.txt") as class_names_file:
            class_names = class_names_file.readlines()
            class_names = class_names[:num_classes]
            class_names = [name.strip() for name in class_names]

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

            input_tensor = TRANSFORM(img).unsqueeze(0).to(device)

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

    return app


if __name__ == "__main__":
    app = create_app()

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, lifespan="on")
