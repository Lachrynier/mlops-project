import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omegaconf import DictConfig
import pytest

import proj.api


IMAGE_ROOT = "./tests/images/"
EPSILON = 1e-3


@pytest.fixture(scope="module")
def app(cfg: DictConfig):
    return proj.api.create_app(cfg.model.architecture, cfg.model.num_classes)


def test_image_predictions(app: FastAPI):
    """Test predictions for all images in the IMAGE_ROOT directory."""

    with TestClient(app) as client:
        assert os.path.exists(IMAGE_ROOT), f"Directory not found: {IMAGE_ROOT}"

        image_files = [f for f in os.listdir(IMAGE_ROOT) if f.endswith((".jpg", ".png"))]
        assert image_files, "No image files found in the directory."

        for image_file in image_files:
            image_path = os.path.join(IMAGE_ROOT, image_file)

            with open(image_path, "rb") as img:
                files = {"image": img}
                response = client.post("/predict/", files=files)

                assert response.status_code == 200, (
                    f"Failed for {image_file}: {response.status_code}: {response.json()} "
                )
                assert response.headers.get("Content-Type") == "application/json", f"Failed for {image_file}."

                json_response = response.json()

                assert "prediction" in json_response, f"Failed for {image_file}."
                pred = json_response["prediction"]
                assert isinstance(pred, int)
                assert 0 <= pred <= 256
                print(f"Prediction for {image_file}: {json_response}")

                assert "probabilities" in json_response, f"Failed for {image_file}"
                prob = json_response["probabilities"]
                assert isinstance(prob, list)
                total_prob = sum(prob)
                assert abs(total_prob - 1.0) < EPSILON


if __name__ == "__main__":
    test_image_predictions()
