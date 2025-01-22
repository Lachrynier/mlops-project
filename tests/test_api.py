import os
from fastapi.testclient import TestClient
from proj.api import app

IMAGE_ROOT = "./tests/images/"

client = TestClient(app)


def test_image_predictions():
    """Test predictions for all images in the IMAGE_ROOT directory."""
    assert os.path.exists(IMAGE_ROOT), f"Directory not found: {IMAGE_ROOT}"

    image_files = [f for f in os.listdir(IMAGE_ROOT) if f.endswith((".jpg", ".png"))]
    assert image_files, "No image files found in the directory."

    for image_file in image_files:
        image_path = os.path.join(IMAGE_ROOT, image_file)

        with open(image_path, "rb") as img:
            files = {"image": img}
            response = client.post("/predict/", files=files)

            assert response.status_code == 200, f"Failed for {image_file}: {response.status_code}"
            assert response.headers.get("Content-Type") == "application/json", f"Failed for {image_file}."

            json_response = response.json()
            assert "predicted_class" in json_response, f"Failed for {image_file}."
            pred = json_response["predicted_class"]
            assert isinstance(pred, int)
            assert 0 <= pred <= 256
            print(f"Prediction for {image_file}: {json_response}")


if __name__ == "__main__":
    test_image_predictions()
