import os

import pandas as pd
from PIL import Image
import requests
import streamlit as st

from google.cloud import run_v2


@st.cache_resource
def get_backend_url() -> str | None:
    """Get the URL of the backend service."""

    # try environment
    if (backend := os.environ.get("BACKEND", None)) is not None:
        print(f"Obtained backend from environment variable: {backend}")
        return backend

    # try GCP
    parent = "projects/mlops-project-77/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)

    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri

    return None


def classify_image(image, backend: str) -> tuple[int, list[float]] | None:
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"
    response = requests.post(predict_url, files={"image": image})
    if response.status_code != 200:
        print(response.content)
        return None

    response_dict = response.json()

    prediction = response_dict["prediction"]
    probabilities = response_dict["probabilities"]

    return prediction, probabilities


def get_class_names(backend: str) -> list[str] | None:
    class_names_url = f"{backend}/class_names/"
    response = requests.get(class_names_url)

    if response.status_code != 200:
        return None

    response_dict = response.json()

    return response_dict["class_names"]


def main() -> None:
    backend = get_backend_url()

    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Caltech 256 Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    # get classification
    result = classify_image(uploaded_file, backend=backend)
    if result is None:
        st.write("Failed to get prediction")
        return
    prediction, probabilities = result

    # get class names
    class_names = get_class_names(backend=backend)
    if class_names is None:
        st.write("Failed to get class names")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.image(Image.open(uploaded_file), caption="Uploaded Image")

    with col_right:
        st.write("Prediction:", class_names[prediction])

        # make a bar chart with top 10 classes
        data = {"Class": class_names, "Probability": probabilities}
        df = pd.DataFrame.from_dict(data)
        df.set_index("Class", inplace=True)
        top = df.nlargest(10, columns="Probability")
        st.bar_chart(top, y="Probability", horizontal=True)


if __name__ == "__main__":
    main()
