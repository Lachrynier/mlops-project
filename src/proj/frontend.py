import os

import pandas as pd
from PIL import Image
import requests
import streamlit as st


from google.cloud import run_v2


@st.cache_resource
def get_backend_url() -> str | None:
    """Get the URL of the backend service."""
    parent = "projects/mlops-project-77/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)

    for service in services:
        if service.name.split("/")[-1] == "api":
            return service.uri

    return os.environ.get("BACKEND", None)


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"
    response = requests.post(predict_url, files={"image": image})
    if response.status_code != 200:
        return None

    return response.json()


def main() -> None:
    """Main function of the Streamlit frontend."""
    # backend = get_backend_url()

    # if backend is None:
    #     msg = "Backend service not found"
    #     raise ValueError(msg)

    backend = "https://backend-658849725274.europe-west1.run.app"

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    result = classify_image(uploaded_file, backend=backend)

    if result is None:
        st.write("Failed to get prediction")
        return

    prediction = result["prediction"]
    probabilities = result["probabilities"]

    # show the image and prediction
    st.image(Image.open(uploaded_file), caption="Uploaded Image")
    st.write("Prediction:", prediction)

    # make a nice bar chart
    data = {"Class": [f"Class {i}" for i in range(10)], "Probability": probabilities}
    df = pd.DataFrame(data)
    df.set_index("Class", inplace=True)
    st.bar_chart(df, y="Probability")


if __name__ == "__main__":
    main()
