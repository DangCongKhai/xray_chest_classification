import streamlit as st
from utils import (
    process_image_pytorch,
    get_prediction_pytorch,
    make_gradcam_heatmap,
    get_gradcam,
    get_last_conv_layer,
)

from model import BaselineCNN, EfficientAdvanced
from PIL import Image
import os

import numpy as np
import traceback
import torch.nn as nn
from torchvision import transforms

models = ["BaselineCNN", "EfficientNet"]


@st.cache_resource
def load_model(model_name):

    import torch.nn as nn
    import torch
    from torchvision import models

    if model_name == "BaselineCNN":

        # Load PyTorch model
        model = BaselineCNN(3)
        state_dict = torch.load(
            "model_checkpoints/BaselineModel_model_weights.pth",
            weights_only=True,
            map_location="cpu",
        )["state_dict"]
    else:
        model = EfficientAdvanced(3)
        state_dict = torch.load(
            "model_checkpoints/best_EfficientNetAdvanced_finetuned.pth",
            weights_only=True,
            map_location="cpu",
        )["state_dict"]
        for param in model.parameters():
            param.requires_grad = True
    model.load_state_dict(state_dict)
    model.eval()

    return model


# Title
st.title("Covid/Pneumonia Detection System")
st.write("Upload a chest X-ray image to detect Pneumonia or Covid")


# File uploader
uploaded_file = st.file_uploader("Upload your X-RAY image", type=["jpg", "jpeg", "png"])
model_name = st.selectbox("Select your model:", (models))
model = load_model(model_name)
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# target_layers = [get_last_conv_layer(model.sequential)]
# targets = [ClassifierOutputTarget(0)]

target_layer = get_last_conv_layer(model)


if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Display the uploaded image in the first column
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray Image", width="stretch")

    # Create a container for the second column (will be filled after prediction)
    with col2:
        heatmap_placeholder = st.empty()

    # Center the predict button using columns
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_btn = st.button("Predict", width="stretch")

    if predict_btn:
        with st.spinner("Processing..."):
            # Save temporary file
            temp_path = "temp_image.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                processed_image = process_image_pytorch(temp_path, transform)
                prediction, confidence = get_prediction_pytorch(model, processed_image)

                heatmap = make_gradcam_heatmap(model, target_layer, processed_image)

                visualization = get_gradcam(np.array(image), heatmap)

                heatmap_placeholder.image(
                    visualization,
                    caption="Grad-CAM Heatmap",
                    width="stretch",
                )

                st.success(
                    f"**Prediction**: {prediction}. **Confidence**: {confidence}"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(traceback.print_exc())
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
