import streamlit as st
import os
import numpy as np
import traceback
import torch
from PIL import Image
from torchvision import transforms
from model import BaselineCNN, EfficientAdvanced
from utils import (
    process_image_pytorch,
    get_prediction_pytorch,
    make_gradcam_heatmap,
    get_gradcam,
    get_last_conv_layer,
    plot_xray_result)

# PAGE CONFIGS
st.set_page_config(
    page_title="X-ray Disease Detection",
    page_icon="🫁",
    layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    h1 {
        text-align: center;
        color: #1f1f1f;
    }
</style>
""", unsafe_allow_html=True)

# MODEL LOADING
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")

@st.cache_resource
def load_model(model_name):
    if model_name == 'BaselineCNN':
        model = BaselineCNN(num_classes=3)
        ckpt = 'model_checkpoints/BaselineModel_model_weights.pth'
    else:
        model = EfficientAdvanced(num_classes=3)
        ckpt = 'model_checkpoints/best_EfficientNetAdvanced_finetuned.pth'
        
    state_dict = torch.load(ckpt, map_location=DEVICE)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model

# PAGE
st.title("🫁 COVID-19 / Pneumonia Detection through X-Ray")
st.write("Upload a chest X-ray image to detect **Covid** or **Pneumonia**.")

st.sidebar.header("Configuration")
MODEL_NAME = st.sidebar.radio(
    "Choose Model Architecture:",
    ['BaselineCNN', 'EfficientNet'])
model = load_model(MODEL_NAME)
target_layer = get_last_conv_layer(model)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )])

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image...",
    type=['jpg', 'jpeg', 'png'])

# MAIN
if uploaded_file:
    st.divider()
    col_left, col_right = st.columns([1, 2])
    
    # LEFT. INPUT
    with col_left:
        st.subheader("Input X-Ray")
        
        image = Image.open(uploaded_file).convert("RGB")
        try:
            st.image(image, use_container_width=True)
        except:
            st.image(image, width='stretch')
            
        uploaded_file.seek(0)
        st.markdown("")
        
        if st.button("🔍 Analyze X-Ray"):
            with st.spinner("Running diagnosis..."):
                temp_path = "temp_xray.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
            try: 
                processed = process_image_pytorch(temp_path, transform)
                prediction, confidence = get_prediction_pytorch(model, processed)
                
                heatmap = make_gradcam_heatmap(model, target_layer, processed)
                gradcam_img = get_gradcam(np.array(image), heatmap)
                
                st.session_state['result'] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'gradcam': gradcam_img}
                
                st.session_state["fig"] = plot_xray_result(
                    input_img=image,
                    gradcam_img=gradcam_img,
                    prediction=prediction,
                    confidence=confidence)
                
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
    # RIGHT. RESULT
    with col_right:
        st.subheader("Analysis Result")
        if "result" in st.session_state:
            res = st.session_state['result']
            if res['prediction'].lower() in ['covid', 'pneumonia']:
                st.error(f"### {res['prediction']}")
            else:
                st.success(f'### {res['prediction']}')
                
            st.metric("Confidence Score", f"{res['confidence']:.4f}")
            if "fig" in st.session_state:
                st.pyplot(st.session_state["fig"])
else:
    st.info("Please upload a chest X-ray image to begin.")