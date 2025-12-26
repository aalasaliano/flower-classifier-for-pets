import streamlit as st
from PIL import Image
import time
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import pandas as pd
import base64


# ---------------------------------------------------
# ✧ Background & Custom CSS
# ---------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("stbackgroundz.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    div[data-testid="stVerticalBlock"] > div:not(:has(div[data-testid="stSidebar"])), 
    div[data-testid="stFileUploader"], 
    div[data-testid="stImage"],
    div[data-testid="stAlert"] {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }}
    
    h1, h3, h4, .stMarkdown p {{
        color: #1a1a1a !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.7);
    }}
    
    div[data-testid="stAlert"] * {{
        color: inherit !important;
    }}
    
    div[data-testid="stFileUploader"] {{
        border: 2px dashed #173b70;
        background-color: rgba(220, 236, 255, 0.9);
    }}
    
    div[data-testid="stSuccess"] {{
        background-color: rgba(199, 237, 203, 0.9);
        color: #0c6b1e !important;
    }}
    div[data-testid="stInfo"] {{
        background-color: rgba(220, 236, 255, 0.9);
        color: #173b70 !important;
    }}
    
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# ✧ UI Header ✧
# ---------------------------------------------------
st.set_page_config(page_title="Flora4Pets", page_icon="🌷")
st.title("Flora4Pets ✧")

# ---------------------------------------------------
# ✧ Device ✧
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# ✧ Load Model ✧
# ---------------------------------------------------
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 299)

state_dict = torch.load("model_efficientNetV3.pth", map_location=device, weights_only=False)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# ---------------------------------------------------
# ✧ Load Classes.txt ✧
# ---------------------------------------------------
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f]

# sanity safe:
if len(class_names) != 299:
    st.warning("Warning: classes.txt does not have 299 classes >_<")

# # ---------------------------------------------------
# # ✧ Load flower_info.csv ✧ -> USE AFTER CSV IS FIXED
# # ---------------------------------------------------
# import csv

# toxicity_dict = {}
# with open("flower_info.csv", "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         cname = row["class_name"].strip()
#         msg = row["warning"].strip()
#         toxicity_dict[cname] = msg

# ---------------------------------------------------
# ✧ Load toxicity CSV safely ✧
# ---------------------------------------------------
import csv

toxicity_dict = {}
with open("flower_info.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cname = (row.get("class_name") or "").strip()
        msg = (row.get("warning") or "").strip()

        if cname:  # only add valid class names
            toxicity_dict[cname] = msg


# ---------------------------------------------------
# ✧ Image Transform ✧
# ---------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# ---------------------------------------------------
# ✧ File Upload ✧
# ---------------------------------------------------
tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])
image = None

with tab1:
    uploaded = st.file_uploader("Upload file here", type=["jpg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Flower image you uploaded.")
with tab2:
    cameraphoto = st.camera_input("Take a picture of the flower")
    if cameraphoto:
        image = Image.open(cameraphoto).convert("RGB")
        st.image(image, caption="Flower image you took.")

if image:
    imgtensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(imgtensor)

    predidx = pred.argmax(dim=1).item()
    predclass = class_names[predidx]

    st.success(f'Prediction: **{predclass}**')

    # ---------------------------------------------------
    # ✧ Lookup toxicity info ✧
    # ---------------------------------------------------
    if predclass in toxicity_dict:
        warning_msg = toxicity_dict[predclass]

        # If the message contains anything (safe OR dangerous),
        # just show it exactly as written!
        st.info(warning_msg)

    else:
        st.info("No toxicity info available for this flower yet…")
