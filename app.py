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
# ✧ Background ✧
# ---------------------------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("stbackgroundz.jpg")

st.markdown(f"""
    <style>
    # background
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    # overlay gelap
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}
    
    # text jd putih ??? idk lemme try
    .stApp {{
        color: white !important;
    }}
    
    # header + title
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }}
    
    # markdown
    .stMarkdown {{
        color: white !important;
    }}
    
    # file uploader
    .stFileUploader {{
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }}
    
    .stFileUploader label {{
        color: white !important;
        font-weight: bold;
    }}
    
    # success box
    .stSuccess {{
        background-color: rgba(0, 200, 0, 0.2) !important;
        color: white !important;
        border: 2px solid #00ff00;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }}
    
    # info box
    .stInfo {{
        background-color: rgba(0, 150, 255, 0.2) !important;
        color: white !important;
        border: 2px solid #00aaff;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }}
    
    # warning box
    .stWarning {{
        background-color: rgba(255, 200, 0, 0.2) !important;
        color: white !important;
        border: 2px solid #ffcc00;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }}
    
    # image capt
    .stImage > div {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 10px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# ✧ UI Header ✧
# ---------------------------------------------------
st.markdown("### ✧ Welcome to the Magical Predictor ✧")
st.title("Flora4Pets ✧")

# ---------------------------------------------------
# ✧ Device ✧
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# ✧ Load Model ✧
# ---------------------------------------------------
model = models.resnet50(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 299)

state_dict = torch.load("resnet50_trained_5epoch.pth", map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# model = torch.load("model.pth", map_location=device)
# model.eval()

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
uploaded = st.file_uploader("Upload file here", type=["jpg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Flower image you uploaded.")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    pred_idx = pred.argmax(dim=1).item()
    pred_class = class_names[pred_idx]

    st.success(f"Prediction: **{pred_class}**")

    # ---------------------------------------------------
    # ✧ Lookup toxicity info ✧
    # ---------------------------------------------------
    if pred_class in toxicity_dict:
        warning_msg = toxicity_dict[pred_class]

        # If the message contains anything (safe OR dangerous),
        # just show it exactly as written!
        st.info(warning_msg)

    else:
        st.info("No toxicity info available for this flower yet…")
