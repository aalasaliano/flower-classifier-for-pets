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

img_base64 = get_base64_image("bungaunsplash.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# ✧ UI Header ✧
# ---------------------------------------------------
st.set_page_config(page_title="Flora4Pets", page_icon="🌷")
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
