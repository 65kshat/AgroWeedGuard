# utils/model_loader.py

import streamlit as st
import torch
import torch.nn as nn

from pathlib import Path
from ultralytics import YOLO
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Class Names
CLASS_NAMES = [ "Class_0",
                "Class_1",
                "Class_2",
                "Class_3",
                "Class_4",
                "Class_5",
                "Class_6",
                "Class_7",
                "Class_8",
                "Class_9",
                "Class_10",
                "Class_11"]


# Loading Yolo Model
@st.cache_resource
def load_yolo():
    model_path = (MODELS_DIR / "YOLO" / "yolov8n_weed_detector" / "weights" / "best.pt")
    model = YOLO(str(model_path))

    return model


# Loading CNN Model
@st.cache_resource
def load_cnn():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 12)
    model.load_state_dict(torch.load(MODELS_DIR / "CNN" / "best_cnn_model.pth", map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    return model


# Loading ViT Model
@st.cache_resource
def load_vit():
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 12)
    model.load_state_dict(torch.load(MODELS_DIR / "ViT" / "best_vit_model.pth", map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    return model


# Loading Faster R-CNN Model
@st.cache_resource
def load_fasterrcnn():
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = (model.roi_heads.box_predictor.cls_score.in_features)

    model.roi_heads.box_predictor = (models.detection.faster_rcnn.FastRCNNPredictor(in_features, 13))

    model.load_state_dict(torch.load(MODELS_DIR / "FasterRCNN" / "best_faster_rcnn_model.pth", map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    return model