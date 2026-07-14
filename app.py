"""
SmartVision AI - Intelligent Multi-Class Object Recognition System
Single-file Streamlit application combining:
  - 4 CNN classifiers (VGG16, ResNet50, MobileNetV2, EfficientNetB0)
  - YOLOv8 object detection
  - Model performance dashboard
  - End-to-end detection + classification pipeline

Expected project structure (this file lives in app/app.py):

SmartVision/
├── smartvision_dataset/
├── notebooks/
├── models/
│   ├── vgg16_best.pth
│   ├── resnet50_best.pth
│   ├── mobilenetv2_best.pth
│   ├── efficientnetb0_best.pth
│   ├── yolov8_best.pt
│   ├── classification_class_to_idx.json
│   └── model_selection.json
├── results/
│   ├── metrics.json
│   ├── yolo_metrics.json
│   ├── confusion_matrices/
│   └── eda_plots/
└── app/
    └── app.py   <-- this file

Run with:  streamlit run app.py   (from inside the app/ folder)
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_class_mapping():
    """Alphabetical class order used by the classification models (ImageFolder order)."""
    mapping_path = MODELS_DIR / "classification_class_to_idx.json"
    with open(mapping_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


@st.cache_resource
def load_vgg16():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, NUM_CLASSES))
    model.load_state_dict(torch.load(MODELS_DIR / "vgg16_best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, NUM_CLASSES))
    model.load_state_dict(torch.load(MODELS_DIR / "resnet50_best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_mobilenetv2():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, NUM_CLASSES))
    model.load_state_dict(torch.load(MODELS_DIR / "mobilenetv2_best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_efficientnetb0():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODELS_DIR / "efficientnetb0_best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO

    return YOLO(str(MODELS_DIR / "yolov8_best.pt"))


CLASSIFIER_LOADERS = {
    "VGG16": load_vgg16,
    "ResNet50": load_resnet50,
    "MobileNetV2": load_mobilenetv2,
    "EfficientNetB0": load_efficientnetb0,
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def classify_image(model, pil_image, idx_to_class, top_k=5):
    """Run a single classifier on a PIL image, return top-k (label, confidence) pairs."""
    input_tensor = CLASSIFY_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)
    return [
        (idx_to_class[idx.item()], round(prob.item(), 4))
        for prob, idx in zip(top_probs, top_idxs)
    ]


def run_detection_pipeline(yolo_model, pil_image, conf_threshold=0.5, iou_threshold=0.45):
    """Run YOLO detection on a PIL image, return annotated image (RGB np array) + detections list."""
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = yolo_model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        verbose=False,
    )[0]

    annotated_bgr = results.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        detections.append(
            {
                "label": yolo_model.names[class_id],
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    return annotated_rgb, detections


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_home():
    st.title("🔍 SmartVision AI")
    st.subheader("Intelligent Multi-Class Object Recognition System")

    st.markdown(
        """
        SmartVision AI combines **transfer-learning image classification** and
        **YOLOv8 object detection** into a single computer vision platform,
        trained on a curated 25-class subset of the COCO dataset.

        **What this app does:**
        - Classify a single cropped object across 4 different CNN architectures
        - Detect and localize multiple objects in a full scene with bounding boxes
        - Compare model performance (accuracy, speed, size) side by side
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Classes Supported", "25")
    with col2:
        st.metric("Classification Models", "4")
    with col3:
        st.metric("Detection Model", "YOLOv8n")

    st.markdown("---")
    st.markdown("### How to use this app")
    st.markdown(
        """
        1. Go to **Image Classification** to upload a cropped image of a single object
           and see predictions from all 4 trained models.
        2. Go to **Object Detection** to upload a full scene and see every detected
           object with bounding boxes, labels, and confidence scores.
        3. Go to **Model Performance** to view accuracy, speed, and size comparisons
           across all models, plus confusion matrices.
        4. Go to **About** for dataset details, architecture info, and tech stack.
        """
    )


def page_classification():
    st.title("🖼️ Image Classification")
    st.markdown("Upload an image of a **single object** to classify it across all 4 models.")

    class_to_idx, idx_to_class = load_class_mapping()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        col_img, col_results = st.columns([1, 2])
        with col_img:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        with col_results:
            st.markdown("### Predictions from all 4 models")

            with st.spinner("Running inference..."):
                all_results = {}
                for name, loader in CLASSIFIER_LOADERS.items():
                    model = loader()
                    all_results[name] = classify_image(model, pil_image, idx_to_class, top_k=5)

            # Top-1 comparison table
            top1_rows = [
                {"Model": name, "Prediction": preds[0][0], "Confidence": f"{preds[0][1]:.1%}"}
                for name, preds in all_results.items()
            ]
            st.dataframe(pd.DataFrame(top1_rows), use_container_width=True, hide_index=True)

            st.markdown("### Top-5 breakdown per model")
            tabs = st.tabs(list(all_results.keys()))
            for tab, (name, preds) in zip(tabs, all_results.items()):
                with tab:
                    df = pd.DataFrame(preds, columns=["Class", "Confidence"])
                    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload an image to get started.")


def page_detection():
    st.title("📦 Object Detection")
    st.markdown("Upload a full scene image to detect **all objects** with bounding boxes.")

    yolo_model = load_yolo()

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="detect_upload")

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        with st.spinner("Running detection..."):
            annotated_rgb, detections = run_detection_pipeline(
                yolo_model, pil_image, conf_threshold=conf_threshold
            )

        col_img, col_results = st.columns([2, 1])
        with col_img:
            st.image(annotated_rgb, caption="Detections", use_container_width=True)

        with col_results:
            st.markdown(f"### {len(detections)} object(s) detected")
            if detections:
                df = pd.DataFrame(detections)
                df["confidence"] = df["confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(
                    df[["label", "confidence"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No objects detected above this confidence threshold.")
    else:
        st.info("Upload an image to get started.")


def page_model_performance():
    st.title("📊 Model Performance")

    metrics_path = RESULTS_DIR / "metrics.json"
    yolo_metrics_path = RESULTS_DIR / "yolo_metrics.json"
    selection_path = MODELS_DIR / "model_selection.json"

    if metrics_path.exists():
        st.markdown("### Classification Model Comparison")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        df = pd.DataFrame(metrics)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Test Accuracy**")
            st.bar_chart(df.set_index("Model")["Test Accuracy"])
        with col2:
            st.markdown("**Avg Inference Time (ms)**")
            st.bar_chart(df.set_index("Model")["Avg Inference Time (ms)"])
    else:
        st.warning("Classification metrics not found. Run the model comparison notebook first.")

    if selection_path.exists():
        with open(selection_path, "r") as f:
            selection = json.load(f)
        st.markdown("### Recommended Model")
        st.success(
            f"**Primary:** {selection['primary_model']} — {selection['primary_reason']}"
        )
        st.info(
            f"**Alternative:** {selection['alternative_model']} — {selection['alternative_reason']}"
        )

    if yolo_metrics_path.exists():
        st.markdown("### YOLOv8 Detection Performance")
        with open(yolo_metrics_path, "r") as f:
            yolo_metrics = json.load(f)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP@0.5", f"{yolo_metrics['mAP50']:.1%}")
        col2.metric("mAP@0.5:0.95", f"{yolo_metrics['mAP50-95']:.1%}")
        col3.metric("Precision", f"{yolo_metrics['precision']:.1%}")
        col4.metric("Recall", f"{yolo_metrics['recall']:.1%}")

    st.markdown("### Confusion Matrices")
    cm_dir = RESULTS_DIR / "confusion_matrices"
    if cm_dir.exists():
        cm_files = sorted(cm_dir.glob("*.png"))
        if cm_files:
            selected_cm = st.selectbox(
                "Select a model", [f.stem.replace("_confusion_matrix", "") for f in cm_files]
            )
            matching = [f for f in cm_files if f.stem.replace("_confusion_matrix", "") == selected_cm]
            if matching:
                st.image(str(matching[0]), use_container_width=True)
    else:
        st.info("No confusion matrices found yet.")


def page_about():
    st.title("ℹ️ About SmartVision AI")

    st.markdown(
        """
        ### Project Overview
        SmartVision AI is a computer vision platform combining transfer-learning
        image classification and YOLO-based object detection across 25 curated
        classes from the COCO dataset, spanning vehicles, animals, people,
        furniture, and everyday objects.

        ### Dataset
        - **Source:** COCO 2017 (via Hugging Face `detection-datasets/coco`)
        - **Classes:** 25 (person, vehicles, animals, furniture, kitchen items, etc.)
        - **Classification set:** 2,500 cropped images (100/class), 70/15/15 split
        - **Detection set:** ~2,000 full images with YOLO-format bounding box annotations

        ### Models
        | Model | Task | Approach |
        |---|---|---|
        | VGG16 | Classification | Frozen conv base + custom head |
        | ResNet50 | Classification | Fine-tuned last block (layer4) |
        | MobileNetV2 | Classification | Frozen base, optimized for speed |
        | EfficientNetB0 | Classification | Fine-tuned last 2 blocks + BatchNorm head |
        | YOLOv8n | Detection | Fine-tuned on 25-class subset |

        ### Technical Stack
        Python · PyTorch · Torchvision · Ultralytics YOLOv8 · OpenCV · Streamlit ·
        scikit-learn · pandas · matplotlib

        ### Notes
        All models were trained on CPU. Some classes (notably `person`) show
        lower performance across all classifiers due to high scale/occlusion
        variance in their training crops (small, distant, or partially-visible
        instances) — a data characteristic rather than a model defect,
        documented during EDA and confirmed across every trained model.
        """
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Home": page_home,
    "Image Classification": page_classification,
    "Object Detection": page_detection,
    "Model Performance": page_model_performance,
    "About": page_about,
}


def main():
    st.sidebar.title("SmartVision AI")
    selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[selection]()


if __name__ == "__main__":
    main()