# SmartVision AI — Intelligent Multi-Class Object Recognition System

A computer vision platform combining **transfer-learning image classification** and **YOLOv8 object detection**, trained on a curated 25-class subset of the COCO dataset. Built as a multi-page Streamlit application with four classification architectures and a fine-tuned YOLOv8 detector.

## Features

- **Image Classification** — upload a cropped image of a single object and get predictions from 4 CNN architectures side by side (VGG16, ResNet50, MobileNetV2, EfficientNetB0), each with top-5 confidence breakdowns.
- **Object Detection** — upload a full scene and get every detected object localized with bounding boxes, labels, and confidence scores, with an adjustable confidence threshold.
- **Model Performance Dashboard** — compare accuracy, inference speed, and model size across all classifiers, plus per-model confusion matrices.

## Dataset

- **Source:** COCO 2017, via Hugging Face (`detection-datasets/coco`)
- **Classes:** 25 curated categories spanning vehicles, animals, people, furniture, and household/kitchen items
- **Classification set:** 2,500 images (100/class), cropped directly to fill 224×224 (no letterbox padding), 70/15/15 train/val/test split
- **Detection set:** ~2,000 full images with YOLO-format bounding box annotations, same split ratio

25 classes: `person, bicycle, car, motorcycle, airplane, bus, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, couch, potted plant, bed`

## Model Performance

### Classification (test set)

| Model          | Test Accuracy | Top-5 Accuracy | F1 (macro) | Inference Time | Model Size |
|----------------|---------------|-----------------|------------|-----------------|------------|
| **ResNet50**       | 80.0%     | 94.4%           | 0.801      | 17.2 ms         | 90.2 MB    |
| EfficientNetB0 | 77.6%     | 92.0%           | 0.774      | 6.1 ms          | 16.9 MB    |
| VGG16          | 76.5%     | 93.9%           | 0.762      | 41.3 ms         | 512.6 MB   |
| MobileNetV2    | 71.2%     | 92.8%           | 0.710      | 4.8 ms          | 8.8 MB     |

**Selected model:** EfficientNetB0 is used as the app's primary model — best accuracy-to-efficiency tradeoff (2.8x faster and 5.3x smaller than ResNet50 for a 2.4-point accuracy tradeoff), well suited to cloud CPU deployment. ResNet50 is retained as the highest-accuracy alternative.

### Object Detection (YOLOv8n, test set)

| Metric | Value |
|---|---|
| mAP@0.5 | 75.1% |
| mAP@0.5:0.95 | 41.2% |
| Precision | 87.4% |
| Recall | 68.6% |
| Inference speed | ~17 ms/image (CPU) |

## Known Limitations

- All models were trained on **CPU** (no GPU available during development), which capped total training epochs relative to typical GPU-trained baselines.
- The `person` class underperforms across every classification model (lowest F1-score consistently), traced during EDA to high scale/occlusion variance in its training crops (many small, distant, or partially-visible instances in crowd scenes) — a data characteristic, not a model defect.
- A small fraction (~0.1%) of classification crops have minor rotation artifacts inherited from the source dataset preparation pipeline.
- The original YOLO detection labels contained boundary-clipping errors (bounding box coordinates slightly exceeding image bounds) affecting ~77-80% of files; these were repaired by stripping invalid annotation lines while preserving all other valid boxes in each image (see `notebooks/07_yolo_training.ipynb` for the diagnostic and fix).

## Project Structure

```
SmartVision/
├── app.py                          # Streamlit application (this is the entry point)
├── requirements.txt
├── README.md
├── models/                         # Trained model weights (tracked via Git LFS)
│   ├── vgg16_best.pth
│   ├── resnet50_best.pth
│   ├── mobilenetv2_best.pth
│   ├── efficientnetb0_best.pth
│   ├── yolov8_best.pt
│   ├── classification_class_to_idx.json
│   └── model_selection.json
├── notebooks/                      # Training & analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_vgg16_training.ipynb
│   ├── 03_resnet50.ipynb
│   ├── 04_mobilenetv2.ipynb
│   ├── 05_efficientnetb0.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_yolo_training.ipynb
│   ├── 08_vali_split.ipynb
│   └── 09_pipeline_integration.ipynb
└── results/                        # Metrics, plots, confusion matrices
    ├── metrics.json
    ├── yolo_metrics.json
    ├── model_selection.json
    ├── confusion_matrices/
    └── eda_plots/
```

Note: `smartvision_dataset/` (raw + processed data, ~334MB) is excluded from this repository via `.gitignore`. See the Dataset section above for how to regenerate it.

## Local Setup

1. Clone the repository:
   ```
   git clone <your-repo-url>
   cd SmartVision
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Technical Stack

Python · PyTorch · Torchvision · Ultralytics YOLOv8 · OpenCV · Streamlit · scikit-learn · pandas · matplotlib

## Author

Ramesh krishna K