"""
This file fine-tunes a pretrained RCNN model on the Dentalai dataset.
NOTE: This file is meant to be run in Google Colab.
"""

# Install required packages
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import dataset_tools as dtools
from google.colab import drive
import torch
import cv2
import numpy as np
import json
import os
!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install dataset-tools

setup_logger()

# Mount Google Drive (optional, for saving results)
drive.mount('/content/drive')

# Download dataset
dtools.download(dataset='Dentalai', dst_dir='/content/dataset-ninja/')

# Update paths for Colab
DATASET_ROOT = "/content/dataset-ninja/dentalai/"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

# Utility function to load annotations


def parse_annotations(img_dir, ann_dir):
    dataset_dicts = []

    # Iterate over annotation files
    for ann_file in sorted(os.listdir(ann_dir)):
        if not ann_file.endswith(".json"):
            continue

        # Load JSON annotation
        ann_path = os.path.join(ann_dir, ann_file)
        with open(ann_path, "r") as f:
            annotation = json.load(f)

        # Image information
        filename = os.path.join(img_dir, ann_file.replace(".json", ""))
        height, width = annotation["size"]["height"], annotation["size"]["width"]

        record = {
            "file_name": filename,
            "height": height,
            "width": width,
            "image_id": ann_file.split(".")[0],
            "annotations": []
        }

        # Parse objects
        for obj in annotation["objects"]:
            if obj["classTitle"] != "Tooth":  # Filter only "Tooth" objects
                continue

            # Polygon points
            poly = obj["points"]["exterior"]
            poly = [p for x in poly for p in x]  # Flatten list

            # Bounding box
            x_coords = [p[0] for p in obj["points"]["exterior"]]
            y_coords = [p[1] for p in obj["points"]["exterior"]]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,  # Single class: "Tooth"
            })

        dataset_dicts.append(record)
    return dataset_dicts

# Register datasets with Detectron2


def register_dentalai():
    for split, directory in [("train", TRAIN_DIR), ("valid", VALID_DIR)]:
        img_dir = os.path.join(directory, "img")
        ann_dir = os.path.join(directory, "ann")

        DatasetCatalog.register(
            f"dentalai_{split}", lambda d=img_dir, a=ann_dir: parse_annotations(d, a))
        MetadataCatalog.get(f"dentalai_{split}").set(thing_classes=["Tooth"])


register_dentalai()

# Load metadata
dentalai_metadata = MetadataCatalog.get("dentalai_train")

# FINE TUNE


# Configuration for training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("dentalai_train",)
cfg.DATASETS.TEST = ("dentalai_valid",)  # For evaluation during training
cfg.DATALOADER.NUM_WORKERS = 2

# Model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2  # Reduced batch size for T4
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
cfg.SOLVER.MAX_ITER = 3000  # Adjust based on dataset size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: "Tooth"
cfg.MODEL.DEVICE = "cuda"

# Set output directory
cfg.OUTPUT_DIR = "/content/output_dentalai"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
