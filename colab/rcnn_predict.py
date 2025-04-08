"""
This file makes predictions using a pre-trained RCNN model from rcnn_train.py.
NOTE: This file is meant to be run in Google Colab.
"""

# Install required packages
import dataset_tools as dtools
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import json
import os
from google.colab import drive
!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install dataset-tools

# Mount Google Drive
drive.mount('/content/drive')


# Core imports


# Check if dataset directory exists and contains files
dst_dir = '/content/dataset-ninja/'
if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
    print("Downloading dataset...")
    dtools.download(dataset='Dentalai', dst_dir=dst_dir)
else:
    print("Dataset already exists in", dst_dir)


class DentalPredictor:
    def __init__(self, model_path):
        # Setup configuration
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)

        # Setup metadata
        MetadataCatalog.get("dental").set(thing_classes=["Tooth"])
        self.metadata = MetadataCatalog.get("dental")

    def predict_image(self, image_path):
        """Run prediction on single image"""
        img = cv2.imread(image_path)
        return self.predictor(img)

    def visualize(self, image_path, show=True, save_path=None):
        """Visualize predictions"""
        img = cv2.imread(image_path)
        outputs = self.predictor(img)

        v = Visualizer(img[:, :, ::-1],
                       metadata=self.metadata,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = v.get_image()

        if show:
            plt.figure(figsize=(12, 12))
            plt.imshow(result)
            plt.axis("off")
            plt.show()

        if save_path:
            cv2.imwrite(save_path, result[:, :, ::-1])

        return outputs

    def save_predictions(self, image_path, output_dir):
        """Save minimal teeth detection data for further processing"""
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]

        try:
            outputs = self.predict_image(image_path)
            instances = outputs["instances"].to("cpu")

            # Extract centroids and areas for spatial relationships
            masks = instances.pred_masks.numpy()
            boxes = instances.pred_boxes.tensor.numpy()

            # Create combined mask image - ensure integer dimensions
            height, width = int(masks[0].shape[0]), int(masks[0].shape[1])
            combined_mask = np.zeros((height, width, 3), dtype=np.uint8)

            # Generate distinct colors for each tooth
            num_teeth = len(masks)
            colors = plt.cm.rainbow(np.linspace(0, 1, num_teeth))[:, :3] * 255

            teeth_data = []
            for idx, (mask, box, color) in enumerate(zip(masks, boxes, colors)):
                # Calculate centroid
                y_indices, x_indices = np.where(mask)
                centroid = (np.mean(x_indices), np.mean(y_indices))

                # Add colored mask to combined image
                color_mask = np.zeros((height, width, 3), dtype=np.uint8)
                color_mask[mask] = color.astype(np.uint8)
                combined_mask = cv2.addWeighted(
                    combined_mask, 1, color_mask, 1, 0)

                # Calculate area and other metrics
                area = float(mask.sum())
                bbox_width = float(box[2] - box[0])
                bbox_height = float(box[3] - box[1])

                tooth_info = {
                    "tooth_id": idx,
                    "centroid": [float(centroid[0]), float(centroid[1])],
                    "area": area,
                    "width": bbox_width,
                    "height": bbox_height,
                    "bbox": [float(x) for x in box.tolist()],
                    "confidence": float(instances.scores[idx]),
                    "relative_position": {
                        "x": float(centroid[0] - instances.image_size[1]/2),
                        "y": float(centroid[1] - instances.image_size[0]/2)
                    }
                }
                teeth_data.append(tooth_info)

            # Sort teeth by position (left to right)
            teeth_data.sort(key=lambda x: x["centroid"][0])

            output_data = {
                "image_size": [int(x) for x in instances.image_size],
                "teeth": teeth_data
            }

            # Save JSON data
            json_path = os.path.join(output_dir, f"{basename}_teeth.json")
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            # Save combined mask image
            mask_path = os.path.join(output_dir, f"{basename}_masks.png")
            cv2.imwrite(mask_path, combined_mask)

            return output_data

        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")
            return None

########


# Example usage in Colab
# Update with your model path
MODEL_PATH = "/content/drive/MyDrive/Colab/models/dentalai_model.pth"

# Create predictor
predictor = DentalPredictor(MODEL_PATH)

# Test on single image
# Update with your test image path
test_img = "/content/dataset-ninja/dentalai/valid/img/15_jpg.rf.6dcd260c8f8a91502441b93d88b8b398.jpg"
predictor.visualize(test_img)

# Process directory
# Update with your input directory
input_dir = "/content/drive/MyDrive/Colab/datasets/allineatori"
# Results will be saved to Drive
output_dir = "/content/drive/MyDrive/Colab/datasets/allineatori/predictions"
os.makedirs(output_dir, exist_ok=True)

# Process all images
image_extensions = ['.jpg', '.jpeg', '.png']
image_files = [f for f in os.listdir(input_dir)
               if os.path.splitext(f)[1].lower() in image_extensions]

for img_file in image_files:
    print(f"Processing {img_file}...")
    img_path = os.path.join(input_dir, img_file)
    img_output_dir = os.path.join(output_dir, os.path.splitext(img_file)[0])

    try:
        predictor.save_predictions(img_path, img_output_dir)
        print(f"Successfully processed {img_file}")
    except Exception as e:
        print(f"Error processing {img_file}: {str(e)}")
