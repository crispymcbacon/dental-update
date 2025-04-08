from fastapi import FastAPI, File, UploadFile, HTTPException
from src.dental_analyzer.dental_analyzer import DentalAnalyzer
from src.multiview_analyzer.multiview_analyzer import MultiViewAnalyzer
from src.dental_predictor.dental_predictor import DentalPredictor
import os
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
import json
import shutil
import tempfile

app = FastAPI()

# Configuration - Update with your actual paths and model
MODEL_PATH = "models/dentalai_model.pth"  # Update with your model path

# Create predictor instance (load model once)
predictor = DentalPredictor(MODEL_PATH)


class PredictionResponse(BaseModel):
    message: str
    mask: str  # Base64 encoded mask image
    mask_format: str  # Metadata for the mask image
    json_data: dict


class CaseResponse(BaseModel):
    message: str
    case_name: str
    output_json: dict


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def process_single_case(json_data, mask_image):
    """
    Process a single dental case given JSON data and a mask image.
    """
    try:
        # Validate input data
        if not json_data or not isinstance(json_data, dict):
            raise ValueError("Invalid JSON data format")

        if mask_image is None or not isinstance(mask_image, np.ndarray):
            raise ValueError("Invalid mask image format")

        # Create temporary files for mask and output JSON
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_file:
            mask_path = mask_file.name
            cv2.imwrite(mask_path, mask_image)

        output_json = {}
        try:
            # Initialize and process the dental analyzer
            analyzer = DentalAnalyzer()
            analyzer.load_prediction(json_data)

            # Add default values for missing fields
            if 'teeth' not in json_data:
                json_data['teeth'] = []

            for tooth in json_data.get('teeth', []):
                if 'tooth_number' not in tooth:
                    tooth['tooth_number'] = None

            analyzer.assign_teeth_numbers(mask_path=mask_path)

            # Create a temporary file for output JSON
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as output_file:
                output_path = output_file.name
                analyzer.save_results(output_path)

            # Read the output JSON from the temporary file and convert numpy types
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                output_json = convert_numpy_types(output_json)

            if not output_json:
                output_json = {"teeth": [], "message": "No teeth detected"}

            return output_json

        finally:
            # Clean up temporary files
            if os.path.exists(mask_path):
                os.remove(mask_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)

    except Exception as e:
        print(f"Error processing case: {str(e)}")
        return {"error": str(e), "teeth": [], "message": "Processing failed"}


@app.post("/predict_image/", response_model=PredictionResponse)
async def predict_image(image: UploadFile = File(...)):
    """
    Predicts dental features from a single image.

    Args:
        image: The image file to process.

    Returns:
        A dictionary containing a message, the base64 encoded mask image, and the JSON data.
    """
    try:
        # Validate the uploaded file
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload an image.")

        # Read the uploaded file into memory
        image_data = await image.read()
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Predict and get results
        output_data = predictor.save_predictions(
            img, "in_memory")  # Pass the in-memory image

        if output_data is None:
            raise HTTPException(status_code=500, detail="Prediction failed.")

        # Process the prediction data
        processed_json = process_single_case(
            output_data["json_data"], output_data["mask_image"])

        if processed_json is None:
            raise HTTPException(status_code=500, detail="Processing failed.")

        # Encode the mask image directly from memory
        _, buffer = cv2.imencode('.png', output_data["mask_image"])
        encoded_mask = base64.b64encode(buffer).decode("utf-8")

        return {
            "message": "Image prediction successful",
            "mask": encoded_mask,
            "mask_format": "image/png",
            "json_data": processed_json,
        }

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}")
