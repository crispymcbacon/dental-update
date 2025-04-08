# File: main.py
from dental_analyzer.dental_analyzer import DentalAnalyzer
from multiview_analyzer.multiview_analyzer import MultiViewAnalyzer
from multiview_analyzer_simple.multiview_analyzer import MultiViewAnalyzerSimple
from dental_predictor.dental_predictor import DentalPredictor
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json


MODEL_PATH = "../models/dentalai_model.pth"  # Update with your model path


def visualize_single_image(image_path):
    # Create predictor
    predictor = DentalPredictor(MODEL_PATH)

    # Test on single image
    predictor.visualize(image_path)


def process_single_case(json_path):
    """
    Process a single dental case given a JSON path for predictions.
    """
    try:
        # Derive paths based on the JSON file
        base_path = os.path.dirname(json_path)
        case_name = os.path.basename(base_path)
        mask_path = os.path.join(base_path, f"{case_name}_masks.png")
        output_path = os.path.join(
            base_path, f"{case_name}_teeth_labeled.json")

        # Initialize and process the dental analyzer
        analyzer = DentalAnalyzer()
        analyzer.load_prediction(json_path)
        analyzer.assign_teeth_numbers(mask_path=mask_path)
        analyzer.visualize(mask_path=mask_path)
        analyzer.save_results(output_path)

        print(f"Successfully processed: {case_name}")
        return True
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return False


def predict_single_image(dir, case_name, out_dir, view=None):
    image_path = os.path.join(dir, f"{case_name}.JPG")

    # Create predictor
    predictor = DentalPredictor(MODEL_PATH)

    predictor.predict_image(image_path)

    # Test on single image
    predictor.save_predictions(image_path, out_dir, view=view)


def process_single(dir, case_name):
    """
    Process a single dental case a dir and case name.
    """
    try:
        # Derive paths based on the JSON file
        json_path = os.path.join(dir, f"{case_name}_teeth.json")
        mask_path = os.path.join(dir, f"{case_name}_mask.png")
        out_dir = os.path.join(dir, f"{case_name}_teeth_labeled.json")

        # Initialize and process the dental analyzer
        analyzer = DentalAnalyzer()
        analyzer.load_prediction(json_path, is_path=True)
        analyzer.assign_teeth_numbers(mask_path=mask_path)
        analyzer.visualize(mask_path=mask_path)
        analyzer.save_results(out_dir)

        print(f"Successfully processed: {case_name}")
        return True
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return False


def process_all_cases(predictions_dir):
    """
    Process all cases in the given directory.
    """
    successful = 0
    failed = 0

    # Iterate through all files and subdirectories
    for root, dirs, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith('_teeth.json'):
                json_path = os.path.join(root, file)
                if process_single_case(json_path):
                    successful += 1
                else:
                    failed += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} cases")
    print(f"Failed: {failed} cases")


def process_all(cases):
    """
    Process all dental cases using the cases object.
    Each visit entry should have a 'dirs' key providing the input/output folder,
    and each view name as keys for the case names.
    """
    views = ['frontal', 'upper', 'lower', 'left', 'right']
    successful = 0
    failed = 0

    # Process each visit defined in cases
    for visit_id, visit_data in cases.items():
        in_dir = visit_data['dirs']
        out_dir = visit_data['dirs']
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing visit {visit_id} in directory: {in_dir}")

        for view in views:
            case_name = visit_data[view]
            print(f"\nProcessing {view} view: {case_name}")

            try:
                predict_single_image(in_dir, case_name, out_dir, view=view)
                if process_single(out_dir, case_name):
                    successful += 1
                    print(f"Successfully processed {case_name}")
                else:
                    failed += 1
                    print(f"Failed to process {case_name}")
            except Exception as e:
                failed += 1
                print(f"Error processing {case_name}: {str(e)}")
                continue

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} cases")
    print(f"Failed: {failed} cases")


if __name__ == "__main__":

    # ----- DEBUG
    # # Debug settings
    # debug_mode = True  # Set to False for processing all cases
    # predictions_dir = "../../data/allineatori/predictions"
    # debug_case = os.path.join(predictions_dir, "DSC_0688_copia", "DSC_0688_copia_teeth.json")
    # debug_case2 = os.path.join(predictions_dir, "DSC_0692", "DSC_0692_teeth.json")
    # img_path = "../../data/allineatori/DSC_0688.JPG"

    # if debug_mode:
    #     # Process single case for debugging
    #     print("DEBUG MODE: Processing single case...")
    #     visualize_single_image(img_path)
    #     process_single_case(debug_case)
    #     #process_single_case(debug_case2)
    # else:
    #     # Process all cases
    #     print("Processing all cases...")
    #     process_all_cases(predictions_dir)

    # # TEST
    # dir = "../../data/casi_tesi/BL/inizio/"
    # case_name = "c_DSC_1239"
    # view="right"
    # out_dir = "../../data/casi_tesi/predictions/BL/inizio/"

    # # ensure output directory exists
    # os.makedirs(out_dir, exist_ok=True)

    # #predict_single_image(dir, case_name, out_dir, view=view)
    # process_single(out_dir, case_name)

    # -----

    # Load configuration from JSON file
    config_path = "../data/config_1.json"  # Path to your config file
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        # Extract the 'visits' dictionary
        cases = config_data.get("visits", {})
        if not cases:
            print(
                f"Warning: No 'visits' found in {config_path} or the file is empty.")
            exit()  # Exit if no visits data is found
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {config_path}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        exit()

    # Call the processing function with the loaded cases
    process_all(cases)
