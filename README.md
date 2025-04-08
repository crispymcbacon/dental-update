# 3D Dental representation from 2D Images

## Intro

This project is related to the Master Thesis _"AI-Driven Dental Model Evolution: Updating 3D
Representations Through Sequential 2D Image Analysis"_ in collaboration between the Hospital of Trieste and University of Trieste. The goal is to **reconstruct a simple 3d model from standard intraoral pictures**, in order to track teeth movements during aligners therapy.

NOTE: this project is still in development status.

## Quickstart

**Train the segmentation model:**

Copy-paste `rcnn_train.py` into a colab notebook and run it, save the segmentation model.

(Optional) Copy-paste `rcnn_predict.py` into a colab notebook and run it, in order to test the inference.

**Copy the data**

Copy the data inside the `/data` folder, refer to the data section for more details.

Copy the segmentation model under `/model` with name `dentalai_model.pth`.

**Generate the mask and the basic JSON data**

There are two ways to generate the mask and the basic JSON data:

1. Using the REST API (suggested)
2. Using the `main_process.py` script

**Using the REST API**

For now the API is still in development and support only the prediction of the mask and the basic JSON data.

Run with:

```sh
fastapi dev api.py
```

Use the endpoint `/predict_image/` to generate the mask and the basic JSON data (if you are using the web interface, it can be done in the interface).

Up to now it requires a manual process to add the apex and base points of each tooth, this is done with a web interface (see [dental-reconstruction-interface](https://github.com/crispymcbacon/dental-update-webapp)).

When you have all the data you can stop the API.

**Using the `main_process.py` script**

Edit the config path of the `main_process.py` script and run it, it will generate the mask and the basic JSON data.

You still need to add the apex and base points of each tooth, this is done with a web interface (see [dental-reconstruction-interface](https://github.com/crispymcbacon/dental-update-webapp)).

**Run the reconstruction**

In the `src/main_final.py` edit the configuration file, select the options and run the code.

## Data structure

Load the patient data to the `/data` folder or specify the paths in the configuration file.

Each patient data should have configuration file in JSON format (`eg. config_1.json`), refer to the structure specified in the `data/data-structure.md` file, an example is provided.

## Project

Under `/src` there is:

- `main_process.py` that do the segmentation an the numeration
- `main_final.py` the final code for reconstruction, parameters can be passed in order to test more parameters
- `main_parallel_multi_visit.py` perform a grid search of best paramters, optimized to run in parallel
- `results_final.py` analyze the results with summary statistics
- `results_full.py` the full results with tests.

### Structure

#### DentalAnalyzer

`/src/dental_analyzer`

The `DentalAnalyzer` class serves as the main orchestrator for dental image analysis, handling the complete workflow from data loading to visualization:

1. **Data Loading**: Uses `TeethDataLoader` to import dental prediction data from JSON.

2. **Teeth Numbering**: Employs `TeethNumberAssigner` to label each detected tooth according to the FDI numbering system (11-48) based on the view:

   - Supports various dental views: frontal, upper, lower, left, and right
   - Handles tooth quadrant assignment using spatial boundaries
   - Uses clustering algorithms to determine jaw separation

3. **Visualization**: Renders the analyzed teeth with their assigned numbers using `TeethVisualizer`.

4. **Results Storage**: Saves the processed data for further use.

#### DentalPredictor

`/src/dental_predictor`

The `DentalPredictor` class is a specialized tool for dental image analysis that implements tooth detection and segmentation using deep learning. Here's what it does:

**Core Functionality**

- **Initialization**: Sets up a Mask R-CNN instance segmentation model (from Detectron2) configured for tooth detection
- **Prediction**: Takes dental images and identifies individual teeth using the trained model
- **Visualization**: Renders predictions with color-coded segmentation masks
- **Data Export**: Saves detailed teeth information for further processing

**Key Methods**

- `__init__(model_path)`: Configures the model with appropriate parameters
- `predict_image(image_path)`: Runs inference on a single image
- `visualize(image_path, show, save_path)`: Creates a visual representation of detected teeth
- `save_predictions(image_input, output_dir, view)`: Generates structured data including:
  - Centroid coordinates for each tooth
  - Area and bounding box measurements
  - Confidence scores
  - Relative tooth positions
  - JSON data output and visualization masks

#### MultiviewAnalyzer

The `MultiViewAnalyzer` class in the `/src/multiview_analyzer` package is a specialized tool that performs 3D reconstruction.

## Core Functionality

- **3D Reconstruction**: Creates a 3D model of teeth from multiple 2D views (frontal, upper, lower, left, right)
- **Camera Configuration**: Manages virtual camera setups for different views
- **Triangulation**: Uses multi-view triangulation to determine 3D tooth positions
- **Comparison**: Analyzes differences between:
  - Reconstructed models and ground truth (STL files)
  - Different visits (to track tooth movement over time)

## Key Methods

- `triangulation()`: Performs 3D reconstruction with configurable parameters
- `compare_with_original()`: Compares reconstructed teeth with ground truth models
- `compare_reconstructions()`: Analyzes differences between two reconstructions (e.g., visits)
- `compare_reconstructions_all()`: Extended comparison including all landmarks (centroids, apex, base)
- `visualize_camera_planes()`: Debug visualization of camera setups
- `save_model()` and `load_model()`: Handles persistence of reconstructed models

The analyzer works in conjunction with `DataProcessor` and `Plot` helper classes to process dental data and visualize results.

## Segmentation model

Under the `/colab` folder there are two files that can copy-pasted into colab notebook in order to train (`rcnn_train.py`) and make inference (`rcnn_predict.py`) with a segmentation model that identify each single tooth.

## REST API

Make sure you have the segmentation model under `/model` with name `dentalai_model.pth`.

Run with:

```sh
fastapi dev api.py
```

Endpoints provided:

- `/predict_image/` - Predict the mask and the basic JSON data
