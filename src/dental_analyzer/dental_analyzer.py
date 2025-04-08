from .teeth_data_loader import TeethDataLoader
from .teeth_number_assigner import TeethNumberAssigner
from .teeth_visualizer import TeethVisualizer

# Serves as a high-level orchestrator that ties together the functionalities of data loading,
# number assignment, orientation calculation, and visualization.
#
# What it does:
# - Loads data using TeethDataLoader.
# - Assigns tooth numbers via TeethNumberAssigner.
# - Calculates tooth orientation using TeethOrientationCalculator.
# - Visualizes the data with TeethVisualizer.
# - Saves the processed results.

class DentalAnalyzer:
    def __init__(self):
        self.data_loader = TeethDataLoader()
        self.number_assigner = None
        self.visualizer = None
        self.data = None

    def load_prediction(self, json_data, is_path=False):
        """
        Load prediction data directly from a JSON dictionary.
        """
        if is_path:
            self.data = self.data_loader.load_prediction(json_data)
        else:
            self.data = json_data
        return self.data

    def assign_teeth_numbers(self, view=None, mask_path=None):
        self.number_assigner = TeethNumberAssigner(self.data)
        self.data = self.number_assigner.assign_teeth_numbers(view=view, mask_path=mask_path)
        return self.data

    def visualize(self, image_path=None, mask_path=None, view=None):
        if self.number_assigner is None:
            raise ValueError("Teeth numbers must be assigned before visualization.")
        self.visualizer = TeethVisualizer(self.data, self.number_assigner.boundaries)
        self.visualizer.visualize(image_path=image_path, mask_path=mask_path, view=view)

    def save_results(self, output_path):
        self.data_loader.data = self.data
        self.data_loader.save_results(output_path)
