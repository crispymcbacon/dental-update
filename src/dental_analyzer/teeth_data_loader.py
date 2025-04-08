import json
import os

# Handles loading and saving of JSON data that contains teeth information.
#
# What it does:
# - Load prediction data from a JSON file.
# - Save processed data (e.g., with assigned numbers or calculated orientations) back to a JSON file.

class TeethDataLoader:
    def __init__(self):
        self.data = None

    def load_prediction(self, json_path):
        """Load teeth prediction data from JSON"""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        return self.data

    def save_results(self, output_path):
        """Save labeled teeth data to JSON file"""
        if self.data is None:
            raise ValueError("No data to save. Please load or process data first.")
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_data(self):
        if self.data is None:
            raise ValueError("Data not loaded.")
        return self.data
