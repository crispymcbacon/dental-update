from Patient import Patient
from itertools import product
import psutil

# Example usage:
if __name__ == "__main__":

    # Initialize the Patient object with the configuration file
    config_path = "../data/config_1.json"
    patient = Patient(config_path)

    visits_to_process = [0, 1]
    computations = ["initialize", "add_visits", "triangulation",
                    "reload", "compare_original", "compare_visits"]
    patient.process_all_visits(visits_to_process, computations)

    # ---------
    # Code to test the accuracy of the reconstruction
    # ---------
    # # Choose the options for the patient,
    # visits_to_process = [0]
    # computations = ["initialize", "add_visits"]
    # patient.process_all_visits(visits_to_process, computations)

    # param_options = {
    #     'upper_lower_weight': 1.5,  # Remove the list brackets
    #     'camera_translation': True,  # Remove the list brackets
    #     'camera_rotations': {  # Remove the outer list, just pass the dictionary
    #         'frontal': 1.0,
    #         'upper': 0.5,
    #         'lower': 0.5,
    #         'left': 1.5,
    #         'right': 1.5
    #     },
    #     'use_arch_constraints': True,
    #     'use_only_centroids': False,
    #     'use_alpha': True
    # }

    # test_output_file = "reconstruction_test_results.csv"
    # patient.test_reconstruction(
    #     visit_id=0,
    #     output_file=test_output_file,
    #     **param_options
    # )
