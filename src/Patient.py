import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from multiview_analyzer.multiview_analyzer import MultiViewAnalyzer

class Patient:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file.
            
        Returns:
            A dictionary with configuration data.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    @staticmethod
    def build_paths(dir_path: str, base_name: str) -> Dict[str, str]:
        """
        Build file paths for the image, mask, and JSON annotation.
        
        Args:
            dir_path: Directory where files are stored.
            base_name: Base name of the file.
            
        Returns:
            A dictionary with paths for 'image', 'mask', and 'json'.
        """
        return {
            'image': os.path.join(dir_path, f"{base_name}.JPG"),
            'mask': os.path.join(dir_path, f"{base_name}_mask.png"),
            'json': os.path.join(dir_path, f"{base_name}_teeth_labeled.json")
        }

    @staticmethod
    def scale_camera_config(camera_config: Dict[str, Dict[str, np.ndarray]], scale: float) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Scale all camera translations by the given factor.
        
        Args:
            camera_config: Dictionary with camera parameters.
            scale: Scale factor.
            
        Returns:
            The scaled camera configuration.
        """
        for view in camera_config:
            camera_config[view]['t'] *= scale
        return camera_config

    def _default_compare(self, visit_data: Dict[str, Any], final_refined_teeth: Dict[Any, Any], scale_factor: float):
        """
        Perform a default comparison of the reconstruction with the original STL models
        for both upper and lower jaws.
        """
        # Compare upper jaw.
        truth_points_file_upper = visit_data.get('truth-upper', None)
        self.analyzer.compare_with_original(
            visit_data['dirs'],
            visit_data['stl-upper'],
            final_refined_teeth,
            region="upper",
            rotation_angle=0,
            rotation_angle_points_x=-3,
            rotation_angle_points_y=-3,
            rotation_angle_points_z=0,
            scale_factor=scale_factor,
            offset_x=0,
            offset_y=0.38,
            offset_z=-0.04,
            truth_points_file=truth_points_file_upper
        )
        # Compare lower jaw.
        truth_points_file_lower = visit_data.get('truth-lower', None)
        self.analyzer.compare_with_original(
            visit_data['dirs'],
            visit_data['stl-lower'],
            final_refined_teeth,
            region="lower",
            rotation_angle=0,
            rotation_angle_points_x=-3,
            rotation_angle_points_y=-3,
            rotation_angle_points_z=0,
            scale_factor=scale_factor,
            offset_x=0,
            offset_y=0.38,
            offset_z=0,
            truth_points_file=truth_points_file_lower
        )

    def _adjust_comparison_parameters(self, model_path: str, visit_data: Dict[str, Any], region: str = "upper", **kwargs):
        """
        Quickly adjust comparison parameters without rerunning the full analysis.
        
        Args:
            model_path: Path to the saved model file.
            visit_data: Visit data dictionary.
            region: 'upper' or 'lower' jaw.
            kwargs: Parameter adjustments (e.g., scale_factor, offsets, rotation angles).
        """
        analyzer, _, loaded_teeth, _ = self.reload_model(model_path, debug=False)
        if region == "upper":
            truth_points_file = kwargs.get('truth_points_file', visit_data.get('truth-upper', None))
            analyzer.compare_with_original(
                visit_data['dirs'],
                visit_data['stl-upper'],
                loaded_teeth,
                rotation_angle=kwargs.get('rotation_angle', 0),
                rotation_angle_points_x=kwargs.get('rotation_angle_points_x', 0),
                rotation_angle_points_y=kwargs.get('rotation_angle_points_y', 0),
                rotation_angle_points_z=kwargs.get('rotation_angle_points_z', 0),
                region=region,
                scale_factor=kwargs.get('scale_factor', 0.0130),
                offset_x=kwargs.get('offset_x', -0.3),
                offset_y=kwargs.get('offset_y', 0.3),
                offset_z=kwargs.get('offset_z', 0.75),
                truth_points_file=truth_points_file
            )
        elif region == "lower":
            truth_points_file = kwargs.get('truth_points_file', visit_data.get('truth-lower', None))
            analyzer.compare_with_original(
                visit_data['dirs'],
                visit_data['stl-lower'],
                loaded_teeth,
                rotation_angle=kwargs.get('rotation_angle', 0),
                rotation_angle_points_x=kwargs.get('rotation_angle_points_x', 0),
                rotation_angle_points_y=kwargs.get('rotation_angle_points_y', 0),
                rotation_angle_points_z=kwargs.get('rotation_angle_points_z', 0),
                region=region,
                scale_factor=kwargs.get('scale_factor', 0.0130),
                offset_x=kwargs.get('offset_x', -0.3),
                offset_y=kwargs.get('offset_y', 0.3),
                offset_z=kwargs.get('offset_z', 0.75),
                truth_points_file=truth_points_file
            )
        else:
            print(f"Invalid region: {region}. Must be 'upper' or 'lower'.")

    def __init__(self, config_path: str):
        """
        Initialize a Patient instance by loading the configuration file,
        setting up the visits data, and creating a MultiViewAnalyzer instance.
        """
        self.config = self.load_config(config_path)
        self.visits_data = self.config.get('visits', {})
        self.analyzer = MultiViewAnalyzer()
        self.results = {}  # To store analysis results per visit
        self.view_order = ['frontal', 'upper', 'lower', 'left', 'right']
    
    def extract_camera_config_from_json(self, visit_config: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract camera configuration from the JSON config for a specific visit.
        
        Args:
            visit_config: Dictionary with visit configuration.
            
        Returns:
            A dictionary with camera matrices (R, t, K) per view.
        """
        camera_config = {}
        extrinsics = visit_config['camera']['extrinsics']
        for view, params in extrinsics.items():
            R = np.array(params['R'], dtype=np.float64)
            t = np.array(params['t'], dtype=np.float64)
            K = np.array(params['K'], dtype=np.float64)
            camera_config[view] = {'R': R, 't': t, 'K': K}
        return camera_config
    
    def initialize_camera_config(self):
        """
        Initialize camera configuration using the first visit's settings.
        The camera translations are scaled, and the configuration is set in the analyzer.
        """
        first_visit_id = next(iter(self.visits_data))
        first_visit = self.visits_data[first_visit_id]
        camera_config = self.extract_camera_config_from_json(first_visit)
        # Scale the camera translations (initial guess)
        camera_config = Patient.scale_camera_config(camera_config, 0.01)
        self.analyzer.set_camera_config(camera_config, debug=True)
    
    def process_all_visits(self, 
                        visits_to_process: Optional[List[Union[str, int]]] = None, 
                        computations: Optional[List[str]] = None):
        """
        Process multiple visits according to the specified computation steps.
        
        The available computation steps are:
        - "initialize": Initialize camera configuration from the first visit.
        - "add_visits": Add each visit (with file paths) to the analyzer.
        - "triangulation": Visualize camera planes, perform triangulation, print output, and save the model.
        - "reload": Reload the saved model for visualization.
        - "compare_original": Compare the reconstruction with original STL models.
        - "compare_visits": Compare reconstructions between visits (if 2+ visits are processed).
        
        If computations is None, all steps will be executed.
        
        Args:
            visits_to_process: Optional list of visit IDs to process. If None, all visits will be processed.
            computations: Optional list of steps (as strings) to perform.
        """
        # If no computation steps are provided, do everything.
        if computations is None:
            computations = ["initialize", "add_visits", "triangulation", "reload", "compare_original", "compare_visits"]
        
        if visits_to_process is None:
            visits_to_process = list(self.visits_data.keys())
        else:
            visits_to_process = [str(v) for v in visits_to_process]
        
        # Step 1: Initialize camera configuration.
        if "initialize" in computations:
            self.initialize_camera_config()
        
        # Step 2: Add visits to the analyzer.
        if "add_visits" in computations:
            for visit_id in self.visits_data.keys():
                visit_data = self.visits_data[visit_id]['data']
                visit_paths = {view: Patient.build_paths(visit_data['dirs'], visit_data[view]) 
                            for view in self.view_order}
                self.analyzer.add_visit(visit_paths, int(visit_id))
        
        # Step 3: Process each visit individually.
        for visit_id in visits_to_process:
            visit_id_int = int(visit_id)
            print(f"\n{'='*50}")
            print(f"Processing Visit {visit_id}")
            print(f"{'='*50}")
            
            # Triangulation and visualization.
            if "triangulation" in computations:
                self.analyzer.visualize_camera_planes(visit_id=visit_id_int, plane_distance=0.1)
                
                refined_cameras, final_refined_teeth, alpha = self.analyzer.triangulation(
                    visit_id=visit_id_int,
                    active_views=self.view_order,
                    debug=True,
                    upper_lower_weight=1,
                    camera_translation=True,
                    camera_rotations={
                        'frontal': 0.0,
                        'upper': 0.0,
                        'lower': 0.0,
                        'left': 1.0,
                        'right': 1.0
                    },
                    use_arch_constraints=True,
                    use_only_centroids=False,
                    use_alpha=True
                )
                print(f"\nOptimized alpha = {alpha:.2f}")
                print("\nCamera configuration after optimization:")
                for view, (R, t) in refined_cameras.items():
                    print(f"  {view}:")
                    print("    R =\n", R)
                    print("    t =", t.flatten())
                
                print("\nTriangulated 3D positions for all teeth/sub-landmarks:")
                for tooth_number, pt in final_refined_teeth.items():
                    if isinstance(tooth_number, int):
                        point_type = "centroid"
                    elif isinstance(tooth_number, tuple) and len(tooth_number) == 2:
                        point_type = tooth_number[1]
                        tooth_number = tooth_number[0]
                    else:
                        point_type = "unknown"
                    print(f"  Tooth {tooth_number} ({point_type}): {pt}")
                
                # Save the model.
                visit_data = self.visits_data[visit_id]['data']
                model_file = os.path.join(visit_data['dirs'], f"mt_{visit_id}.pkl")
                self.analyzer.save_model(model_file, refined_cameras, final_refined_teeth, alpha)
                
                # Store the results for later use.
                self.results[visit_id_int] = {
                    'model_file': model_file,
                    'refined_cameras': refined_cameras,
                    'final_refined_teeth': final_refined_teeth,
                    'alpha': alpha
                }
            
            # Reload the saved model.
            if "reload" in computations:
                # Use the model file saved in the previous step.
                model_file = self.results.get(visit_id_int, {}).get('model_file', os.path.join(visit_data['dirs'], f"mt_{visit_id}.pkl"))
                self.reload_model(model_file)
            
            # Compare with original STL models.
            if "compare_original" in computations:
                visit_data = self.visits_data[visit_id]['data']
                comparison_settings = self.visits_data[visit_id].get('comparison', None)
                if comparison_settings:
                    if 'upper' in comparison_settings:
                        upper_params = comparison_settings['upper']
                        if 'truth_points_file' not in upper_params and 'truth-upper' in visit_data:
                            upper_params['truth_points_file'] = visit_data['truth-upper']
                        self.analyzer.compare_with_original(
                            visit_data['dirs'],
                            visit_data['stl-upper'],
                            self.results[visit_id_int]['final_refined_teeth'] if visit_id_int in self.results else {},
                            region="upper",
                            **upper_params
                        )
                    else:
                        self._default_compare(visit_data, 
                                            self.results[visit_id_int]['final_refined_teeth'] if visit_id_int in self.results else {},
                                            scale_factor=0.0126)
                    
                    if 'lower' in comparison_settings:
                        lower_params = comparison_settings['lower']
                        if 'truth_points_file' not in lower_params and 'truth-lower' in visit_data:
                            lower_params['truth_points_file'] = visit_data['truth-lower']
                        self.analyzer.compare_with_original(
                            visit_data['dirs'],
                            visit_data['stl-lower'],
                            self.results[visit_id_int]['final_refined_teeth'] if visit_id_int in self.results else {},
                            region="lower",
                            **lower_params
                        )
                else:
                    self._default_compare(visit_data, 
                                        self.results[visit_id_int]['final_refined_teeth'] if visit_id_int in self.results else {},
                                        scale_factor=0.0126)
        
        # Step 4: If more than one visit, compare reconstructions between visits.
        if "compare_visits" in computations and len(visits_to_process) >= 2:
            reconstruction_comparison_settings = {}
            for i, vid in enumerate(visits_to_process[:2]):
                visit_key = f"visit{i+1}"
                vid_str = str(vid)
                if 'comparison' in self.visits_data[vid_str] and 'upper' in self.visits_data[vid_str]['comparison']:
                    upper_params = self.visits_data[vid_str]['comparison']['upper']
                    reconstruction_comparison_settings[visit_key] = {
                        "rotation_angle_points_x": upper_params.get('rotation_angle_points_x', 0),
                        "rotation_angle_points_y": upper_params.get('rotation_angle_points_y', 0),
                        "rotation_angle_points_z": upper_params.get('rotation_angle_points_z', 0)
                    }
            int_visits = [int(v) for v in visits_to_process]
            self.compare_visits(int_visits, reconstruction_comparison_settings)

    
    def reload_model(self, model_path: str, debug: bool = True):
        """
        Reload the saved model file and, if requested, visualize the 3D reconstruction.
        
        Args:
            model_path: Path to the saved model file.
            debug: If True, visualize the reconstruction.
            
        Returns:
            Tuple with the analyzer, loaded cameras, teeth, and alpha value.
        """
        analyzer = MultiViewAnalyzer()
        loaded_cameras, loaded_teeth, alpha = analyzer.load_model(model_path)
        if debug:
            analyzer.plotter.visualize_reconstruction_3d(
                loaded_cameras,
                global_points=None,
                final_refined_teeth=loaded_teeth,
                alpha=alpha,
                camera_scale=0.1
            )
        return analyzer, loaded_cameras, loaded_teeth, alpha
    
    def compare_visits(self, visits: List[int], comparison_settings: Optional[Dict[str, Any]] = None):
        """
        Compare reconstructions from two visits.
        
        Args:
            visits: List of visit IDs (at least 2) to compare.
            comparison_settings: Optional settings for the comparison.
        """
        if len(visits) < 2:
            print("Need at least 2 visits to compare")
            return
        print(f"\n{'='*50}")
        print(f"Comparing Visits: {visits[0]} vs {visits[1]}")
        print(f"{'='*50}")
        if self.results:
            model_file1 = self.results[visits[0]]['model_file']
            model_file2 = self.results[visits[1]]['model_file']
        else:
            model_file1 = os.path.join(self.visits_data[str(visits[0])]['data']['dirs'], f"mt_{visits[0]}.pkl")
            model_file2 = os.path.join(self.visits_data[str(visits[1])]['data']['dirs'], f"mt_{visits[1]}.pkl")
        self.compare_visit_reconstructions(model_file1, model_file2, visits[0], visits[1], comparison_settings)
    
    def compare_visit_reconstructions(self, model_file1: str, model_file2: str, visit1_id: int, visit2_id: int,
                                      comparison_settings: Optional[Dict[str, Any]] = None):
        """
        Compare reconstructions between two different visits.
        
        Args:
            model_file1: Path to the first visit's model file.
            model_file2: Path to the second visit's model file.
            visit1_id: Visit ID for the first model.
            visit2_id: Visit ID for the second model.
            comparison_settings: Optional settings for the comparison.
        """
        print(f"\n{'='*50}")
        print(f"Comparing Reconstructions: Visit {visit1_id} vs Visit {visit2_id}")
        print(f"{'='*50}")
        analyzer = MultiViewAnalyzer()
        print(f"\nLoading model from {model_file1}...")
        _, final_refined_teeth1, _ = analyzer.load_model(model_file1)
        print(f"Loading model from {model_file2}...")
        _, final_refined_teeth2, _ = analyzer.load_model(model_file2)
        
        if comparison_settings is None:
            comparison_settings = {
                "visit1": {"rotation_angle_points_x": 0, "rotation_angle_points_y": 0, "rotation_angle_points_z": 0},
                "visit2": {"rotation_angle_points_x": 0, "rotation_angle_points_y": 0, "rotation_angle_points_z": 0}
            }
        print("\nComparing upper jaw reconstructions (centroids only)...")
        analyzer.compare_reconstructions(
            final_refined_teeth1,
            final_refined_teeth2,
            comparison_settings=comparison_settings,
            region="upper",
            visit1_name=f"Visit {visit1_id}",
            visit2_name=f"Visit {visit2_id}"
        )
        print("\nComparing lower jaw reconstructions (centroids only)...")
        analyzer.compare_reconstructions(
            final_refined_teeth1,
            final_refined_teeth2,
            comparison_settings=comparison_settings,
            region="lower",
            visit1_name=f"Visit {visit1_id}",
            visit2_name=f"Visit {visit2_id}"
        )
        print("\nComparing upper jaw reconstructions with all points...")
        analyzer.compare_reconstructions_all(
            final_refined_teeth1,
            final_refined_teeth2,
            comparison_settings=comparison_settings,
            region="upper",
            visit1_name=f"Visit {visit1_id}",
            visit2_name=f"Visit {visit2_id}"
        )
        print("\nComparing lower jaw reconstructions with all points...")
        analyzer.compare_reconstructions_all(
            final_refined_teeth1,
            final_refined_teeth2,
            comparison_settings=comparison_settings,
            region="lower",
            visit1_name=f"Visit {visit1_id}",
            visit2_name=f"Visit {visit2_id}"
        )
    
    def process_visit(self, visit_id: Union[str, int], params: Dict[str, Any]):
        """
        Process a single visit with parameter adjustments.
        
        This method loads the saved model for the given visit and then adjusts
        the comparison parameters for both the upper and lower jaws.
        
        Args:
            visit_id: The visit ID to process.
            params: Dictionary with adjustment parameters for 'upper' and 'lower' jaws.
        """
        if isinstance(visit_id, str):
            visit_id = int(visit_id)
        print(f"\n{'='*50}")
        print(f"Processing Visit {visit_id} with parameter adjustment")
        print(f"{'='*50}")
        model_file = os.path.join(self.visits_data[str(visit_id)]['data']['dirs'], f"mt_{visit_id}.pkl")
        visit_data = self.visits_data[str(visit_id)]['data']
        if params is None:
            raise ValueError("Params are not provided")
        # Adjust for upper jaw.
        self._adjust_comparison_parameters(model_file, visit_data, region="upper", **params['upper'])
        # Adjust for lower jaw.
        self._adjust_comparison_parameters(model_file, visit_data, region="lower", **params['lower'])
        
    def test_reconstruction(self, visit_id: Union[str, int], output_file: str, 
                           upper_lower_weight: float = 1.0,
                           camera_translation: bool = True,
                           camera_rotations: Dict[str, float] = None,
                           use_arch_constraints: bool = True,
                           use_only_centroids: bool = False,
                           use_alpha: bool = True):
        import os
        import csv
        """
        Test the accuracy of reconstruction with different options.
        
        This method performs the reconstruction with the specified options, aligns the
        reconstructed 3D points to the truth 3D points, and calculates the distances.
        The results are appended to an external file. Both unscaled distances and scaled
        distances in millimeters are calculated using the distanceAB11 value from the config.
        
        Args:
            visit_id: The visit ID to process.
            output_file: Path to the file where results will be appended.
            upper_lower_weight: Weight given to the upper vs lower jaw during triangulation.
            camera_translation: Whether to optimize camera translations during triangulation.
            camera_rotations: Dictionary specifying rotation weights for each view.
            use_arch_constraints: Whether to use arch constraints during triangulation.
            use_only_centroids: Whether to use only centroids for triangulation.
            use_alpha: Whether to use the alpha parameter for triangulation.
        """
        if isinstance(visit_id, str):
            visit_id = int(visit_id)
            
        # Set default camera rotations if not provided
        if camera_rotations is None:
            camera_rotations = {
                'frontal': 0.0,
                'upper': 0.0,
                'lower': 0.0,
                'left': 1.0,
                'right': 1.0
            }
            
        print(f"\n{'='*50}")
        print(f"Testing reconstruction for Visit {visit_id} with custom options")
        print(f"{'='*50}")
        
        # Get visit data
        visit_data = self.visits_data[str(visit_id)]['data']
        
        # 1. Perform the reconstruction with the specified options
        refined_cameras, final_refined_teeth, alpha = self.analyzer.triangulation(
            visit_id=visit_id,
            active_views=self.view_order,
            debug=True,
            upper_lower_weight=upper_lower_weight,
            camera_translation=camera_translation,
            camera_rotations=camera_rotations,
            use_arch_constraints=use_arch_constraints,
            use_only_centroids=use_only_centroids,
            use_alpha=use_alpha
        )
        
        # Print information about the reconstructed points
        print(f"\nReconstructed {len(final_refined_teeth)} points:")
        centroid_count = sum(1 for k in final_refined_teeth.keys() if 
                           isinstance(k, int) or 
                           (isinstance(k, tuple) and k[1] == 'centroid'))
        print(f"  Centroids: {centroid_count}")
        print(f"  Other points: {len(final_refined_teeth) - centroid_count}")
        
        # 2. Load truth points for both upper and lower jaws
        truth_points_upper = None
        truth_points_lower = None
        
        if 'truth-upper' in visit_data:
            truth_file_path_upper = os.path.join(visit_data['dirs'], visit_data['truth-upper'])
            truth_points_upper = self.analyzer.load_truth_points(truth_file_path_upper)
            
        if 'truth-lower' in visit_data:
            truth_file_path_lower = os.path.join(visit_data['dirs'], visit_data['truth-lower'])
            truth_points_lower = self.analyzer.load_truth_points(truth_file_path_lower)
        
        # 3. Prepare data for weighted alignment and distance calculation
        from multiview_analyzer.utility import is_upper_tooth, is_lower_tooth
        
        # Filter reconstructed teeth by region
        upper_teeth = {}
        lower_teeth = {}
        
        for key, tooth_data in final_refined_teeth.items():
            # Extract tooth number for classification
            tooth_num = key
            if isinstance(tooth_num, tuple) and len(tooth_num) == 2:
                tooth_num = tooth_num[0]  # Use the tooth number part
            
            if isinstance(tooth_num, str):
                try:
                    tooth_num = int(tooth_num)
                except ValueError:
                    continue
                    
            # Add to appropriate jaw based on tooth number
            if is_upper_tooth(tooth_num):
                upper_teeth[key] = tooth_data
            elif is_lower_tooth(tooth_num):
                lower_teeth[key] = tooth_data
        
        # 4. Calculate weighted distances for upper jaw
        upper_distances = {}
        if truth_points_upper and upper_teeth:
            upper_distances = self._calculate_weighted_distances(upper_teeth, truth_points_upper)
            
        # 5. Calculate weighted distances for lower jaw
        lower_distances = {}
        if truth_points_lower and lower_teeth:
            lower_distances = self._calculate_weighted_distances(lower_teeth, truth_points_lower)
        
        # 6. Calculate mean distances
        upper_mean_centroid_distance = 0.0
        upper_mean_all_distance = 0.0
        lower_mean_centroid_distance = 0.0
        lower_mean_all_distance = 0.0
        
        if upper_distances:
            # Calculate mean for centroids only
            centroid_distances = [dist for key, dist in upper_distances.items() 
                                if isinstance(key, int) or 
                                (isinstance(key, tuple) and key[1] == 'centroid')]
            if centroid_distances:
                upper_mean_centroid_distance = sum(centroid_distances) / len(centroid_distances)
            
            # Calculate mean for all points
            if upper_distances:
                upper_mean_all_distance = sum(upper_distances.values()) / len(upper_distances)
        
        if lower_distances:
            # Calculate mean for centroids only
            centroid_distances = [dist for key, dist in lower_distances.items() 
                                if isinstance(key, int) or 
                                (isinstance(key, tuple) and key[1] == 'centroid')]
            if centroid_distances:
                lower_mean_centroid_distance = sum(centroid_distances) / len(centroid_distances)
            
            # Calculate mean for all points
            if lower_distances:
                lower_mean_all_distance = sum(lower_distances.values()) / len(lower_distances)
        
        # Get the distanceAB11 value from the config for scaling to mm
        visit_str_id = str(visit_id)
        distance_ab11_mm = self.visits_data.get(visit_str_id, {}).get('distanceAB11', None)
        
        # Calculate the scaling factor if distanceAB11 is available
        mm_scale_factor = None
        if distance_ab11_mm is not None:
            # Find the distance between A-11 and B-11 in the reconstructed model
            a11_point = None
            b11_point = None
            
            # Look for A-11 and B-11 in upper_teeth
            if (11, 'apex') in upper_teeth:
                a11_point = upper_teeth[(11, 'apex')]
            if (11, 'base') in upper_teeth:
                b11_point = upper_teeth[(11, 'base')]
                
            # Calculate the unscaled distance between A-11 and B-11
            unscaled_distance_ab11 = None
            if a11_point is not None and b11_point is not None:
                unscaled_distance_ab11 = np.linalg.norm(a11_point - b11_point)
                mm_scale_factor = distance_ab11_mm / unscaled_distance_ab11
                print(f"\nScaling factor: {mm_scale_factor:.6f} (based on tooth 11 height of {distance_ab11_mm} mm)")
            else:
                print("\nWarning: Could not find A-11 and B-11 points to calculate scaling factor")
        
        # Scale the distances to mm if scaling factor is available
        upper_mean_centroid_distance_mm = None
        upper_mean_all_distance_mm = None
        lower_mean_centroid_distance_mm = None
        lower_mean_all_distance_mm = None
        overall_mean_centroid_distance_mm = None
        overall_mean_all_distance_mm = None
        
        if mm_scale_factor is not None:
            upper_mean_centroid_distance_mm = upper_mean_centroid_distance * mm_scale_factor
            upper_mean_all_distance_mm = upper_mean_all_distance * mm_scale_factor
            lower_mean_centroid_distance_mm = lower_mean_centroid_distance * mm_scale_factor
            lower_mean_all_distance_mm = lower_mean_all_distance * mm_scale_factor
            overall_mean_centroid_distance_mm = (upper_mean_centroid_distance_mm + lower_mean_centroid_distance_mm) / 2
            overall_mean_all_distance_mm = (upper_mean_all_distance_mm + lower_mean_all_distance_mm) / 2
        
        # 7. Append results to the CSV file
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
        
        # Prepare the main result row
        main_result_row = {
            'Test ID': f"Visit{visit_id}_{len(upper_distances) + len(lower_distances)}_points",
            'Visit ID': visit_id,
            'Upper-Lower Weight': upper_lower_weight,
            'Camera Translation': camera_translation,
            'Camera Rotations': str(camera_rotations),
            'Use Arch Constraints': use_arch_constraints,
            'Use Only Centroids': use_only_centroids,
            'Use Alpha': use_alpha,
            'Alpha Value': alpha,
            'Upper Mean Centroid Distance': upper_mean_centroid_distance,
            'Upper Mean All Points Distance': upper_mean_all_distance,
            'Lower Mean Centroid Distance': lower_mean_centroid_distance,
            'Lower Mean All Points Distance': lower_mean_all_distance,
            'Overall Mean Centroid Distance': (upper_mean_centroid_distance + lower_mean_centroid_distance) / 2,
            'Overall Mean All Points Distance': (upper_mean_all_distance + lower_mean_all_distance) / 2,
            'Scaling Factor (mm)': mm_scale_factor,
            'Upper Mean Centroid Distance (mm)': upper_mean_centroid_distance_mm,
            'Upper Mean All Points Distance (mm)': upper_mean_all_distance_mm,
            'Lower Mean Centroid Distance (mm)': lower_mean_centroid_distance_mm,
            'Lower Mean All Points Distance (mm)': lower_mean_all_distance_mm,
            'Overall Mean Centroid Distance (mm)': overall_mean_centroid_distance_mm,
            'Overall Mean All Points Distance (mm)': overall_mean_all_distance_mm
        }
        
        # Prepare individual tooth distance rows
        individual_rows = []
        
        # Custom sorting function to handle mixed types (int and tuple)
        def sort_key(item):
            key = item[0]  # The tooth identifier
            if isinstance(key, tuple):
                return (key[0], key[1])  # Sort by tooth number, then by point type
            return (key, "")  # For integer keys, use empty string as second sort key
        
        # Add upper jaw individual distances
        for tooth, distance in sorted(upper_distances.items(), key=sort_key):
            tooth_id = tooth[0] if isinstance(tooth, tuple) else tooth
            point_type = tooth[1] if isinstance(tooth, tuple) else "centroid"
            
            # Calculate distance in mm if scaling factor is available
            distance_mm = distance * mm_scale_factor if mm_scale_factor is not None else None
            
            row = {
                'Test ID': main_result_row['Test ID'],
                'Jaw': 'Upper',
                'Tooth Number': tooth_id,
                'Point Type': point_type,
                'Distance': distance,
                'Distance (mm)': distance_mm
            }
            individual_rows.append(row)
        
        # Add lower jaw individual distances
        for tooth, distance in sorted(lower_distances.items(), key=sort_key):
            tooth_id = tooth[0] if isinstance(tooth, tuple) else tooth
            point_type = tooth[1] if isinstance(tooth, tuple) else "centroid"
            
            # Calculate distance in mm if scaling factor is available
            distance_mm = distance * mm_scale_factor if mm_scale_factor is not None else None
            
            row = {
                'Test ID': main_result_row['Test ID'],
                'Jaw': 'Lower',
                'Tooth Number': tooth_id,
                'Point Type': point_type,
                'Distance': distance,
                'Distance (mm)': distance_mm
            }
            individual_rows.append(row)
        
        # Write to the main results CSV file
        with open(output_file, 'a', newline='') as f:
            # Define fieldnames for the main results
            main_fieldnames = [
                'Test ID', 'Visit ID', 'Upper-Lower Weight', 'Camera Translation',
                'Camera Rotations', 'Use Arch Constraints', 'Use Only Centroids',
                'Use Alpha', 'Alpha Value', 'Upper Mean Centroid Distance',
                'Upper Mean All Points Distance', 'Lower Mean Centroid Distance',
                'Lower Mean All Points Distance', 'Overall Mean Centroid Distance',
                'Overall Mean All Points Distance', 'Scaling Factor (mm)',
                'Upper Mean Centroid Distance (mm)', 'Upper Mean All Points Distance (mm)',
                'Lower Mean Centroid Distance (mm)', 'Lower Mean All Points Distance (mm)',
                'Overall Mean Centroid Distance (mm)', 'Overall Mean All Points Distance (mm)'
            ]
            
            writer = csv.DictWriter(f, fieldnames=main_fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write the main result row
            writer.writerow(main_result_row)
        
        # Write individual tooth distances to a separate CSV file
        individual_file = output_file.replace('.csv', '_individual.csv')
        individual_file_exists = os.path.isfile(individual_file) and os.path.getsize(individual_file) > 0
        
        with open(individual_file, 'a', newline='') as f:
            # Define fieldnames for individual results
            individual_fieldnames = ['Test ID', 'Jaw', 'Tooth Number', 'Point Type', 'Distance', 'Distance (mm)']
            
            writer = csv.DictWriter(f, fieldnames=individual_fieldnames)
            
            # Write header if file doesn't exist
            if not individual_file_exists:
                writer.writeheader()
            
            # Write all individual rows
            writer.writerows(individual_rows)
        
        print(f"\nTest results written to {output_file} and {individual_file}")
        
        return refined_cameras, final_refined_teeth, alpha
    
    def _calculate_weighted_distances(self, reconstructed_teeth, truth_points):
        """
        Calculate weighted distances between reconstructed teeth and truth points.
        
        This method applies a weighted alignment that prioritizes front teeth (11,21,31,41)
        over back teeth (17,27,37,47) when aligning the reconstructed points to the truth points.
        
        Args:
            reconstructed_teeth: Dictionary of reconstructed teeth points.
            truth_points: Dictionary of truth points.
            
        Returns:
            Dictionary with distances between corresponding points after weighted alignment.
        """
        # Find common tooth numbers between reconstructed and truth points
        reconstructed_keys = set()
        for key in reconstructed_teeth.keys():
            if isinstance(key, int):
                reconstructed_keys.add(key)  # Add centroid key
                reconstructed_keys.add((key, 'centroid'))  # Also add as tuple format
            elif isinstance(key, tuple) and len(key) == 2:
                reconstructed_keys.add(key)  # Add (tooth_num, point_type) key
                if key[1] == 'centroid':
                    reconstructed_keys.add(key[0])  # Also add as simple integer if it's a centroid
        
        truth_keys = set(truth_points.keys())
        common_keys = reconstructed_keys.intersection(truth_keys)
        
        if not common_keys:
            print("No common tooth points found between reconstructed and truth points")
            return {}
        
        # Prepare points for alignment by separating them into centroid, apex, and base points
        recon_centroids = {}
        truth_centroids = {}
        recon_apex = {}
        truth_apex = {}
        recon_base = {}
        truth_base = {}
        
        # First, categorize all points by their type
        for key in common_keys:
            point_type = 'centroid'
            tooth_num = key
            
            # Determine the point type and tooth number
            if isinstance(key, tuple):
                tooth_num, point_type = key
            
            # Get the reconstructed point based on point type
            if point_type == 'centroid':
                if tooth_num in reconstructed_teeth:
                    recon_point = reconstructed_teeth[tooth_num]
                elif (tooth_num, 'centroid') in reconstructed_teeth:
                    recon_point = reconstructed_teeth[(tooth_num, 'centroid')]
                else:
                    continue
                recon_centroids[tooth_num] = recon_point
            elif point_type == 'apex':
                if (tooth_num, 'apex') in reconstructed_teeth:
                    recon_point = reconstructed_teeth[(tooth_num, 'apex')]
                    recon_apex[tooth_num] = recon_point
                else:
                    continue
            elif point_type == 'base':
                if (tooth_num, 'base') in reconstructed_teeth:
                    recon_point = reconstructed_teeth[(tooth_num, 'base')]
                    recon_base[tooth_num] = recon_point
                else:
                    continue
            
            # Get the truth point based on point type
            if point_type == 'centroid':
                if tooth_num in truth_points:
                    truth_point = truth_points[tooth_num]
                elif (tooth_num, 'centroid') in truth_points:
                    truth_point = truth_points[(tooth_num, 'centroid')]
                else:
                    continue
                truth_centroids[tooth_num] = truth_point
            elif point_type == 'apex':
                if (tooth_num, 'apex') in truth_points:
                    truth_point = truth_points[(tooth_num, 'apex')]
                    truth_apex[tooth_num] = truth_point
                else:
                    continue
            elif point_type == 'base':
                if (tooth_num, 'base') in truth_points:
                    truth_point = truth_points[(tooth_num, 'base')]
                    truth_base[tooth_num] = truth_point
                else:
                    continue
        
        # Create weighted alignment based on tooth position (front teeth have higher weight)
        # Convert dictionaries to lists for weighted alignment
        tooth_numbers = []
        recon_points = []
        truth_points_list = []
        weights = []
        
        # Process centroids first (they are more important)
        for tooth_num in sorted(set(recon_centroids.keys()).intersection(truth_centroids.keys())):
            tooth_numbers.append(tooth_num)
            recon_points.append(recon_centroids[tooth_num])
            truth_points_list.append(truth_centroids[tooth_num])
            
            # Calculate weight based on tooth position
            # Front teeth (11,21,31,41) have higher weight than back teeth (17,27,37,47)
            tooth_position = tooth_num % 10
            weight = max(1.0, 5.0 - 0.5 * (tooth_position - 1))  # Weight decreases as we move back
            weights.append(weight)
        
        # Process apex points
        for tooth_num in sorted(set(recon_apex.keys()).intersection(truth_apex.keys())):
            tooth_numbers.append((tooth_num, 'apex'))
            recon_points.append(recon_apex[tooth_num])
            truth_points_list.append(truth_apex[tooth_num])
            
            # Apex points get slightly lower weight than centroids
            tooth_position = tooth_num % 10
            weight = max(0.7, 3.5 - 0.5 * (tooth_position - 1)) 
            weights.append(weight)
        
        # Process base points
        for tooth_num in sorted(set(recon_base.keys()).intersection(truth_base.keys())):
            tooth_numbers.append((tooth_num, 'base'))
            recon_points.append(recon_base[tooth_num])
            truth_points_list.append(truth_base[tooth_num])
            
            # Base points get slightly lower weight than centroids
            tooth_position = tooth_num % 10
            weight = max(0.7, 3.5 - 0.5 * (tooth_position - 1))
            weights.append(weight)
        
        # Convert to numpy arrays
        recon_points_array = np.array(recon_points)
        truth_points_array = np.array(truth_points_list)
        weights_array = np.array(weights)
        
        # Perform weighted alignment
        # Center both point sets using weighted centroids
        weighted_centroid_recon = np.average(recon_points_array, axis=0, weights=weights_array)
        weighted_centroid_truth = np.average(truth_points_array, axis=0, weights=weights_array)
        
        recon_centered = recon_points_array - weighted_centroid_recon
        truth_centered = truth_points_array - weighted_centroid_truth
        
        # Calculate weighted covariance matrix
        H = np.zeros((3, 3))
        for i in range(len(recon_centered)):
            H += weights_array[i] * np.outer(recon_centered[i], truth_centered[i])
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Determine the rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Calculate weighted scaling factor
        numerator = 0
        denominator = 0
        for i in range(len(recon_centered)):
            numerator += weights_array[i] * np.dot(truth_centered[i], np.dot(R, recon_centered[i]))
            denominator += weights_array[i] * np.dot(recon_centered[i], recon_centered[i])
        
        s = numerator / denominator if denominator > 0 else 1.0
        
        # Calculate the translation
        t = weighted_centroid_truth - s * np.dot(weighted_centroid_recon, R.T)
        
        # Apply the transformation to all reconstructed points
        aligned_recon_points = s * np.dot(recon_points_array, R.T) + t
        
        # Calculate distances after alignment
        distances = np.sqrt(np.sum((truth_points_array - aligned_recon_points) ** 2, axis=1))
        
        # Create a dictionary with tooth numbers and distances
        result = {tooth: distance for tooth, distance in zip(tooth_numbers, distances)}
        
        return result