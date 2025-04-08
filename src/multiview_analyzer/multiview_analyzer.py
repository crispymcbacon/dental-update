import cv2
import numpy as np
import os
from stl import mesh
from .data_processor import DataProcessor
from .plot import Plot
from .utility import (
    is_valid_tooth_number, 
    is_upper_tooth, 
    is_lower_tooth,
    build_projection_matrix
)
from scipy.optimize import least_squares


class MultiViewAnalyzer:
    def __init__(self):
        self.all_visits = {}         # visit_id -> preprocessed data
        self.data_processor = DataProcessor()
        self.plotter = Plot()
        self.camera_config = {}      # user-specified camera extrinsics/intrinsics
        self.flip_upper_lower = False
        self.visit_centroids = {}    # output of get_raw_centroids(...) for each visit

    def add_visit(self, visit_paths, visit_id):
        # Load raw data
        self.all_visits[visit_id] = visit_paths
        data_loaded = self.data_processor.load_data(visit_paths)
        data_preprocessed = self.data_processor.preprocess(data_loaded, visit_id, visualize=False)
        data_preprocessed = self.data_processor.extract_landmarks(data_preprocessed, visualize=False)
        self.all_visits[visit_id] = data_preprocessed

        # Compute and store centroid data
        self.visit_centroids[visit_id] = self.data_processor.get_raw_centroids(data_preprocessed)

    def get_visit(self, visit_id):
        if visit_id not in self.all_visits:
            raise ValueError(f"No visit found with ID {visit_id}")
        return self.all_visits[visit_id]

    def triangulation(self, visit_id, active_views=None, debug=False,
                      upper_lower_weight=1.0,
                      camera_translation=False,
                      camera_rotations='none',
                      use_arch_constraints=False,
                      use_only_centroids=True,
                      use_alpha=True):
        """
        Orchestrates triangulation by calling the data_processor's method.
        """
        visit_data = self.get_visit(visit_id)
        centroids = self.visit_centroids[visit_id]

        refined_cameras, final_refined_teeth, alpha = self.data_processor.triangulation(
            visit_data=visit_data,
            centroids=centroids,
            camera_config=self.camera_config,
            active_views=active_views,
            debug=debug,
            upper_lower_weight=upper_lower_weight,
            camera_translation=camera_translation,
            camera_rotations=camera_rotations,
            use_arch_constraints=use_arch_constraints,
            use_only_centroids=use_only_centroids,
            use_alpha=use_alpha
        )
        return refined_cameras, final_refined_teeth, alpha

    def visualize_landmarks(self, visit_id=None, mode='centroids'):
        if visit_id is None:
            visits_to_visualize = self.all_visits
        else:
            if visit_id not in self.all_visits:
                raise ValueError(f"No visit found with ID {visit_id}")
            visits_to_visualize = {visit_id: self.all_visits[visit_id]}

        for v_id, visit_data in visits_to_visualize.items():
            self.plotter.visualize_landmarks(visit_data, visit_id=v_id, mode=mode)

    def visualize_camera_planes(self, visit_id, plane_distance=0.25, scale=0.3):
        """
        Visual debug: draws a small plane in front of each camera and the centroid points.
        This is mostly for quick visualization. It's still using the old 'normalized' approach
        or a simplified approach. You can keep it or adapt it as needed.
        """
        if visit_id not in self.all_visits:
            raise ValueError(f"No visit found with ID {visit_id}")
        if not self.camera_config:
            print("No camera config found. Please call set_camera_config() first.")
            return

        visit_data = self.all_visits[visit_id]
        all_centroids = self.visit_centroids[visit_id]  # raw pixel coords

        # We do a rough plane. We'll just treat the pixel coords in some scale for display:
        points_3d_per_view = {}
        plane_boxes = {}
        labels_per_view = {}

        for view, info in all_centroids.items():
            if view not in self.camera_config:
                points_3d_per_view[view] = np.zeros((0, 3))
                labels_per_view[view] = []
                continue

            R = self.camera_config[view]['R']
            t = self.camera_config[view]['t'].reshape(3)
            # We'll define the plane origin in front of the camera:
            plane_origin = t + plane_distance * R[:, 2]

            centroids_px = info['centroids_px']  # {tooth_number -> (u_px,v_px)}
            image_width, image_height = info['dims']  # Get image dimensions
            
            pts3d = []
            labels = []
            for tn, (u_px, v_px) in centroids_px.items():
                # Center the pixel coordinates relative to the image center
                u_centered = u_px - image_width / 2
                v_centered = v_px - image_height / 2
                
                # Project the centered coordinates onto the 3D plane
                offset_vec = (u_centered * R[:, 0] + v_centered * R[:, 1])
                pt_world = plane_origin + scale * 1e-3 * offset_vec  # pixel to "mm" factor is arbitrary here
                pts3d.append(pt_world)
                labels.append(str(tn))

            if pts3d:
                points_3d_per_view[view] = np.vstack(pts3d)
                labels_per_view[view] = labels
            else:
                points_3d_per_view[view] = np.zeros((0, 3))
                labels_per_view[view] = []

            # corners of plane
            corners_norm = np.array([
                [-1, -1],
                [ 1, -1],
                [ 1,  1],
                [-1,  1]
            ])
            corners_3d = []
            for (u_n, v_n) in corners_norm:
                offset_2d = (u_n * R[:, 0] + v_n * R[:, 1])
                pt_corner = plane_origin + scale * offset_2d
                corners_3d.append(pt_corner)
            plane_boxes[view] = np.vstack(corners_3d)

        self.plotter.visualize_cameras_with_points(self.camera_config, points_3d_per_view, plane_boxes, labels_per_view)

    def set_camera_config(self, camera_config, debug=False):
        self.camera_config = camera_config
        if debug:
            self.plotter.visualize_cameras(self.camera_config)

    def debug_2d_centroids(self, visit_id):
        visit_data = self.data_processor.load_data(self.all_visits[visit_id])
        self.plotter.debug_plot_2d_centroids(visit_data)
    
    def save_model(self, file_path, refined_cameras, final_refined_teeth, alpha):
        import pickle
        model = {
            'refined_cameras': refined_cameras,
            'final_refined_teeth': final_refined_teeth,
            'alpha': alpha
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        import pickle
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from", file_path)
        return model['refined_cameras'], model['final_refined_teeth'], model['alpha']

    def load_stl_model(self, stl_path):
        try:
            stl_model = mesh.Mesh.from_file(stl_path)
            return stl_model
        except Exception as e:
            raise IOError(f"Failed to load STL model from {stl_path}: {e}")

    def load_truth_points(self, truth_file_path):
        """
        Load truth points from a text file.
        
        Args:
            truth_file_path: Path to the truth points file
            
        Returns:
            Dictionary containing parsed points: {tooth_num: point}, {(tooth_num, 'apex'): point}, {(tooth_num, 'base'): point}
        """
        truth_points = {}
        
        try:
            with open(truth_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 4:
                        continue
                    
                    tooth_id = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    
                    # Check if it's an apex or base point
                    if tooth_id.startswith('A-'):
                        tooth_num = int(tooth_id[2:])
                        truth_points[(tooth_num, 'apex')] = np.array([x, y, z])
                    elif tooth_id.startswith('B-'):
                        tooth_num = int(tooth_id[2:])
                        truth_points[(tooth_num, 'base')] = np.array([x, y, z])
                    else:
                        # It's a centroid point
                        tooth_num = int(tooth_id)
                        truth_points[tooth_num] = np.array([x, y, z])
            
            return truth_points
            
        except Exception as e:
            raise IOError(f"Failed to load truth points from {truth_file_path}: {e}")

    def compare_with_original(self, dirs, stl_filename, triangulated_model,
                              rotation_angle=0, scale_factor=1.0, region=None,
                              offset_x=0, offset_y=0, offset_z=0, 
                              rotation_angle_points_x=0, rotation_angle_points_y=0, rotation_angle_points_z=0,
                              truth_points_file=None):
        full_stl_path = os.path.join(dirs, stl_filename)
        stl_model = self.load_stl_model(full_stl_path)
        
        # If the triangulated model is dict per tooth, you can separate upper/lower or pass entire
        # ...
        from .utility import is_upper_tooth, is_lower_tooth
        if region is not None and isinstance(triangulated_model, dict) and region in triangulated_model:
            teeth_subset = triangulated_model[region]
        else:
            if isinstance(triangulated_model, dict):
                upper_teeth = {}
                lower_teeth = {}
                
                # Process all points including centroids, apex, and base
                for key, tooth_data in triangulated_model.items():
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
                
                if region == 'upper':
                    teeth_subset = upper_teeth
                elif region == 'lower':
                    teeth_subset = lower_teeth
                else:
                    teeth_subset = triangulated_model
            else:
                teeth_subset = triangulated_model

        # If truth points file is provided, compare with truth points
        if truth_points_file:
            truth_points_path = os.path.join(dirs, truth_points_file)
            truth_points = self.load_truth_points(truth_points_path)
            
            # Filter truth points based on region if specified
            filtered_truth_points = {}
            for key, point in truth_points.items():
                tooth_num = key
                if isinstance(key, tuple) and len(key) == 2:
                    tooth_num = key[0]  # Use the tooth number part
                    
                if region == 'upper' and is_upper_tooth(tooth_num):
                    filtered_truth_points[key] = point
                elif region == 'lower' and is_lower_tooth(tooth_num):
                    filtered_truth_points[key] = point
                elif region is None:
                    filtered_truth_points[key] = point
            
            # Find common tooth numbers between reconstructed and truth points
            # We need to handle both simple integer keys and tuple keys (tooth_num, point_type)
            reconstructed_keys = set()
            for key in teeth_subset.keys():
                if isinstance(key, int):
                    reconstructed_keys.add(key)  # Add centroid key
                    reconstructed_keys.add((key, 'centroid'))  # Also add as tuple format
                elif isinstance(key, tuple) and len(key) == 2:
                    reconstructed_keys.add(key)  # Add (tooth_num, point_type) key
                    if key[1] == 'centroid':
                        reconstructed_keys.add(key[0])  # Also add as simple integer if it's a centroid
            
            truth_keys = set(filtered_truth_points.keys())
            common_keys = reconstructed_keys.intersection(truth_keys)
            
            if not common_keys:
                print(f"No common tooth points found between reconstructed and truth points for region: {region}")
                # Fall back to just showing the STL model and reconstructed points
                self.plotter.compare_models(stl_model, teeth_subset, rotation_angle, scale_factor, region=region,
                                        offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                                        rotation_angle_points_x=rotation_angle_points_x, 
                                        rotation_angle_points_y=rotation_angle_points_y, 
                                        rotation_angle_points_z=rotation_angle_points_z)
                return
            
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
                    if tooth_num in teeth_subset:
                        recon_point = teeth_subset[tooth_num]
                    elif (tooth_num, 'centroid') in teeth_subset:
                        recon_point = teeth_subset[(tooth_num, 'centroid')]
                    else:
                        continue
                    recon_centroids[tooth_num] = recon_point
                elif point_type == 'apex':
                    if (tooth_num, 'apex') in teeth_subset:
                        recon_point = teeth_subset[(tooth_num, 'apex')]
                        recon_apex[tooth_num] = recon_point
                    else:
                        continue
                elif point_type == 'base':
                    if (tooth_num, 'base') in teeth_subset:
                        recon_point = teeth_subset[(tooth_num, 'base')]
                        recon_base[tooth_num] = recon_point
                    else:
                        continue
                
                # Get the truth point based on point type
                if point_type == 'centroid':
                    if tooth_num in filtered_truth_points:
                        truth_point = filtered_truth_points[tooth_num]
                    elif (tooth_num, 'centroid') in filtered_truth_points:
                        truth_point = filtered_truth_points[(tooth_num, 'centroid')]
                    else:
                        continue
                    truth_centroids[tooth_num] = truth_point
                elif point_type == 'apex':
                    if (tooth_num, 'apex') in filtered_truth_points:
                        truth_point = filtered_truth_points[(tooth_num, 'apex')]
                        truth_apex[tooth_num] = truth_point
                    else:
                        continue
                elif point_type == 'base':
                    if (tooth_num, 'base') in filtered_truth_points:
                        truth_point = filtered_truth_points[(tooth_num, 'base')]
                        truth_base[tooth_num] = truth_point
                    else:
                        continue
            
            # Apply initial rotations to reconstructed points if specified
            rotation_angles = {
                'x': rotation_angle_points_x,
                'y': rotation_angle_points_y,
                'z': rotation_angle_points_z
            }
            
            # Calculate optimal alignment and distances using centroids
            distances, transformation = self.data_processor.calculate_point_distances(
                truth_centroids,  # Use truth centroids as reference
                recon_centroids,  # Align reconstructed centroids to truth centroids
                rotation_angles1=None,  # No rotation for truth points
                rotation_angles2=rotation_angles  # Apply rotation to reconstructed points
            )
            
            # Calculate mean error
            if distances:
                mean_error = sum(distances.values()) / len(distances)
                print(f"\nMean Euclidean distance between truth and reconstructed points: {mean_error:.4f} units")
                print(f"Individual distances:")
                for tooth, distance in sorted(distances.items()):
                    if isinstance(tooth, tuple):
                        # Handle tuple keys (tooth_number, point_type)
                        print(f"  Tooth {tooth[0]} ({tooth[1]}): {distance:.4f} units")
                    else:
                        # Handle integer keys (just tooth_number)
                        print(f"  Tooth {tooth}: {distance:.4f} units")
            else:
                print("No distances calculated - no common points found")
            
            # Visualize the comparison with truth points including apex and base points
            title = f"Comparison: Truth vs Reconstructed ({region.capitalize() if region else 'All'})"
            self.plotter.compare_reconstructions_all(
                truth_centroids,  # Truth centroids
                recon_centroids,  # Reconstructed centroids
                truth_apex,       # Truth apex points
                recon_apex,       # Reconstructed apex points
                truth_base,       # Truth base points
                recon_base,       # Reconstructed base points
                distances,
                title=title,
                region=region,
                model1_name="Truth Points",
                model2_name="Reconstructed Points",
                transformation=transformation
            )
            
            # Show only the STL model and reconstructed points (without truth points)
            self.plotter.compare_models(
                stl_model, 
                teeth_subset,
                rotation_angle, 
                scale_factor, 
                region=region,
                offset_x=offset_x, 
                offset_y=offset_y, 
                offset_z=offset_z,
                rotation_angle_points_x=rotation_angle_points_x, 
                rotation_angle_points_y=rotation_angle_points_y, 
                rotation_angle_points_z=rotation_angle_points_z
            )
        else:
            # If no truth points provided, just show the STL model and reconstructed points
            self.plotter.compare_models(stl_model, teeth_subset, rotation_angle, scale_factor, region=region,
                                    offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                                    rotation_angle_points_x=rotation_angle_points_x, 
                                    rotation_angle_points_y=rotation_angle_points_y, 
                                    rotation_angle_points_z=rotation_angle_points_z)

    def compare_reconstructions(self, reconstruction1, reconstruction2, comparison_settings=None, 
                               region=None, visit1_name="Visit 1", visit2_name="Visit 2"):
        """
        Compare two reconstructions from different visits with rotation adjustments and visualization.
        Performs optimal alignment to minimize distances between corresponding points.
        The optimal alignment includes rotation, translation, and scaling transformations.
        
        Args:
            reconstruction1: First reconstruction model from triangulation
            reconstruction2: Second reconstruction model from triangulation
            comparison_settings: Dictionary with rotation angles and other settings for comparison
            region: Region to compare ("upper", "lower", or None for all)
            visit1_name: Name for the first visit/reconstruction
            visit2_name: Name for the second visit/reconstruction
            
        Returns:
            Dictionary with distances between corresponding points after optimal alignment
        """
        # Default comparison settings if none provided
        if comparison_settings is None:
            comparison_settings = {
                "visit1": {
                    "rotation_angle_points_x": 0,
                    "rotation_angle_points_y": 0,
                    "rotation_angle_points_z": 0
                },
                "visit2": {
                    "rotation_angle_points_x": 0,
                    "rotation_angle_points_y": 0,
                    "rotation_angle_points_z": 0
                }
            }
        
        # Extract rotation angles for both visits
        rotation_angles1 = {
            'x': comparison_settings.get('visit1', {}).get('rotation_angle_points_x', 0),
            'y': comparison_settings.get('visit1', {}).get('rotation_angle_points_y', 0),
            'z': comparison_settings.get('visit1', {}).get('rotation_angle_points_z', 0)
        }
        
        rotation_angles2 = {
            'x': comparison_settings.get('visit2', {}).get('rotation_angle_points_x', 0),
            'y': comparison_settings.get('visit2', {}).get('rotation_angle_points_y', 0),
            'z': comparison_settings.get('visit2', {}).get('rotation_angle_points_z', 0)
        }
        
        # Import helper functions
        from .utility import is_upper_tooth, is_lower_tooth
        
        # Process upper and lower teeth separately if region is not specified
        if region is None:
            # Get results for upper teeth
            upper_results = self.compare_reconstructions(
                reconstruction1, reconstruction2,
                comparison_settings=comparison_settings,
                region="upper",
                visit1_name=visit1_name,
                visit2_name=visit2_name
            )
            
            # Get results for lower teeth
            lower_results = self.compare_reconstructions(
                reconstruction1, reconstruction2,
                comparison_settings=comparison_settings,
                region="lower",
                visit1_name=visit1_name,
                visit2_name=visit2_name
            )
            
            # Combine results
            return {**upper_results, **lower_results}
        
        # Filter reconstructions by region
        recon1_region = {}
        recon2_region = {}
        
        # Filter to only include centroid points for comparison
        filtered_recon1 = {}
        for key, point in reconstruction1.items():
            # If key is an integer, it's a centroid
            if isinstance(key, int):
                filtered_recon1[key] = point
            # If key is a tuple with first element as integer and second as 'centroid', include it
            elif isinstance(key, tuple) and len(key) == 2 and key[1] == 'centroid':
                filtered_recon1[key[0]] = point  # Use just the tooth number as key
        
        filtered_recon2 = {}
        for key, point in reconstruction2.items():
            # If key is an integer, it's a centroid
            if isinstance(key, int):
                filtered_recon2[key] = point
            # If key is a tuple with first element as integer and second as 'centroid', include it
            elif isinstance(key, tuple) and len(key) == 2 and key[1] == 'centroid':
                filtered_recon2[key[0]] = point  # Use just the tooth number as key
        
        for tooth_num, point in filtered_recon1.items():
            if isinstance(tooth_num, str):
                tooth_num = int(tooth_num)
            
            if (region == 'upper' and is_upper_tooth(tooth_num)) or \
               (region == 'lower' and is_lower_tooth(tooth_num)):
                recon1_region[tooth_num] = point
        
        for tooth_num, point in filtered_recon2.items():
            if isinstance(tooth_num, str):
                tooth_num = int(tooth_num)
                
            if (region == 'upper' and is_upper_tooth(tooth_num)) or \
               (region == 'lower' and is_lower_tooth(tooth_num)):
                recon2_region[tooth_num] = point
        
        # Check if we have points to compare
        if not recon1_region or not recon2_region:
            print(f"No {region} teeth found in one or both reconstructions")
            return {}
        
        # Calculate distances between corresponding points after optimal alignment
        distances, transformation = self.data_processor.calculate_point_distances(
            recon1_region, 
            recon2_region,
            rotation_angles1=rotation_angles1,
            rotation_angles2=rotation_angles2
        )
        
        # Print information about the optimal transformation
        print(f"\nOptimal transformation for {region} teeth:")
        print(f"Rotation matrix:\n{transformation['rotation_matrix']}")
        print(f"Translation vector: {transformation['translation_vector']}")
        print(f"Scale factor: {transformation['scale_factor']:.4f}")
        
        # Visualize the comparison
        title = f"Comparison: {visit1_name} vs {visit2_name}"
        self.plotter.compare_reconstructions(
            recon1_region, 
            recon2_region, 
            distances,
            title=title,
            region=region,
            model1_name=visit1_name,
            model2_name=visit2_name,
            transformation=transformation
        )
        
        return distances

    def compare_reconstructions_all(self, reconstruction1, reconstruction2, comparison_settings=None, 
                                   region=None, visit1_name="Visit 1", visit2_name="Visit 2"):
        """
        Compare two reconstructions from different visits with rotation adjustments and visualization.
        This method uses all points (centroids, apex, base) for visualization, connecting only the centroids.
        Performs optimal alignment to minimize distances between corresponding points.
        The optimal alignment includes rotation, translation, and scaling transformations.
        
        Args:
            reconstruction1: First reconstruction model from triangulation
            reconstruction2: Second reconstruction model from triangulation
            comparison_settings: Dictionary with rotation angles and other settings for comparison
            region: Region to compare ("upper", "lower", or None for all)
            visit1_name: Name for the first visit/reconstruction
            visit2_name: Name for the second visit/reconstruction
            
        Returns:
            Dictionary with distances between corresponding points after optimal alignment
        """
        # Default comparison settings if none provided
        if comparison_settings is None:
            comparison_settings = {
                "visit1": {
                    "rotation_angle_points_x": 0,
                    "rotation_angle_points_y": 0,
                    "rotation_angle_points_z": 0
                },
                "visit2": {
                    "rotation_angle_points_x": 0,
                    "rotation_angle_points_y": 0,
                    "rotation_angle_points_z": 0
                }
            }
        
        # Extract rotation angles for both visits
        rotation_angles1 = {
            'x': comparison_settings.get('visit1', {}).get('rotation_angle_points_x', 0),
            'y': comparison_settings.get('visit1', {}).get('rotation_angle_points_y', 0),
            'z': comparison_settings.get('visit1', {}).get('rotation_angle_points_z', 0)
        }
        
        rotation_angles2 = {
            'x': comparison_settings.get('visit2', {}).get('rotation_angle_points_x', 0),
            'y': comparison_settings.get('visit2', {}).get('rotation_angle_points_y', 0),
            'z': comparison_settings.get('visit2', {}).get('rotation_angle_points_z', 0)
        }
        
        # Import helper functions
        from .utility import is_upper_tooth, is_lower_tooth
        
        # Process upper and lower teeth separately if region is not specified
        if region is None:
            # Get results for upper teeth
            upper_results = self.compare_reconstructions_all(
                reconstruction1, reconstruction2,
                comparison_settings=comparison_settings,
                region="upper",
                visit1_name=visit1_name,
                visit2_name=visit2_name
            )
            
            # Get results for lower teeth
            lower_results = self.compare_reconstructions_all(
                reconstruction1, reconstruction2,
                comparison_settings=comparison_settings,
                region="lower",
                visit1_name=visit1_name,
                visit2_name=visit2_name
            )
            
            # Combine results
            return {**upper_results, **lower_results}
        
        # Extract points based on region
        recon1_region = {}
        recon2_region = {}
        
        # Filter by region and separate by point type (centroid, apex, base)
        recon1_centroids = {}
        recon1_apex = {}
        recon1_base = {}
        recon2_centroids = {}
        recon2_apex = {}
        recon2_base = {}
        
        # Process reconstruction1
        for key, point in reconstruction1.items():
            tooth_num = None
            point_type = None
            
            # Parse key to determine tooth number and point type
            if isinstance(key, int) or isinstance(key, str):
                # It's a centroid
                tooth_num = int(key) if isinstance(key, str) else key
                point_type = 'centroid'
            elif isinstance(key, tuple) and len(key) == 2:
                tooth_num = key[0]
                point_type = key[1]
            
            if tooth_num is None or point_type is None:
                continue
                
            # Filter by region
            if (region == 'upper' and is_upper_tooth(tooth_num)) or \
               (region == 'lower' and is_lower_tooth(tooth_num)):
                if point_type == 'centroid':
                    recon1_centroids[tooth_num] = point
                elif point_type == 'apex':
                    recon1_apex[tooth_num] = point
                elif point_type == 'base':
                    recon1_base[tooth_num] = point
        
        # Process reconstruction2
        for key, point in reconstruction2.items():
            tooth_num = None
            point_type = None
            
            # Parse key to determine tooth number and point type
            if isinstance(key, int) or isinstance(key, str):
                # It's a centroid
                tooth_num = int(key) if isinstance(key, str) else key
                point_type = 'centroid'
            elif isinstance(key, tuple) and len(key) == 2:
                tooth_num = key[0]
                point_type = key[1]
            
            if tooth_num is None or point_type is None:
                continue
                
            # Filter by region
            if (region == 'upper' and is_upper_tooth(tooth_num)) or \
               (region == 'lower' and is_lower_tooth(tooth_num)):
                if point_type == 'centroid':
                    recon2_centroids[tooth_num] = point
                elif point_type == 'apex':
                    recon2_apex[tooth_num] = point
                elif point_type == 'base':
                    recon2_base[tooth_num] = point
        
        # Combine all points for region filtering
        recon1_region = {**recon1_centroids, **{(k, 'apex'): v for k, v in recon1_apex.items()}, 
                         **{(k, 'base'): v for k, v in recon1_base.items()}}
        recon2_region = {**recon2_centroids, **{(k, 'apex'): v for k, v in recon2_apex.items()}, 
                         **{(k, 'base'): v for k, v in recon2_base.items()}}
        
        # Check if we have points to compare
        if not recon1_centroids or not recon2_centroids:
            print(f"No {region} teeth centroids found in one or both reconstructions")
            return {}
        
        # Calculate distances between centroid points after optimal alignment
        # We use centroid points for alignment
        distances, transformation = self.data_processor.calculate_point_distances(
            recon1_centroids, 
            recon2_centroids,
            rotation_angles1=rotation_angles1,
            rotation_angles2=rotation_angles2
        )
        
        # Print information about the optimal transformation
        print(f"\nOptimal transformation for {region} teeth:")
        print(f"Rotation matrix:\n{transformation['rotation_matrix']}")
        print(f"Translation vector: {transformation['translation_vector']}")
        print(f"Scale factor: {transformation['scale_factor']:.4f}")
        
        # Visualize the comparison with all points
        title = f"Comparison All Points: {visit1_name} vs {visit2_name}"
        self.plotter.compare_reconstructions_all(
            recon1_centroids, 
            recon2_centroids,
            recon1_apex,
            recon2_apex,
            recon1_base,
            recon2_base, 
            distances,
            title=title,
            region=region,
            model1_name=visit1_name,
            model2_name=visit2_name,
            transformation=transformation
        )
        
        return distances
