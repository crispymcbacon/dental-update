import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .plot import Plot
from .utility import LandmarkExtractor, build_projection_matrix, is_valid_tooth_number

class DataProcessor:
    def __init__(self):
        self.plotter = Plot()
        self.landmark_extractor = LandmarkExtractor()

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def load_data(self, visit):
        visit_data = {}
        for view, paths in visit.items():
            image = cv2.imread(paths['image'])
            mask = cv2.imread(paths['mask'], cv2.IMREAD_GRAYSCALE)
            json_data = self.load_json(paths['json'])
            
            # Horizontally flip left, right, upper, and lower images and their data
            if view in ['left', 'right', 'upper', 'lower']:
                # Flip the image
                image = cv2.flip(image, 1)  # 1 means horizontal flip
                
                # Flip the mask
                mask = cv2.flip(mask, 1)
                
                # Update JSON data for flipped images
                if 'teeth' in json_data:
                    img_width = image.shape[1]  # Get width of the image
                    for tooth in json_data['teeth']:
                        # Flip bounding box coordinates
                        if 'bbox' in tooth:
                            x_min, y_min, x_max, y_max = tooth['bbox']
                            tooth['bbox'] = [img_width - x_max, y_min, img_width - x_min, y_max]
                        
                        # Flip centroid coordinates
                        if 'centroid' in tooth and len(tooth['centroid']) == 2:
                            cx, cy = tooth['centroid']
                            tooth['centroid'] = [img_width - cx, cy]
                            
                        # Flip apex coordinates if present
                        if 'apex' in tooth and len(tooth['apex']) == 2:
                            ax, ay = tooth['apex']
                            tooth['apex'] = [img_width - ax, ay]
                            
                        # Flip base coordinates if present
                        if 'base' in tooth and len(tooth['base']) == 2:
                            bx, by = tooth['base']
                            tooth['base'] = [img_width - bx, by]
            
            visit_data[view] = {
                'image': image,
                'mask': mask,
                'json': json_data
            }
        return visit_data

    def _clamp_bbox(self, bbox, image_shape):
        h, w = image_shape[:2]
        x_min, y_min, x_max, y_max = map(int, bbox)
        return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)

    def _extract_roi(self, image, mask, bbox):
        x_min, y_min, x_max, y_max = self._clamp_bbox(bbox, image.shape)
        mask_region = mask[y_min:y_max, x_min:x_max]
        unique_values, counts = np.unique(mask_region, return_counts=True)
        non_zero = unique_values > 0
        if not np.any(non_zero):
            return None, None
        max_val = unique_values[non_zero][np.argmax(counts[non_zero])]
        tooth_mask = (mask == max_val).astype(np.uint8) * 255
        roi_mask = tooth_mask[y_min:y_max, x_min:x_max]
        roi_image = image[y_min:y_max, x_min:x_max]
        roi_extracted = cv2.bitwise_and(roi_image, roi_image, mask=roi_mask)
        return roi_extracted, (x_min, y_min)

    def preprocess(self, visit_data, visit_id=0, visualize=True):
        """
        For each view: extracts the teeth ROIs from the mask + bounding box
        and optionally shows them in a grid.
        """
        for view, data in visit_data.items():
            image = data['image']
            mask = data['mask']
            labels = data['json']
            data['rois'] = {}
            if visualize:
                teeth_count = len(labels.get('teeth', []))
                fig_cols = min(6, teeth_count) if teeth_count > 0 else 1
                fig_rows = (teeth_count + fig_cols - 1) // fig_cols if teeth_count > 0 else 1
                fig = plt.figure(figsize=(3 * fig_cols, 2 * fig_rows))
                fig.suptitle(f'Visit {visit_id} - {view.capitalize()} View Teeth ROIs')
                gs = gridspec.GridSpec(fig_rows, fig_cols, figure=fig)

            for idx, tooth in enumerate(labels.get('teeth', []), 1):
                tooth_number = tooth['tooth_number']
                bbox = tooth["bbox"]
                roi, offset = self._extract_roi(image, mask, bbox)
                if roi is None:
                    continue
                data['rois'][tooth_number] = {'roi': roi, 'offset': offset}
                if visualize:
                    ax = fig.add_subplot(gs[idx - 1])
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    ax.imshow(roi_rgb)
                    ax.set_title(f'Tooth {tooth_number}', fontsize=8)
                    ax.axis('off')

            if visualize:
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                plt.show()
        return visit_data

    def extract_landmarks(self, visit_data, visualize=False):
        """
        Extract per-tooth landmarks from each ROI (centroid, top, bottom, etc.).
        """
        result = {}
        for view, data in visit_data.items():
            result[view] = {
                'image': data['image'],
                'mask': data['mask'],
                'json': data['json'],
                'rois': data.get('rois', {}),
                'landmarks': {},
                'landmarks_norm': {}
            }
            rois = data.get('rois', {})
            view_landmarks = {}
            for tooth_number, roi_dict in rois.items():
                roi = roi_dict['roi']
                offset = roi_dict['offset']
                lm = self.landmark_extractor.extract_landmarks_from_roi(roi, offset=offset)
                if lm is not None:
                    view_landmarks[tooth_number] = lm
            result[view]['landmarks'] = view_landmarks
            # We won't rely on normalized landmarks for perspective math, 
            # but we keep the same function structure:
            result[view]['landmarks_norm'] = self.landmark_extractor.normalize_landmarks(view_landmarks)

            if visualize:
                print(f"\nView {view} - Normalized landmarks (for debugging only):")
                for tooth_number, lm_dict in view_landmarks.items():
                    print(f"  Tooth {tooth_number}: {lm_dict}")
        return result

    def get_raw_centroids(self, visit_data):
        """
        Gather the raw centroid, apex, and base points (pixel) from the JSON.
        We do NOT normalize to [-1,+1] anymore— we keep pixel coordinates.
        """
        all_centroids = {}
        for view, data in visit_data.items():
            json_data = data['json']
            image_h, image_w = data['image'].shape[:2]
            centroids_px = {}
            apex_px = {}
            base_px = {}
            for tooth in json_data.get('teeth', []):
                tooth_number = tooth.get('tooth_number')
                # Raw pixel coords from JSON
                if 'centroid' in tooth and len(tooth['centroid']) == 2:
                    cx, cy = tooth['centroid']
                    if cx > 0 and cy > 0 and is_valid_tooth_number(tooth_number):
                        centroids_px[tooth_number] = (float(cx), float(cy))
                        
                # Extract apex points if present
                if 'apex' in tooth and len(tooth['apex']) == 2:
                    ax, ay = tooth['apex']
                    if ax > 0 and ay > 0 and is_valid_tooth_number(tooth_number):
                        apex_px[(tooth_number, 'apex')] = (float(ax), float(ay))
                
                # Extract base points if present
                if 'base' in tooth and len(tooth['base']) == 2:
                    bx, by = tooth['base']
                    if bx > 0 and by > 0 and is_valid_tooth_number(tooth_number):
                        base_px[(tooth_number, 'base')] = (float(bx), float(by))

            # Store all points with their respective keys
            all_centroids[view] = {
                'centroids_px': centroids_px,
                'apex_px': apex_px,
                'base_px': base_px,
                'dims': (image_w, image_h)
            }
        return all_centroids

    def get_all_sublandmarks(self, visit_data):
        """
        Similar for sub-landmarks, but again we store direct pixel coords.
        """
        all_landmarks = {}
        for view, data in visit_data.items():
            image = data['image']
            h, w = image.shape[:2]
            tooth_landmarks = data.get('landmarks', {})
            pts_px = {}
            for tooth_number, lm_dict in tooth_landmarks.items():
                for sublm_name, (xval, yval) in lm_dict.items():
                    if xval > 0 and yval > 0:
                        try:
                            tnum_int = int(tooth_number)
                        except:
                            tnum_int = tooth_number
                        pts_px[(tnum_int, sublm_name)] = (float(xval), float(yval))

            all_landmarks[view] = {
                'points_px': pts_px,
                'dims': (w, h)
            }
        return all_landmarks

    def triangulation(self, 
                      visit_data, 
                      centroids, 
                      camera_config,
                      active_views=None, 
                      debug=False, 
                      upper_lower_weight=1.0,
                      camera_translation=False,
                      camera_rotations='none',
                      use_arch_constraints=False,
                      use_only_centroids=True,
                      use_alpha=True):
        """
        Triangulation with perspective (using camera intrinsics K) and optional alpha scale.
        """
        from scipy.optimize import least_squares

        # 1) Gather 2D measurements in pixel coordinates
        if use_only_centroids:
            all_points = {}
            if active_views:
                for v in active_views:
                    if v in centroids:
                        # store {tooth_number -> (u_px,v_px)}
                        all_points[v] = {"points_px": dict(centroids[v]["centroids_px"])}
            else:
                # all
                for v, info in centroids.items():
                    all_points[v] = {"points_px": dict(info["centroids_px"])}
        else:
            # Include centroids, apex, and base points
            all_points = {}
            if active_views:
                for v in active_views:
                    if v in centroids:
                        points_px = dict(centroids[v]["centroids_px"])
                        # Add apex and base points if they exist
                        if "apex_px" in centroids[v]:
                            points_px.update(centroids[v]["apex_px"])
                        if "base_px" in centroids[v]:
                            points_px.update(centroids[v]["base_px"])
                        all_points[v] = {"points_px": points_px}
            else:
                for v, info in centroids.items():
                    points_px = dict(info["centroids_px"])
                    # Add apex and base points if they exist
                    if "apex_px" in info:
                        points_px.update(info["apex_px"])
                    if "base_px" in info:
                        points_px.update(info["base_px"])
                    all_points[v] = {"points_px": points_px}

        # 2) Build correspondences: (view, tooth_key, u_px, v_px)
        corr_list = []
        key_counts = {}
        for v, info in all_points.items():
            for key, (u_px, v_px) in info["points_px"].items():
                corr_list.append((v, key, u_px, v_px))
                key_counts[key] = key_counts.get(key, 0) + 1

        valid_keys = {k for k, cnt in key_counts.items() if cnt >= 2}

        # 3) Identify active cameras
        if active_views is not None:
            active_cam_list = [v for v in active_views if v in camera_config]
        else:
            active_cam_list = list(camera_config.keys())

        if debug:
            print("[triangulation] # correspondences:", len(corr_list))
            print("valid_keys:", valid_keys)
            print("active_cam_list:", active_cam_list)
            print("upper_lower_weight:", upper_lower_weight)
            print("camera_translation:", camera_translation)
            print("camera_rotations:", camera_rotations)
            print("use_arch_constraints:", use_arch_constraints)
            print("use_only_centroids:", use_only_centroids)
            print("use_alpha:", use_alpha)

        # 4) Build parameter vector:
        #    param = [ alpha,
        #              per-cam translation offsets (3 each) if camera_translation=True,
        #              per-cam rotation deltas (3 each) if camera_rotations != 'none',
        #              3D coords for each valid key (3 each) ]
        def pack_params(alpha, cam_offsets, cam_angles, points_3d):
            p = []
            p.append(alpha)
            if camera_translation:
                for v in active_cam_list:
                    p.extend(cam_offsets[v])
            if camera_rotations != 'none':
                for v in active_cam_list:
                    p.extend(cam_angles[v])
            for k in sorted(valid_keys, key=str):
                p.extend(points_3d[k])
            return np.array(p, dtype=float)

        def unpack_params(p):
            alpha = p[0]
            idx = 1

            cam_offsets = {}
            if camera_translation:
                for v in active_cam_list:
                    cam_offsets[v] = p[idx:idx+3]
                    idx += 3
            else:
                for v in active_cam_list:
                    cam_offsets[v] = np.zeros(3)

            cam_angles = {}
            if camera_rotations != 'none':
                for v in active_cam_list:
                    cam_angles[v] = p[idx:idx+3]
                    idx += 3
            else:
                for v in active_cam_list:
                    cam_angles[v] = np.zeros(3)

            points_3d = {}
            for k in sorted(valid_keys, key=str):
                points_3d[k] = p[idx:idx+3]
                idx += 3

            return alpha, cam_offsets, cam_angles, points_3d

        alpha0 = 1.0
        cam_offsets0 = {v: np.zeros(3) for v in active_cam_list}
        cam_angles0  = {v: np.zeros(3) for v in active_cam_list}
        points_3d_0  = {k: np.zeros(3) for k in valid_keys}

        p0 = pack_params(alpha0, cam_offsets0, cam_angles0, points_3d_0)

        # 5) Define residual function
        def residuals_func(params):
            # Helper function to check if a key represents a centroid
            def is_centroid_k(key):
                # If key is an integer, it's a centroid
                if isinstance(key, int):
                    return True
                # If key is a tuple with 'centroid' as second element, it's a centroid
                if isinstance(key, tuple) and len(key) == 2 and key[1] == 'centroid':
                    return True
                # Otherwise it's an apex or base point
                return False
                
            alpha, cam_offsets, cam_angles, points_3d = unpack_params(params)

            residuals = []
            # Build each camera's refined extrinsics + intrinsics
            proj_mats = {}
            for v in active_cam_list:
                R_init = camera_config[v]["R"]
                t_init = camera_config[v]["t"]
                K      = camera_config[v].get("K", None)

                # rotation weight
                if isinstance(camera_rotations, dict) and v in camera_rotations:
                    rot_weight = camera_rotations[v]
                elif camera_rotations == 'none':
                    rot_weight = 0.0
                else:
                    rot_weight = 1.0

                angle_vec = rot_weight * cam_angles[v]
                R_delta, _ = cv2.Rodrigues(angle_vec)
                R_refined = R_init @ R_delta

                # If not using alpha, set alpha=1.0 effectively
                alpha_used = alpha if use_alpha else 1.0

                t_refined = alpha_used * t_init + cam_offsets[v]

                # Build projection matrix: K * [R_wc | t_wc]
                #  NOTE: build_projection_matrix handles the transformation
                P = build_projection_matrix(R_refined, t_refined, K=K,
                                            alpha=1.0)  # pass alpha=1.0 so we don't double-scale
                proj_mats[v] = (R_refined, t_refined, P)

            # A) Reprojection residuals
            for (view_name, key, u_meas_px, v_meas_px) in corr_list:
                if key not in valid_keys: 
                    continue
                if view_name not in proj_mats:
                    continue
                X_w = points_3d[key]
                Rr, tr, P = proj_mats[view_name]
                X_hom = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=float)
                x_cam = P @ X_hom
                if abs(x_cam[2]) < 1e-12:
                    continue
                pred_u = x_cam[0] / x_cam[2]
                pred_v = x_cam[1] / x_cam[2]

                # optional weighting for upper/lower views
                w_factor = upper_lower_weight if ("upper" in view_name or "lower" in view_name) else 1.0
                
                # Give more importance to centroids than apex/base points
                # If key is an integer or tuple with 'centroid', it's a centroid point
                point_weight = 1.0
                if not is_centroid_k(key):
                    # Reduce weight for non-centroid points (apex and base)
                    point_weight = 0.3  # This gives centroids ~3x more importance
                
                # Apply both weightings
                total_weight = w_factor * point_weight
                
                residuals.append(total_weight * (pred_u - u_meas_px))
                residuals.append(total_weight * (pred_v - v_meas_px))

            # B) Optional arch constraints (distance-based)
            if use_arch_constraints:
                def is_centroid_k(k):
                    # if k is int => tooth centroid
                    if isinstance(k, int):
                        return True
                    # if k is tuple => (tooth, sublm)
                    return (isinstance(k, tuple) and k[1] == "centroid")

                upper_keys = []
                lower_keys = []
                for k in valid_keys:
                    if not is_centroid_k(k):
                        continue
                    tn = k if isinstance(k, int) else k[0]
                    quadrant = tn // 10
                    if quadrant in [1,2]:
                        upper_keys.append(k)
                    elif quadrant in [3,4]:
                        lower_keys.append(k)

                upper_keys.sort(key=lambda x: x if isinstance(x, int) else x[0])
                lower_keys.sort(key=lambda x: x if isinstance(x, int) else x[0])

                d_nominal = 0.3
                anat_weight = 1.0
                for i in range(len(upper_keys)-1):
                    X1 = points_3d[upper_keys[i]]
                    X2 = points_3d[upper_keys[i+1]]
                    dist_err = np.linalg.norm(X1 - X2) - d_nominal
                    residuals.append(anat_weight * dist_err)
                for i in range(len(lower_keys)-1):
                    X1 = points_3d[lower_keys[i]]
                    X2 = points_3d[lower_keys[i+1]]
                    dist_err = np.linalg.norm(X1 - X2) - d_nominal
                    residuals.append(anat_weight * dist_err)

            return np.array(residuals, dtype=float)

        # 6) Run optimization
        res = least_squares(residuals_func, p0, verbose=2 if debug else 0, max_nfev=500)
        alpha_opt, cam_offsets_opt, cam_angles_opt, points_3d_opt = unpack_params(res.x)

        if debug:
            print("[triangulation] Optimization done.")
            print("alpha_opt =", alpha_opt)
            print("Final cost =", 0.5 * np.sum(res.fun**2))

        # 7) Build refined cameras
        refined_cameras = {}
        for v in active_cam_list:
            R_init = camera_config[v]["R"]
            t_init = camera_config[v]["t"]
            K      = camera_config[v].get("K", None)

            if isinstance(camera_rotations, dict) and v in camera_rotations:
                rot_weight = camera_rotations[v]
            elif camera_rotations == 'none':
                rot_weight = 0.0
            else:
                rot_weight = 1.0

            angle_vec = rot_weight * cam_angles_opt[v]
            R_delta, _ = cv2.Rodrigues(angle_vec)
            R_refined = R_init @ R_delta

            # apply alpha only if use_alpha=True
            alpha_used = alpha_opt if use_alpha else 1.0
            t_refined  = alpha_used * t_init + cam_offsets_opt[v]
            refined_cameras[v] = (R_refined, t_refined)

        # 8) Final 3D points
        final_refined_teeth = points_3d_opt

        return refined_cameras, final_refined_teeth, alpha_opt

    def calculate_point_distances(self, model1, model2, rotation_angles1=None, rotation_angles2=None):
        """
        Calculate distances between corresponding points in two reconstructions after finding
        optimal alignment to minimize distances.
        
        The optimal alignment includes:
        - Initial rotations of both models (if specified)
        - Finding the best scale factor to resize the second model
        - Computing the optimal rotation and translation of the second model
        
        Args:
            model1: First reconstruction model (dictionary of tooth_number -> point coordinates)
            model2: Second reconstruction model (dictionary of tooth_number -> point coordinates)
            rotation_angles1: Dictionary with rotation angles for model1 (x, y, z in degrees)
            rotation_angles2: Dictionary with rotation angles for model2 (x, y, z in degrees)
            
        Returns:
            Dictionary with tooth numbers as keys and distances as values,
            and the optimal transformation parameters (rotation_matrix, translation_vector, scale_factor)
        """
        # Default rotation angles (no rotation)
        if rotation_angles1 is None:
            rotation_angles1 = {'x': 0, 'y': 0, 'z': 0}
        if rotation_angles2 is None:
            rotation_angles2 = {'x': 0, 'y': 0, 'z': 0}
        
        # Find common tooth numbers in both models
        common_teeth = set(model1.keys()) & set(model2.keys())
        
        # Convert models to numpy arrays and match indices
        teeth_ids = list(common_teeth)
        points1 = np.array([model1[tooth] for tooth in teeth_ids])
        points2 = np.array([model2[tooth] for tooth in teeth_ids])
        
        # Apply initial rotations based on provided angles
        points1_rotated = self._apply_rotation(points1, rotation_angles1)
        points2_rotated = self._apply_rotation(points2, rotation_angles2)
        
        # Find optimal alignment to minimize distances
        R, t, s, aligned_points2 = self._find_optimal_alignment(points1_rotated, points2_rotated)
        
        # Calculate distances after alignment
        distances = np.sqrt(np.sum((points1_rotated - aligned_points2) ** 2, axis=1))
        
        # Create a dictionary with tooth numbers and distances
        result = {tooth: distance for tooth, distance in zip(teeth_ids, distances)}
        
        # Return distances and the transformation parameters
        transformation = {
            'rotation_matrix': R,
            'translation_vector': t,
            'scale_factor': s
        }
        
        return result, transformation
    
    def _apply_rotation(self, points, rotation_angles):
        """
        Apply rotation to points based on x, y, z angles.
        
        Args:
            points: Numpy array of 3D points
            rotation_angles: Dictionary with x, y, z rotation angles in degrees
            
        Returns:
            Rotated points as numpy array
        """
        # Center the points around their centroid
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Create rotation matrices
        theta_x = np.deg2rad(rotation_angles['x'])
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        
        theta_y = np.deg2rad(rotation_angles['y'])
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        
        theta_z = np.deg2rad(rotation_angles['z'])
        rot_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations in sequence (X, Y, Z)
        rotated_points = np.dot(centered_points, rot_x.T)
        rotated_points = np.dot(rotated_points, rot_y.T)
        rotated_points = np.dot(rotated_points, rot_z.T)
        
        # Return to original position
        return rotated_points + centroid
    
    def _find_optimal_alignment(self, points1, points2):
        """
        Find the optimal rigid transformation (rotation, translation, and scaling) to align points2 to points1.
        Uses Procrustes analysis to find the best alignment that minimizes the sum of squared distances.
        
        Args:
            points1: First set of points (reference)
            points2: Second set of points (to be aligned)
            
        Returns:
            R: Rotation matrix
            t: Translation vector
            s: Scale factor
            aligned_points2: Points2 after alignment
        """
        # Center both point sets around their respective centroids
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        points1_centered = points1 - centroid1
        points2_centered = points2 - centroid2
        
        # Calculate scaling factor
        norm1 = np.linalg.norm(points1_centered)
        norm2 = np.linalg.norm(points2_centered)
        
        if norm2 > 0:
            s = norm1 / norm2
        else:
            s = 1.0
            
        # Scale the second point set
        points2_scaled = points2_centered * s
        
        # Compute the covariance matrix
        H = np.dot(points2_scaled.T, points1_centered)
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Determine the rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Calculate the translation
        t = centroid1 - s * np.dot(centroid2, R.T)
        
        # Apply the transformation to points2
        aligned_points2 = s * np.dot(points2, R.T) + t
        
        return R, t, s, aligned_points2
