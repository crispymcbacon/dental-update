import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .utility import is_upper_tooth, is_lower_tooth

class Plot:
    def __init__(self):
        pass

    def visualize_landmarks(self, visit, visit_id='N/A', mode='all'):
        valid_views = [view for view, data in visit.items() if 'landmarks' in data]
        if not valid_views:
            return
            
        n_views = len(valid_views)
        n_cols = min(2, n_views)
        n_rows = (n_views + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_views == 1:
            axs = np.array([axs])
        axs = axs.flatten()
        
        for idx, view in enumerate(valid_views):
            data = visit[view]
            image = data['image']
            ax = axs[idx]
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Visit {visit_id} - {view.capitalize()} View')
            if mode == 'centroids':
                for tooth in data.get('json', {}).get('teeth', []):
                    if 'centroid' in tooth and len(tooth['centroid']) == 2:
                        point = tooth['centroid']
                        tooth_id = tooth['tooth_number']
                        ax.scatter(point[0], point[1], s=50)
                        ax.text(point[0], point[1], f'{tooth_id}', fontsize=8, color='red')
            else:
                for tooth_id, lm in data.get('landmarks', {}).items():
                    for key, point in lm.items():
                        marker_size = 50 if key == 'centroid' else 10
                        ax.scatter(point[0], point[1], s=marker_size)
                        ax.text(point[0], point[1], f'{tooth_id}-{key}', fontsize=8, color='red')
        for idx in range(n_views, len(axs)):
            axs[idx].axis('off')
        plt.tight_layout()
        plt.show()

    def _add_camera_to_fig(self, fig, view_name, R, t, axis_scale=0.2):
        center = t.flatten()
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers+text',
            marker=dict(size=8, color='red', symbol='diamond'),
            text=[view_name],
            textposition="top center",
            name=f'Camera {view_name}'
        ))
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            axis_end = center + R[:, i] * axis_scale
            fig.add_trace(go.Scatter3d(
                x=[center[0], axis_end[0]],
                y=[center[1], axis_end[1]],
                z=[center[2], axis_end[2]],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            ))

    def visualize_reconstruction_3d(self, global_cameras, global_points, final_refined_teeth, alpha=None, camera_scale=1.0):
        fig = go.Figure()
        UPPER_Z_OFFSET = 0.01
        
        positions = [t.flatten() for _, (_, t) in global_cameras.items()]
        if final_refined_teeth:
            positions_with_offset = []
            for key, pos in final_refined_teeth.items():
                tooth_id = key if isinstance(key, int) else key[0]
                pos_array = np.array(pos)
                if is_upper_tooth(tooth_id):
                    pos_array[2] += UPPER_Z_OFFSET
                positions_with_offset.append(pos_array)
            positions.extend(positions_with_offset)
        positions = np.array(positions)
        
        scene_center = np.mean(positions, axis=0) if positions.size else np.array([0,0,0])
        max_dist = np.max(np.linalg.norm(positions - scene_center, axis=1)) if positions.size else 1
        axis_scale = max_dist * (camera_scale * 0.5)
        
        for view, (R, t) in global_cameras.items():
            # Scale camera position towards scene center
            t_scaled = scene_center + camera_scale * (t.flatten() - scene_center)
            self._add_camera_to_fig(fig, view, R, t_scaled.reshape(3,1), axis_scale)
            
        if global_points is not None and global_points.shape[0] > 0:
            fig.add_trace(go.Scatter3d(
                x=global_points[:, 0],
                y=global_points[:, 1],
                z=global_points[:, 2],
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.6),
                name='3D Points'
            ))
        
        if final_refined_teeth:
            upper_xyz_centroid, upper_ids_centroid = [], []
            upper_xyz_apex, upper_ids_apex = [], []
            upper_xyz_base, upper_ids_base = [], []
            
            lower_xyz_centroid, lower_ids_centroid = [], []
            lower_xyz_apex, lower_ids_apex = [], []
            lower_xyz_base, lower_ids_base = [], []
            
            upper_centroids, lower_centroids = {}, {}
            
            # First, collect all centroids
            for key, pt in final_refined_teeth.items():
                if isinstance(key, int) or (isinstance(key, tuple) and key[1]=='centroid'):
                    tooth_id = key if isinstance(key, int) else key[0]
                    pt_adj = np.array(pt)
                    if is_upper_tooth(tooth_id):
                        pt_adj[2] += UPPER_Z_OFFSET
                        upper_centroids[tooth_id] = pt_adj
                    elif is_lower_tooth(tooth_id):
                        lower_centroids[tooth_id] = pt
                        
            # Then process all points including apex and base
            for key, pt in final_refined_teeth.items():
                if isinstance(key, int):
                    tooth_id, sublm = key, 'centroid'
                elif isinstance(key, tuple) and len(key)==2:
                    tooth_id, sublm = key
                else:
                    continue
                x, y, z = pt
                if is_upper_tooth(tooth_id):
                    z += UPPER_Z_OFFSET
                label = f"{tooth_id}" if sublm=='centroid' else f"{tooth_id}-{sublm}"
                
                if is_upper_tooth(tooth_id):
                    if sublm == 'centroid':
                        upper_xyz_centroid.append([x, y, z])
                        upper_ids_centroid.append(label)
                    elif sublm == 'apex':
                        upper_xyz_apex.append([x, y, z])
                        upper_ids_apex.append(label)
                    elif sublm == 'base':
                        upper_xyz_base.append([x, y, z])
                        upper_ids_base.append(label)
                elif is_lower_tooth(tooth_id):
                    if sublm == 'centroid':
                        lower_xyz_centroid.append([x, y, z])
                        lower_ids_centroid.append(label)
                    elif sublm == 'apex':
                        lower_xyz_apex.append([x, y, z])
                        lower_ids_apex.append(label)
                    elif sublm == 'base':
                        lower_xyz_base.append([x, y, z])
                        lower_ids_base.append(label)
                        
            def as_array(lst):
                return np.array(lst) if lst else np.zeros((0, 3))
            uc = as_array(upper_xyz_centroid)
            if uc.size:
                fig.add_trace(go.Scatter3d(
                    x=uc[:, 0], y=uc[:, 1], z=uc[:, 2],
                    mode='markers+text',
                    marker=dict(size=5, color='blue', symbol='circle'),
                    text=upper_ids_centroid,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Upper Teeth (centroid)'
                ))
            uo = as_array(upper_xyz_apex)
            if uo.size:
                fig.add_trace(go.Scatter3d(
                    x=uo[:, 0], y=uo[:, 1], z=uo[:, 2],
                    mode='markers+text',
                    marker=dict(size=2, color='blue', symbol='x'),
                    text=upper_ids_apex,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Upper Teeth (apex)'
                ))
                for i, (xyz, label) in enumerate(zip(upper_xyz_apex, upper_ids_apex)):
                    tooth_id = int(label.split('-')[0])
                    if tooth_id in upper_centroids:
                        centroid = upper_centroids[tooth_id]
                        fig.add_trace(go.Scatter3d(
                            x=[xyz[0], centroid[0]],
                            y=[xyz[1], centroid[1]],
                            z=[xyz[2], centroid[2]],
                            mode='lines',
                            line=dict(color='lightblue', width=2),
                            showlegend=False
                        ))
            ub = as_array(upper_xyz_base)
            if ub.size:
                fig.add_trace(go.Scatter3d(
                    x=ub[:, 0], y=ub[:, 1], z=ub[:, 2],
                    mode='markers+text',
                    marker=dict(size=2, color='blue', symbol='x'),
                    text=upper_ids_base,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Upper Teeth (base)'
                ))
                for i, (xyz, label) in enumerate(zip(upper_xyz_base, upper_ids_base)):
                    tooth_id = int(label.split('-')[0])
                    if tooth_id in upper_centroids:
                        centroid = upper_centroids[tooth_id]
                        fig.add_trace(go.Scatter3d(
                            x=[xyz[0], centroid[0]],
                            y=[xyz[1], centroid[1]],
                            z=[xyz[2], centroid[2]],
                            mode='lines',
                            line=dict(color='lightblue', width=2),
                            showlegend=False
                        ))
            lc = as_array(lower_xyz_centroid)
            if lc.size:
                fig.add_trace(go.Scatter3d(
                    x=lc[:, 0], y=lc[:, 1], z=lc[:, 2],
                    mode='markers+text',
                    marker=dict(size=5, color='orange', symbol='circle'),
                    text=lower_ids_centroid,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Lower Teeth (centroid)'
                ))
            lo = as_array(lower_xyz_apex)
            if lo.size:
                fig.add_trace(go.Scatter3d(
                    x=lo[:, 0], y=lo[:, 1], z=lo[:, 2],
                    mode='markers+text',
                    marker=dict(size=2, color='orange', symbol='x'),
                    text=lower_ids_apex,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Lower Teeth (apex)'
                ))
                for i, (xyz, label) in enumerate(zip(lower_xyz_apex, lower_ids_apex)):
                    tooth_id = int(label.split('-')[0])
                    if tooth_id in lower_centroids:
                        centroid = lower_centroids[tooth_id]
                        fig.add_trace(go.Scatter3d(
                            x=[xyz[0], centroid[0]],
                            y=[xyz[1], centroid[1]],
                            z=[xyz[2], centroid[2]],
                            mode='lines',
                            line=dict(color='peachpuff', width=2),
                            showlegend=False
                        ))
            lb = as_array(lower_xyz_base)
            if lb.size:
                fig.add_trace(go.Scatter3d(
                    x=lb[:, 0], y=lb[:, 1], z=lb[:, 2],
                    mode='markers+text',
                    marker=dict(size=2, color='orange', symbol='x'),
                    text=lower_ids_base,
                    textposition="top center",
                    textfont=dict(size=8),
                    name='Lower Teeth (base)'
                ))
                for i, (xyz, label) in enumerate(zip(lower_xyz_base, lower_ids_base)):
                    tooth_id = int(label.split('-')[0])
                    if tooth_id in lower_centroids:
                        centroid = lower_centroids[tooth_id]
                        fig.add_trace(go.Scatter3d(
                            x=[xyz[0], centroid[0]],
                            y=[xyz[1], centroid[1]],
                            z=[xyz[2], centroid[2]],
                            mode='lines',
                            line=dict(color='peachpuff', width=2),
                            showlegend=False
                        ))
        title_str = '3D Reconstruction with Camera Poses and Teeth'
        if alpha is not None:
            title_str += f' (alpha={alpha:.2f})'
        fig.update_layout(
            title=title_str,
            width=800,
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(eye=dict(x=-0.1, y=-0.8, z=0.25))
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                font=dict(size=8),
                itemsizing='constant',
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        fig.show()

    def visualize_cameras(self, camera_config):
        fig = go.Figure()
        for view_name, cam in camera_config.items():
            R = cam['R']
            t = cam['t'].reshape(-1)
            self._add_camera_to_fig(fig, view_name, R, t, axis_scale=1.0)
        fig.update_layout(
            title='User-Specified Camera Setup (Extrinsics)',
            width=600,
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.show()

    def visualize_cameras_with_points(self, camera_config, points_3d_per_view, plane_boxes=None, labels_per_view=None):
        fig = go.Figure()
        axis_len = 1.0
        for view_name, cam in camera_config.items():
            R = cam['R']
            t = cam['t'].reshape(-1)
            self._add_camera_to_fig(fig, view_name, R, t, axis_scale=axis_len)
        for view_name, pts3d in points_3d_per_view.items():
            if pts3d.shape[0] == 0:
                continue
            labels = labels_per_view.get(view_name, []) if labels_per_view else None
            fig.add_trace(go.Scatter3d(
                x=pts3d[:,0],
                y=pts3d[:,1],
                z=pts3d[:,2],
                mode='markers+text' if labels else 'markers',
                marker=dict(size=4),
                text=labels,
                textposition="top center",
                name=f'{view_name} points'
            ))
        if plane_boxes is not None:
            for view, corners in plane_boxes.items():
                corners_closed = np.vstack([corners, corners[0]])
                fig.add_trace(go.Scatter3d(
                    x=corners_closed[:,0],
                    y=corners_closed[:,1],
                    z=corners_closed[:,2],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name=f'{view} plane'
                ))
        fig.update_layout(
            title='Camera Setup with 2D Points (≥ 3 views) and Image Planes',
            width=600,
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            )
        )
        fig.show()

    def compare_models_with_truth(self, stl_model, triangulated_model, truth_points, distances=None, 
                              rotation_angle=0, scale_factor=0.01, region=None,
                              offset_x=0, offset_y=0, offset_z=0, rotation_angle_points_x=0, rotation_angle_points_y=0, rotation_angle_points_z=0):
        """
        Visualize and compare STL model, reconstructed points, and truth points.
        
        Args:
            stl_model: STL model mesh
            triangulated_model: Reconstructed points dictionary
            truth_points: Dictionary of truth points
            distances: Dictionary of distances between corresponding points
            rotation_angle: Rotation angle for STL model (degrees)
            scale_factor: Scale factor for STL model
            region: Region identifier (e.g., "upper", "lower")
            offset_x, offset_y, offset_z: Offset for STL model
            rotation_angle_points_x, rotation_angle_points_y, rotation_angle_points_z: Rotation angles for points (degrees)
        """
        from .utility import is_upper_tooth, is_lower_tooth
        
        # -- 1) Prepare STL Vectors --
        stl_vectors = stl_model.vectors.copy()  # (n_triangles, 3, 3)
        
        # Scale ONLY the STL model
        stl_vectors *= scale_factor
        print("Scaled STL bounding box:", stl_vectors.min(axis=(0,1)), stl_vectors.max(axis=(0,1)))
        
        # -- 2) Apply offset to STL model --
        offset = np.array([offset_x, offset_y, offset_z])
        stl_vectors = stl_vectors + offset
        
        # -- 3) Rotation (around Z-axis) --
        theta = np.deg2rad(rotation_angle)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,             1]
        ])
        stl_vectors = np.dot(stl_vectors.reshape(-1, 3), rot_mat.T).reshape(stl_vectors.shape)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # -- 4) Plot the STL model --
        # For each triangle in the STL
        x_stl, y_stl, z_stl = [], [], []
        i_stl = []
        idx = 0
        
        for triangle in stl_vectors:
            for vertex in triangle:
                x_stl.append(vertex[0])
                y_stl.append(vertex[1])
                z_stl.append(vertex[2])
            # Define indices for this triangle
            i_stl.extend([idx, idx+1, idx+2, idx])
            idx += 3
        
        # Add STL mesh
        fig.add_trace(go.Mesh3d(
            x=x_stl,
            y=y_stl,
            z=z_stl,
            i=i_stl[:-1:4],  # Every 4th element, excluding last
            j=i_stl[1:-1:4],  # Offset by 1
            k=i_stl[2:-1:4],  # Offset by 2
            opacity=0.3,
            color='cyan',
            name='STL Model'
        ))
        
        # -- 5) Separate triangulated points by type (centroid, apex, base) --
        # Initialize collections
        recon_centroids = {}
        recon_apex_points = {}
        recon_base_points = {}
        
        # Classify points
        for key, point in triangulated_model.items():
            if isinstance(key, int):
                # Integer keys are centroids
                recon_centroids[key] = np.array(point)
            elif isinstance(key, tuple) and len(key) == 2:
                tooth_num, point_type = key
                if point_type == 'centroid':
                    recon_centroids[tooth_num] = np.array(point)
                elif point_type == 'apex':
                    recon_apex_points[tooth_num] = np.array(point)
                elif point_type == 'base':
                    recon_base_points[tooth_num] = np.array(point)
        
        # -- 6) Separate truth points by type (centroid, apex, base) --
        truth_centroids = {}
        truth_apex_points = {}
        truth_base_points = {}
        
        # Classify truth points
        for key, point in truth_points.items():
            if isinstance(key, int):
                # Integer keys are centroids
                truth_centroids[key] = np.array(point)
            elif isinstance(key, tuple) and len(key) == 2:
                tooth_num, point_type = key
                if point_type == 'centroid':
                    truth_centroids[tooth_num] = np.array(point)
                elif point_type == 'apex':
                    truth_apex_points[tooth_num] = np.array(point)
                elif point_type == 'base':
                    truth_base_points[tooth_num] = np.array(point)
        
        # Convert to numpy arrays
        recon_centroid_keys = list(recon_centroids.keys())
        recon_centroid_points = np.array([recon_centroids[k] for k in recon_centroid_keys]) if recon_centroid_keys else np.zeros((0, 3))
        
        truth_centroid_keys = list(truth_centroids.keys())
        truth_centroid_points = np.array([truth_centroids[k] for k in truth_centroid_keys]) if truth_centroid_keys else np.zeros((0, 3))
        
        # -- 7) Center all points at origin --
        # Use the centroid of all points (both reconstructed and truth) for centering
        all_centroid_points = np.vstack([recon_centroid_points, truth_centroid_points]) if len(recon_centroid_points) > 0 and len(truth_centroid_points) > 0 else \
                            recon_centroid_points if len(recon_centroid_points) > 0 else \
                            truth_centroid_points if len(truth_centroid_points) > 0 else np.zeros((0, 3))
        
        if len(all_centroid_points) > 0:
            center = np.mean(all_centroid_points, axis=0)
            
            # Center the reconstructed points
            for k in recon_centroids:
                recon_centroids[k] = recon_centroids[k] - center
            for k in recon_apex_points:
                recon_apex_points[k] = recon_apex_points[k] - center
            for k in recon_base_points:
                recon_base_points[k] = recon_base_points[k] - center
            
            # Center the truth points
            for k in truth_centroids:
                truth_centroids[k] = truth_centroids[k] - center
            for k in truth_apex_points:
                truth_apex_points[k] = truth_apex_points[k] - center
            for k in truth_base_points:
                truth_base_points[k] = truth_base_points[k] - center
            
            # Update arrays after centering
            recon_centroid_points = np.array([recon_centroids[k] for k in recon_centroid_keys]) if recon_centroid_keys else np.zeros((0, 3))
            truth_centroid_points = np.array([truth_centroids[k] for k in truth_centroid_keys]) if truth_centroid_keys else np.zeros((0, 3))
        
        # -- 8) Apply rotations if requested --
        if rotation_angle_points_x != 0 or rotation_angle_points_y != 0 or rotation_angle_points_z != 0:
            # Create rotation matrices for each axis
            # X-axis rotation
            theta_x = np.deg2rad(rotation_angle_points_x)
            rot_mat_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            
            # Y-axis rotation
            theta_y = np.deg2rad(rotation_angle_points_y)
            rot_mat_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
            
            # Z-axis rotation
            theta_z = np.deg2rad(rotation_angle_points_z)
            rot_mat_z = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])
            
            # Apply rotations to reconstructed points
            for k in recon_centroids:
                point = recon_centroids[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                recon_centroids[k] = point
            
            for k in recon_apex_points:
                point = recon_apex_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                recon_apex_points[k] = point
            
            for k in recon_base_points:
                point = recon_base_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                recon_base_points[k] = point
            
            # Apply same rotations to truth points for consistency
            for k in truth_centroids:
                point = truth_centroids[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                truth_centroids[k] = point
            
            for k in truth_apex_points:
                point = truth_apex_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                truth_apex_points[k] = point
            
            for k in truth_base_points:
                point = truth_base_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                truth_base_points[k] = point
            
            # Update arrays after rotation
            recon_centroid_points = np.array([recon_centroids[k] for k in recon_centroid_keys]) if recon_centroid_keys else np.zeros((0, 3))
            truth_centroid_points = np.array([truth_centroids[k] for k in truth_centroid_keys]) if truth_centroid_keys else np.zeros((0, 3))

        # -- 9) Plot reconstructed centroid points --
        if len(recon_centroid_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=recon_centroid_points[:, 0],
                y=recon_centroid_points[:, 1],
                z=recon_centroid_points[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,  
                    color='red',
                    opacity=0.8,
                    symbol='circle'
                ),
                text=[str(k) for k in recon_centroid_keys],
                textposition="top center",
                name='Reconstructed Centroids'
            ))
        
        # -- 10) Plot truth centroid points --
        if len(truth_centroid_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=truth_centroid_points[:, 0],
                y=truth_centroid_points[:, 1],
                z=truth_centroid_points[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,  
                    color='blue',
                    opacity=0.8,
                    symbol='diamond'
                ),
                text=[str(k) for k in truth_centroid_keys],
                textposition="top center",
                name='Truth Centroids'
            ))
        
        # -- 11) Draw lines between corresponding points and show distances --
        if distances:
            for key, distance in distances.items():
                if isinstance(key, tuple) and len(key) == 2:
                    tooth_num, point_type = key
                    
                    # Get the reconstructed point
                    if point_type == 'centroid' and tooth_num in recon_centroids:
                        recon_point = recon_centroids[tooth_num]
                    elif point_type == 'apex' and tooth_num in recon_apex_points:
                        recon_point = recon_apex_points[tooth_num]
                    elif point_type == 'base' and tooth_num in recon_base_points:
                        recon_point = recon_base_points[tooth_num]
                    else:
                        continue
                    
                    # Get the truth point
                    if point_type == 'centroid' and tooth_num in truth_centroids:
                        truth_point = truth_centroids[tooth_num]
                    elif point_type == 'apex' and tooth_num in truth_apex_points:
                        truth_point = truth_apex_points[tooth_num]
                    elif point_type == 'base' and tooth_num in truth_base_points:
                        truth_point = truth_base_points[tooth_num]
                    else:
                        continue
                    
                    # Draw line connecting the points
                    fig.add_trace(go.Scatter3d(
                        x=[recon_point[0], truth_point[0]],
                        y=[recon_point[1], truth_point[1]],
                        z=[recon_point[2], truth_point[2]],
                        mode='lines+text',
                        line=dict(
                            color='purple',
                            width=2
                        ),
                        text=[f"{distance:.2f}"],
                        textposition="middle center",
                        showlegend=False
                    ))
                
                elif isinstance(key, int):
                    # Handle integer keys (centroids)
                    if key in recon_centroids and key in truth_centroids:
                        recon_point = recon_centroids[key]
                        truth_point = truth_centroids[key]
                        
                        # Draw line connecting the points
                        fig.add_trace(go.Scatter3d(
                            x=[recon_point[0], truth_point[0]],
                            y=[recon_point[1], truth_point[1]],
                            z=[recon_point[2], truth_point[2]],
                            mode='lines+text',
                            line=dict(
                                color='purple',
                                width=2
                            ),
                            text=[f"{distance:.2f}"],
                            textposition="middle center",
                            showlegend=False
                        ))
        
        # -- 12) Plot reconstructed apex points --
        if recon_apex_points:
            recon_apex_keys = list(recon_apex_points.keys())
            recon_apex_array = np.array([recon_apex_points[k] for k in recon_apex_keys])
            
            fig.add_trace(go.Scatter3d(
                x=recon_apex_array[:, 0],
                y=recon_apex_array[:, 1],
                z=recon_apex_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for apex points
                    color='green',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Reconstructed Apex Points'
            ))
            
            # Draw lines connecting apex points to their centroids
            for tooth_num in recon_apex_keys:
                if tooth_num in recon_centroids:
                    # Get the points
                    centroid = recon_centroids[tooth_num]
                    apex = recon_apex_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], apex[0]],
                        y=[centroid[1], apex[1]],
                        z=[centroid[2], apex[2]],
                        mode='lines',
                        line=dict(
                            color='green',
                            width=2
                        ),
                        showlegend=False
                    ))
        
        # -- 13) Plot truth apex points --
        if truth_apex_points:
            truth_apex_keys = list(truth_apex_points.keys())
            truth_apex_array = np.array([truth_apex_points[k] for k in truth_apex_keys])
            
            fig.add_trace(go.Scatter3d(
                x=truth_apex_array[:, 0],
                y=truth_apex_array[:, 1],
                z=truth_apex_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for apex points
                    color='darkblue',
                    opacity=0.8,
                    symbol='diamond'
                ),
                name='Truth Apex Points'
            ))
            
            # Draw lines connecting apex points to their centroids
            for tooth_num in truth_apex_keys:
                if tooth_num in truth_centroids:
                    # Get the points
                    centroid = truth_centroids[tooth_num]
                    apex = truth_apex_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], apex[0]],
                        y=[centroid[1], apex[1]],
                        z=[centroid[2], apex[2]],
                        mode='lines',
                        line=dict(
                            color='darkblue',
                            width=2
                        ),
                        showlegend=False
                    ))
        
        # -- 14) Plot reconstructed base points --
        if recon_base_points:
            recon_base_keys = list(recon_base_points.keys())
            recon_base_array = np.array([recon_base_points[k] for k in recon_base_keys])
            
            fig.add_trace(go.Scatter3d(
                x=recon_base_array[:, 0],
                y=recon_base_array[:, 1],
                z=recon_base_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for base points
                    color='orange',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Reconstructed Base Points'
            ))
            
            # Draw lines connecting base points to their centroids
            for tooth_num in recon_base_keys:
                if tooth_num in recon_centroids:
                    # Get the points
                    centroid = recon_centroids[tooth_num]
                    base = recon_base_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], base[0]],
                        y=[centroid[1], base[1]],
                        z=[centroid[2], base[2]],
                        mode='lines',
                        line=dict(
                            color='orange',
                            width=2
                        ),
                        showlegend=False
                    ))
        
        # -- 15) Plot truth base points --
        if truth_base_points:
            truth_base_keys = list(truth_base_points.keys())
            truth_base_array = np.array([truth_base_points[k] for k in truth_base_keys])
            
            fig.add_trace(go.Scatter3d(
                x=truth_base_array[:, 0],
                y=truth_base_array[:, 1],
                z=truth_base_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for base points
                    color='darkgreen',
                    opacity=0.8,
                    symbol='diamond'
                ),
                name='Truth Base Points'
            ))
            
            # Draw lines connecting base points to their centroids
            for tooth_num in truth_base_keys:
                if tooth_num in truth_centroids:
                    # Get the points
                    centroid = truth_centroids[tooth_num]
                    base = truth_base_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], base[0]],
                        y=[centroid[1], base[1]],
                        z=[centroid[2], base[2]],
                        mode='lines',
                        line=dict(
                            color='darkgreen',
                            width=2
                        ),
                        showlegend=False
                    ))

        # -- 16) Update layout --
        # Calculate bounds including all objects
        all_points = []
        if len(recon_centroid_points) > 0:
            all_points.append(recon_centroid_points)
        if len(truth_centroid_points) > 0:
            all_points.append(truth_centroid_points)
        if recon_apex_points:
            all_points.append(np.array([recon_apex_points[k] for k in recon_apex_points.keys()]))
        if truth_apex_points:
            all_points.append(np.array([truth_apex_points[k] for k in truth_apex_points.keys()]))
        if recon_base_points:
            all_points.append(np.array([recon_base_points[k] for k in recon_base_points.keys()]))
        if truth_base_points:
            all_points.append(np.array([truth_base_points[k] for k in truth_base_points.keys()]))
        
        # Add the STL vectors
        all_points.append(stl_vectors.reshape(-1, 3))
        
        # Combine all data points for calculating bounds
        combined_data = np.vstack(all_points) if all_points else np.array([[0, 0, 0]])
        x_min, y_min, z_min = combined_data.min(axis=0)
        x_max, y_max, z_max = combined_data.max(axis=0)
        
        # Set title based on region
        title = f"Comparison with Truth Points: {region.capitalize() if region else 'Jaw'}"
        
        # Calculate mean error if distances are provided
        if distances:
            mean_error = sum(distances.values()) / len(distances)
            title += f" (Mean Error: {mean_error:.4f})"
        
        fig.update_layout(
            title=title,
            width=800,
            height=600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max]),
                zaxis=dict(range=[z_min, z_max]),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        fig.show()
        
    def compare_models(self, stl_model, triangulated_model, rotation_angle=0, scale_factor=0.01, region=None,
                      offset_x=0, offset_y=0, offset_z=0, rotation_angle_points_x=0, rotation_angle_points_y=0, rotation_angle_points_z=0):
        from .utility import is_upper_tooth, is_lower_tooth
        
        # -- 1) Prepare STL Vectors --
        stl_vectors = stl_model.vectors.copy()  # (n_triangles, 3, 3)
        
        # Scale ONLY the STL model
        stl_vectors *= scale_factor
        print("Scaled STL bounding box:", stl_vectors.min(axis=(0,1)), stl_vectors.max(axis=(0,1)))
        
        # -- 2) Apply offset to STL model --
        offset = np.array([offset_x, offset_y, offset_z])
        stl_vectors = stl_vectors + offset
        
        # -- 3) Rotation (around Z-axis) --
        theta = np.deg2rad(rotation_angle)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,             1]
        ])
        stl_vectors = np.dot(stl_vectors.reshape(-1, 3), rot_mat.T).reshape(stl_vectors.shape)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # -- 4) Plot the STL model --
        # For each triangle in the STL
        x_stl, y_stl, z_stl = [], [], []
        i_stl = []
        idx = 0
        
        for triangle in stl_vectors:
            for vertex in triangle:
                x_stl.append(vertex[0])
                y_stl.append(vertex[1])
                z_stl.append(vertex[2])
            # Define indices for this triangle
            i_stl.extend([idx, idx+1, idx+2, idx])
            idx += 3
        
        # Add STL mesh
        fig.add_trace(go.Mesh3d(
            x=x_stl,
            y=y_stl,
            z=z_stl,
            i=i_stl[:-1:4],  # Every 4th element, excluding last
            j=i_stl[1:-1:4],  # Offset by 1
            k=i_stl[2:-1:4],  # Offset by 2
            opacity=0.3,
            color='cyan',
            name='STL Model'
        ))
        
        # -- 5) Separate triangulated points by type (centroid, apex, base) --
        # Initialize collections
        centroids = {}
        apex_points = {}
        base_points = {}
        
        # Classify points
        for key, point in triangulated_model.items():
            if isinstance(key, int):
                # Integer keys are centroids
                centroids[key] = np.array(point)
            elif isinstance(key, tuple) and len(key) == 2:
                tooth_num, point_type = key
                if point_type == 'centroid':
                    centroids[tooth_num] = np.array(point)
                elif point_type == 'apex':
                    apex_points[tooth_num] = np.array(point)
                elif point_type == 'base':
                    base_points[tooth_num] = np.array(point)
        
        # Convert to numpy arrays
        centroid_keys = list(centroids.keys())
        centroid_points = np.array([centroids[k] for k in centroid_keys])
        
        # -- 6) Center all points at origin --
        if len(centroid_points) > 0:
            center = np.mean(centroid_points, axis=0)
            
            # Center the points
            for k in centroids:
                centroids[k] = centroids[k] - center
            for k in apex_points:
                apex_points[k] = apex_points[k] - center
            for k in base_points:
                base_points[k] = base_points[k] - center
                
            # Update arrays after centering
            centroid_points = np.array([centroids[k] for k in centroid_keys])
        
        # -- 7) Apply rotations if requested --
        if rotation_angle_points_x != 0 or rotation_angle_points_y != 0 or rotation_angle_points_z != 0:
            # Create rotation matrices for each axis
            # X-axis rotation
            theta_x = np.deg2rad(rotation_angle_points_x)
            rot_mat_x = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ])
            
            # Y-axis rotation
            theta_y = np.deg2rad(rotation_angle_points_y)
            rot_mat_y = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ])
            
            # Z-axis rotation
            theta_z = np.deg2rad(rotation_angle_points_z)
            rot_mat_z = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ])
            
            # Apply rotations to all point collections
            # Centroids
            for k in centroids:
                point = centroids[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                centroids[k] = point
                
            # Apex points
            for k in apex_points:
                point = apex_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                apex_points[k] = point
                
            # Base points
            for k in base_points:
                point = base_points[k]
                point = np.dot(point, rot_mat_x.T)
                point = np.dot(point, rot_mat_y.T)
                point = np.dot(point, rot_mat_z.T)
                base_points[k] = point
                
            # Update arrays after rotation
            centroid_points = np.array([centroids[k] for k in centroid_keys])

        # -- 8) Plot centroid points --
        if len(centroid_points) > 0:
            fig.add_trace(go.Scatter3d(
                x=centroid_points[:, 0],
                y=centroid_points[:, 1],
                z=centroid_points[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,  
                    color='red',
                    opacity=0.8,
                    symbol='circle'
                ),
                text=[str(k) for k in centroid_keys],
                textposition="top center",
                name='Centroids'
            ))
        
        # -- 9) Plot apex points --
        if apex_points:
            apex_keys = list(apex_points.keys())
            apex_array = np.array([apex_points[k] for k in apex_keys])
            
            fig.add_trace(go.Scatter3d(
                x=apex_array[:, 0],
                y=apex_array[:, 1],
                z=apex_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for apex points
                    color='green',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Apex Points'
            ))
            
            # Draw lines connecting apex points to their centroids
            for tooth_num in apex_keys:
                if tooth_num in centroids:
                    # Get the points
                    centroid = centroids[tooth_num]
                    apex = apex_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], apex[0]],
                        y=[centroid[1], apex[1]],
                        z=[centroid[2], apex[2]],
                        mode='lines',
                        line=dict(
                            color='green',
                            width=2
                        ),
                        showlegend=False
                    ))
        
        # -- 10) Plot base points --
        if base_points:
            base_keys = list(base_points.keys())
            base_array = np.array([base_points[k] for k in base_keys])
            
            fig.add_trace(go.Scatter3d(
                x=base_array[:, 0],
                y=base_array[:, 1],
                z=base_array[:, 2],
                mode='markers',
                marker=dict(
                    size=4,  # Smaller size for base points
                    color='blue',
                    opacity=0.8,
                    symbol='circle'
                ),
                name='Base Points'
            ))
            
            # Draw lines connecting base points to their centroids
            for tooth_num in base_keys:
                if tooth_num in centroids:
                    # Get the points
                    centroid = centroids[tooth_num]
                    base = base_points[tooth_num]
                    
                    # Create line
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0], base[0]],
                        y=[centroid[1], base[1]],
                        z=[centroid[2], base[2]],
                        mode='lines',
                        line=dict(
                            color='blue',
                            width=2
                        ),
                        showlegend=False
                    ))

        # -- 11) Update layout --
        # Calculate bounds including all objects
        all_points = []
        if len(centroid_points) > 0:
            all_points.append(centroid_points)
        if apex_points:
            all_points.append(np.array([apex_points[k] for k in apex_points.keys()]))
        if base_points:
            all_points.append(np.array([base_points[k] for k in base_points.keys()]))
        
        # Add the STL vectors
        all_points.append(stl_vectors.reshape(-1, 3))
        
        # Combine all data points for calculating bounds
        combined_data = np.vstack(all_points) if all_points else np.array([[0, 0, 0]])
        x_min, y_min, z_min = combined_data.min(axis=0)
        x_max, y_max, z_max = combined_data.max(axis=0)
        
        # Set title based on region
        title = f"Comparison: {region.capitalize() if region else 'Jaw'}"
        
        fig.update_layout(
            title=title,
            width=800,
            height=600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max]),
                zaxis=dict(range=[z_min, z_max]),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        fig.show()

    def compare_reconstructions(self, model1, model2, distances, title="Model Comparison", 
                              region=None, model1_name="Model 1", model2_name="Model 2",
                              transformation=None):
        """
        Visualize and compare two 3D reconstructions after optimal alignment.
        
        Args:
            model1: First reconstruction model (dictionary of tooth_number -> point coordinates)
            model2: Second reconstruction model (dictionary of tooth_number -> point coordinates)
            distances: Dictionary of distances between corresponding points
            title: Plot title
            region: Region identifier (e.g., "upper", "lower")
            model1_name: Name for the first model (e.g., "Visit 1")
            model2_name: Name for the second model (e.g., "Visit 2")
            transformation: Dictionary containing the optimal transformation (rotation_matrix, translation_vector, scale_factor)
        """
        # Find common tooth numbers in both models
        common_teeth = set(model1.keys()) & set(model2.keys())
        
        # Convert models to arrays for visualization
        teeth_ids = list(common_teeth)
        points1 = np.array([model1[tooth] for tooth in teeth_ids])
        points2 = np.array([model2[tooth] for tooth in teeth_ids])
        
        # Apply transformation to align the second model to the first
        if transformation:
            R = transformation['rotation_matrix']
            t = transformation['translation_vector']
            s = transformation.get('scale_factor', 1.0)  # Get scale factor or default to 1.0
            
            # Apply the optimal alignment to the second model (scaling, rotation, and translation)
            aligned_points2 = s * np.dot(points2, R.T) + t
        else:
            # If no transformation provided, just center both models
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            points1 = points1 - centroid1
            points2 = points2 - centroid2
            aligned_points2 = points2  # No alignment applied
        
        # Create the figure
        fig = go.Figure()
        
        # Add the first model (blue)
        fig.add_trace(go.Scatter3d(
            x=points1[:, 0],
            y=points1[:, 1],
            z=points1[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.8,
                symbol='circle'
            ),
            text=[str(k) for k in teeth_ids],
            textposition="top center",
            name=model1_name
        ))
        
        # Add the second model (red) after transformation
        fig.add_trace(go.Scatter3d(
            x=aligned_points2[:, 0],
            y=aligned_points2[:, 1],
            z=aligned_points2[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                opacity=0.8,
                symbol='circle'
            ),
            text=[str(k) for k in teeth_ids],
            textposition="top center",
            name=model2_name
        ))
        
        # Add connections between corresponding points
        for i, tooth in enumerate(teeth_ids):
            fig.add_trace(go.Scatter3d(
                x=[points1[i, 0], aligned_points2[i, 0]],
                y=[points1[i, 1], aligned_points2[i, 1]],
                z=[points1[i, 2], aligned_points2[i, 2]],
                mode='lines',
                line=dict(
                    color='gray',
                    width=2,
                    dash='dot'
                ),
                showlegend=False
            ))
        
        # Update layout
        region_str = f" - {region.capitalize()}" if region else ""
        transform_info = ""
        if transformation and 'scale_factor' in transformation:
            transform_info = f" (Scale: {transformation['scale_factor']:.3f})"
            
        fig.update_layout(
            title=f"{title}{region_str} (After Optimal Alignment){transform_info}",
            width=800,
            height=600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # Display the plot
        fig.show()
        
        # Print distances
        print(f"\n{title}{region_str} - Point Distances (After Optimal Alignment):")
        if transformation and 'scale_factor' in transformation:
            print(f"Applied scale factor: {transformation['scale_factor']:.4f}")
        print("-" * 40)
        print(f"{'Tooth':<10} {'Distance':>10}")
        print("-" * 40)
        for tooth in sorted(distances.keys()):
            # Handle tuple keys by converting them to string
            tooth_str = str(tooth) if isinstance(tooth, tuple) else tooth
            print(f"{tooth_str:<10} {distances[tooth]:>10.4f}")
        print("-" * 40)
        print(f"Average: {sum(distances.values())/len(distances):>10.4f}")
        print("-" * 40)

    def compare_reconstructions_all(self, model1_centroids, model2_centroids, 
                                  model1_apex, model2_apex,
                                  model1_base, model2_base,
                                  distances, title="Model Comparison", 
                                  region=None, model1_name="Model 1", model2_name="Model 2",
                                  transformation=None):
        """
        Visualize and compare two 3D reconstructions after optimal alignment, including all points (centroids, apex, base).
        Only centroids are connected with lines between corresponding points.
        
        Args:
            model1_centroids: First model centroids (dictionary of tooth_number -> point coordinates)
            model2_centroids: Second model centroids (dictionary of tooth_number -> point coordinates)
            model1_apex: First model apex points (dictionary of tooth_number -> point coordinates)
            model2_apex: Second model apex points (dictionary of tooth_number -> point coordinates)
            model1_base: First model base points (dictionary of tooth_number -> point coordinates)
            model2_base: Second model base points (dictionary of tooth_number -> point coordinates)
            distances: Dictionary of distances between corresponding centroid points
            title: Plot title
            region: Region identifier (e.g., "upper", "lower")
            model1_name: Name for the first model (e.g., "Visit 1")
            model2_name: Name for the second model (e.g., "Visit 2")
            transformation: Dictionary containing the optimal transformation (rotation_matrix, translation_vector, scale_factor)
        """
        # Find common tooth numbers in both centroid models (used for connecting lines)
        common_centroid_teeth = set(model1_centroids.keys()) & set(model2_centroids.keys())
        
        # Convert centroid models to arrays for visualization
        teeth_ids_centroids = list(common_centroid_teeth)
        centroid_points1 = np.array([model1_centroids[tooth] for tooth in teeth_ids_centroids])
        centroid_points2 = np.array([model2_centroids[tooth] for tooth in teeth_ids_centroids])
        
        # Convert apex points to arrays
        common_apex_teeth = set(model1_apex.keys()) & set(model2_apex.keys())
        teeth_ids_apex = list(common_apex_teeth)
        apex_points1 = np.array([model1_apex[tooth] for tooth in teeth_ids_apex]) if teeth_ids_apex else np.empty((0, 3))
        apex_points2 = np.array([model2_apex[tooth] for tooth in teeth_ids_apex]) if teeth_ids_apex else np.empty((0, 3))
        
        # Convert base points to arrays
        common_base_teeth = set(model1_base.keys()) & set(model2_base.keys())
        teeth_ids_base = list(common_base_teeth)
        base_points1 = np.array([model1_base[tooth] for tooth in teeth_ids_base]) if teeth_ids_base else np.empty((0, 3))
        base_points2 = np.array([model2_base[tooth] for tooth in teeth_ids_base]) if teeth_ids_base else np.empty((0, 3))
        
        # Apply transformation to align the second model to the first
        if transformation:
            R = transformation['rotation_matrix']
            t = transformation['translation_vector']
            s = transformation.get('scale_factor', 1.0)  # Get scale factor or default to 1.0
            
            # Apply the optimal alignment to the second model centroids
            aligned_centroid_points2 = s * np.dot(centroid_points2, R.T) + t
            
            # Apply the same transformation to apex and base points
            aligned_apex_points2 = s * np.dot(apex_points2, R.T) + t if apex_points2.size > 0 else np.empty((0, 3))
            aligned_base_points2 = s * np.dot(base_points2, R.T) + t if base_points2.size > 0 else np.empty((0, 3))
        else:
            # If no transformation provided, just center all points
            all_points1 = np.vstack([centroid_points1, apex_points1, base_points1]) if apex_points1.size > 0 and base_points1.size > 0 else centroid_points1
            all_points2 = np.vstack([centroid_points2, apex_points2, base_points2]) if apex_points2.size > 0 and base_points2.size > 0 else centroid_points2
            
            centroid1 = np.mean(all_points1, axis=0)
            centroid2 = np.mean(all_points2, axis=0)
            
            centroid_points1 = centroid_points1 - centroid1
            centroid_points2 = centroid_points2 - centroid2
            apex_points1 = apex_points1 - centroid1 if apex_points1.size > 0 else np.empty((0, 3))
            apex_points2 = apex_points2 - centroid2 if apex_points2.size > 0 else np.empty((0, 3))
            base_points1 = base_points1 - centroid1 if base_points1.size > 0 else np.empty((0, 3))
            base_points2 = base_points2 - centroid2 if base_points2.size > 0 else np.empty((0, 3))
            
            aligned_centroid_points2 = centroid_points2
            aligned_apex_points2 = apex_points2
            aligned_base_points2 = base_points2
        
        # Create the figure
        fig = go.Figure()
        
        # Add the first model centroids (blue)
        fig.add_trace(go.Scatter3d(
            x=centroid_points1[:, 0],
            y=centroid_points1[:, 1],
            z=centroid_points1[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.8,
                symbol='circle'
            ),
            text=[str(tooth) for tooth in teeth_ids_centroids],
            textposition="top center",
            textfont=dict(size=8),
            name=f"{model1_name} Centroids"
        ))
        
        # Add the second model centroids (red) after transformation
        fig.add_trace(go.Scatter3d(
            x=aligned_centroid_points2[:, 0],
            y=aligned_centroid_points2[:, 1],
            z=aligned_centroid_points2[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                opacity=0.8,
                symbol='circle'
            ),
            text=[str(tooth) for tooth in teeth_ids_centroids],
            textposition="top center",
            textfont=dict(size=8),
            name=f"{model2_name} Centroids"
        ))
        
        # Add the first model apex points (cyan)
        if apex_points1.size > 0:
            fig.add_trace(go.Scatter3d(
                x=apex_points1[:, 0],
                y=apex_points1[:, 1],
                z=apex_points1[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color='cyan',
                    opacity=0.8,
                    symbol='diamond'
                ),
                text=[f"A-{tooth}" for tooth in teeth_ids_apex],
                textposition="top center",
                textfont=dict(size=7),
                name=f"{model1_name} Apex"
            ))
        
        # Add the second model apex points (magenta) after transformation
        if aligned_apex_points2.size > 0:
            fig.add_trace(go.Scatter3d(
                x=aligned_apex_points2[:, 0],
                y=aligned_apex_points2[:, 1],
                z=aligned_apex_points2[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color='magenta',
                    opacity=0.8,
                    symbol='diamond'
                ),
                text=[f"A-{tooth}" for tooth in teeth_ids_apex],
                textposition="top center",
                textfont=dict(size=7),
                name=f"{model2_name} Apex"
            ))
        
        # Add the first model base points (green)
        if base_points1.size > 0:
            fig.add_trace(go.Scatter3d(
                x=base_points1[:, 0],
                y=base_points1[:, 1],
                z=base_points1[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color='green',
                    opacity=0.8,
                    symbol='square'
                ),
                text=[f"B-{tooth}" for tooth in teeth_ids_base],
                textposition="top center",
                textfont=dict(size=7),
                name=f"{model1_name} Base"
            ))
        
        # Add the second model base points (yellow) after transformation
        if aligned_base_points2.size > 0:
            fig.add_trace(go.Scatter3d(
                x=aligned_base_points2[:, 0],
                y=aligned_base_points2[:, 1],
                z=aligned_base_points2[:, 2],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color='yellow',
                    opacity=0.8,
                    symbol='square'
                ),
                text=[f"B-{tooth}" for tooth in teeth_ids_base],
                textposition="top center",
                textfont=dict(size=7),
                name=f"{model2_name} Base"
            ))
        
        # Add connections between corresponding centroid points only
        for i, tooth in enumerate(teeth_ids_centroids):
            fig.add_trace(go.Scatter3d(
                x=[centroid_points1[i, 0], aligned_centroid_points2[i, 0]],
                y=[centroid_points1[i, 1], aligned_centroid_points2[i, 1]],
                z=[centroid_points1[i, 2], aligned_centroid_points2[i, 2]],
                mode='lines',
                line=dict(
                    color='gray',
                    width=2,
                    dash='dot'
                ),
                showlegend=False
            ))
        
        # Update layout
        region_str = f" - {region.capitalize()}" if region else ""
        transform_info = ""
        if transformation and 'scale_factor' in transformation:
            transform_info = f" (Scale: {transformation['scale_factor']:.3f})"
            
        fig.update_layout(
            title=f"{title}{region_str} (After Optimal Alignment){transform_info}",
            width=900,
            height=700,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # Display the plot
        fig.show()
        
        # Print distances (only for centroids as that's where we calculated alignment)
        print(f"\n{title}{region_str} - Centroid Point Distances (After Optimal Alignment):")
        if transformation and 'scale_factor' in transformation:
            print(f"Applied scale factor: {transformation['scale_factor']:.4f}")
        print("-" * 40)
        print(f"{'Tooth':<10} {'Distance':>10}")
        print("-" * 40)
        for tooth in sorted(distances.keys()):
            # Handle tuple keys by converting them to string
            tooth_str = str(tooth) if isinstance(tooth, tuple) else tooth
            print(f"{tooth_str:<10} {distances[tooth]:>10.4f}")
        print("-" * 40)
        print(f"Average: {sum(distances.values())/len(distances):>10.4f}")
        print("-" * 40)
