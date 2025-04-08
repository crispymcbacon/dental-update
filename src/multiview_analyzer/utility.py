import cv2
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        pass
    
    def extract_landmarks_from_roi(self, roi, offset=(0, 0)):
        """Extract basic landmarks (centroid, top, bottom, left, right) in pixel coords."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
        ys, xs = np.nonzero(gray)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        centroid = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        top = ((x_min + x_max) // 2, y_min)
        bottom = ((x_min + x_max) // 2, y_max)
        left = (x_min, (y_min + y_max) // 2)
        right = (x_max, (y_min + y_max) // 2)

        def add_offset(pt):
            return (pt[0] + offset[0], pt[1] + offset[1])

        landmarks = {
            'centroid': add_offset(centroid),
            'top': add_offset(top),
            'bottom': add_offset(bottom),
            'left': add_offset(left),
            'right': add_offset(right)
        }
        return landmarks

    def _compute_view_bounding_box(self, landmarks_view):
        all_x, all_y = [], []
        for tooth_number, lm_dict in landmarks_view.items():
            for name, (x, y) in lm_dict.items():
                all_x.append(x)
                all_y.append(y)
        if not all_x:
            return None
        return min(all_x), min(all_y), max(all_x), max(all_y)

    def normalize_landmarks(self, landmarks_view):
        """
        This function returns a normalized version of the landmarks
        but for actual perspective triangulation we do NOT rely on it.
        Keeping it only if you want it for other visualization tasks.
        """
        bbox = self._compute_view_bounding_box(landmarks_view)
        if bbox is None:
            return landmarks_view
        x_min, y_min, x_max, y_max = bbox
        dx = x_max - x_min
        dy = y_max - y_min
        if dx < 1e-9 or dy < 1e-9:
            return landmarks_view
        normalized_landmarks = {}
        for tooth_number, lm_dict in landmarks_view.items():
            normalized_landmarks[tooth_number] = {}
            for name, (x, y) in lm_dict.items():
                nx = (x - x_min) / dx
                ny = (y - y_min) / dy
                normalized_landmarks[tooth_number][name] = (nx, ny)
        return normalized_landmarks

def build_projection_matrix(R, t, K=None, alpha=1.0, offset=None):
    """
    Build a camera projection matrix:  P = K * [R_wc | t_wc]
    If K is None, returns the old 3x4 for unit focal length.
    """
    if offset is None:
        offset = np.zeros(3, dtype=float)
    t_refined = alpha * t + offset

    # Convert world->camera:
    # R_wc = R^T if R is camera->world... but your code's convention is already that R is world->camera,
    # so let's be consistent with your usage.
    R_wc = R.T
    t_wc = -R_wc @ t_refined

    extrinsic = np.hstack((R_wc, t_wc.reshape(3,1)))  # 3x4
    if K is not None:
        return K @ extrinsic
    else:
        # fallback: no intrinsics => treat as f=1 pinhole
        return extrinsic


def is_valid_tooth_number(tn):
    if not isinstance(tn, int):
        return False
    quadrant = tn // 10
    tooth_idx = tn % 10
    return quadrant in [1, 2, 3, 4] and (1 <= tooth_idx <= 8)

def is_upper_tooth(tn):
    # Handle tuple case (tooth_number, point_type)
    if isinstance(tn, tuple) and len(tn) == 2:
        tn = tn[0]  # Extract the tooth number
    
    # Handle string case
    if isinstance(tn, str):
        try:
            tn = int(tn)
        except ValueError:
            return False
    
    # Now we should have an integer
    if not isinstance(tn, int):
        return False
        
    quadrant = tn // 10
    return quadrant in [1, 2]

def is_lower_tooth(tn):
    # Handle tuple case (tooth_number, point_type)
    if isinstance(tn, tuple) and len(tn) == 2:
        tn = tn[0]  # Extract the tooth number
    
    # Handle string case
    if isinstance(tn, str):
        try:
            tn = int(tn)
        except ValueError:
            return False
    
    # Now we should have an integer
    if not isinstance(tn, int):
        return False
        
    quadrant = tn // 10
    return quadrant in [3, 4]
