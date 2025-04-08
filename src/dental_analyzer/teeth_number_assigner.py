import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class TeethNumberAssigner:
    def __init__(self, data):
        self.data = data
        self.boundaries = {}

        # Updated dental quadrants reference in FDI notation:
        # Upper jaw (Quadrants 1 & 2): 11-18 and 21-28
        # Lower jaw (Quadrants 3 & 4): 31-38 and 41-48
        self.upper_jaw_numbers = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
        self.lower_jaw_numbers = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

    def assign_teeth_numbers(self, view=None, mask_path=None):
        """
        Assign tooth numbers based on view-based logic using the FDI system.
        Parameters:
            view (str): Specifies the view of the teeth.
            mask_path (str): Path to the mask image for second point calculation.
        """
        if view is None:
            view = self.data.get('view', 'frontal')  # Default to "frontal"

        teeth = self.data['teeth']
        centroids = np.array([tooth['centroid'] for tooth in teeth])

        # Branch logic by view
        if view in ["left", "right"]:
            horizontal_boundary = self._get_horizontal_boundary(centroids, view)
            self.boundaries['horizontal'] = horizontal_boundary
            if view == "left":
                self._assign_left(teeth, horizontal_boundary)
            else:
                self._assign_right(teeth, horizontal_boundary)
        elif view in ["upper", "lower", "frontal"]:
            vertical_boundary = self._get_vertical_boundary(centroids)
            self.boundaries['vertical'] = vertical_boundary
            horizontal_boundary = None
            if view == "frontal":
                horizontal_boundary = self._get_horizontal_boundary(centroids)
                self.boundaries['horizontal'] = horizontal_boundary
            self._assign_main_views(teeth, view, vertical_boundary, horizontal_boundary)
        
        # Perform second point calculation if mask_path is provided
        if mask_path:
            self._calculate_second_point(mask_path)
        
        # Remove the tooth_id key from each tooth
        for tooth in self.data['teeth']:
            if 'tooth_id' in tooth:
                del tooth['tooth_id']

        return self.data

    def _calculate_second_point(self, mask_path):
        """
        Calculates a second point for each tooth using a hybrid approach.
        The expected direction is computed based on the tooth number: up for upper jaw, down for lower jaw.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found at {mask_path}")

        annotated_mask = mask.copy()

        for tooth in self.data['teeth']:
            x_min, y_min, x_max, y_max = map(int, tooth['bbox'])
            cv2.rectangle(annotated_mask, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            tooth_mask = mask[y_min:y_max, x_min:x_max]
            contours, _ = cv2.findContours(tooth_mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                cnt[:, 0, 0] += x_min
                cnt[:, 0, 1] += y_min

            areas = [cv2.contourArea(c) for c in contours]
            if not areas:
                continue
            avg_area = np.mean(areas)
            min_area_threshold = avg_area * 0.2
            contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            tooth['centroid'] = (cx, cy)

            cv2.circle(annotated_mask, (cx, cy), 5, (0, 255, 0), -1)

            if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
                tooth_color = mask[cy, cx]
            else:
                tooth_color = None

            if np.array_equal(tooth_color, [0, 0, 0]):
                tooth['method'] = 'bounding_box'
                continue

            tooth['color'] = tooth_color.tolist()

            contour_points = contour.reshape(-1, 2).astype(np.float32)
            _, eigenvectors = cv2.PCACompute(contour_points, mean=None)
            principal_axis = eigenvectors[0]

            # Determine expected direction based on tooth number
            if tooth['tooth_number'] in self.upper_jaw_numbers:
                expected_direction = np.array([0, 1], dtype=np.float32)  # Up for upper jaw
            elif tooth['tooth_number'] in self.lower_jaw_numbers:
                expected_direction = np.array([0, -1], dtype=np.float32)  # Down for lower jaw
            else:
                expected_direction = np.array([0, 1], dtype=np.float32)

            if np.dot(principal_axis, expected_direction) < 0:
                principal_axis = -principal_axis

            bbox_height = y_max - y_min
            length = min(bbox_height * 0.5, 100)

            second_point = (
                cx + principal_axis[0] * length,
                cy + principal_axis[1] * length
            )
            tooth['second_point'] = second_point

            cv2.circle(annotated_mask, (int(second_point[0]), int(second_point[1])), 5, (0, 0, 255), -1)
            cv2.line(annotated_mask, (cx, cy), (int(second_point[0]), int(second_point[1])), (255, 0, 0), 2)

    def _visualize_mask(self, mask, title="Debug Visualization"):
        plt.figure(figsize=(10, 10))
        if len(mask.shape) == 2:
            plt.imshow(mask, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Circle((0, 0), radius=5, color='lime', label='Centroid'),
            Circle((0, 0), radius=5, color='red', label='Second Point'),
            Line2D([0], [0], color='blue', label='Direction Line')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def _get_horizontal_boundary(self, centroids, view=None):
        """
        Calculate horizontal boundary. For left/right views, a curved function is used.
        """
        if view in ["left", "right"]:
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(centroids[:, 1].reshape(-1, 1))
            return self._get_curved_boundary(centroids, labels)
        else:
            kmeans_y = KMeans(n_clusters=2, random_state=42)
            kmeans_y.fit(centroids[:, 1].reshape(-1, 1))
            return np.mean(kmeans_y.cluster_centers_)

    def _get_curved_boundary(self, centroids, labels):
        """
        Fit a quadratic curve through midpoints between upper and lower clusters.
        """
        upper = centroids[labels == 0]
        lower = centroids[labels == 1]

        if len(upper) == 0 or len(lower) == 0:
            return lambda x: np.mean(centroids[:, 1])

        midpoints = []
        for u_point in upper:
            distances = np.linalg.norm(lower - u_point, axis=1)
            closest_idx = np.argmin(distances)
            closest_lower = lower[closest_idx]
            midpoint = (u_point + closest_lower) / 2
            midpoints.append(midpoint)

        midpoints = np.array(midpoints)
        if len(midpoints) < 3:
            X = midpoints[:, 0].reshape(-1, 1)
            y = midpoints[:, 1]
            model = LinearRegression().fit(X, y)
            return lambda x: model.predict([[x]])[0]
        else:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(midpoints[:, 0].reshape(-1, 1))
            model = LinearRegression().fit(X_poly, midpoints[:, 1])
            return lambda x: model.predict(poly.transform([[x]]))[0]

    def _get_vertical_boundary(self, centroids):
        kmeans_x = KMeans(n_clusters=2, random_state=42)
        kmeans_x.fit_predict(centroids[:, 0].reshape(-1, 1))
        return np.mean(kmeans_x.cluster_centers_)

    def _assign_left(self, teeth, horizontal_boundary):
        """
        For a left-side view (patient's left side), assign FDI numbers.
        When viewing from the left side:
        Upper jaw: 18, 17, 16, 15, 14, 13, 12, 11
        Lower jaw: 48, 47, 46, 45, 44, 43, 42, 41
        
        Note: If more teeth are detected than expected, they will be assigned -1 as tooth number.
        If fewer teeth are detected, only the available numbers will be used.
        """
        q1 = []  # Upper jaw (would show teeth from Q1)
        q4 = []  # Lower jaw (would show teeth from Q4)
        
        # First separate teeth into upper and lower jaw
        for tooth in teeth:
            x, y = tooth['centroid']
            boundary_y = horizontal_boundary(x) if callable(horizontal_boundary) else horizontal_boundary
            if y < boundary_y:
                q1.append(tooth)
            else:
                q4.append(tooth)

        # Sort by x descending so rightmost teeth get higher numbers (18, 48)
        q1.sort(key=lambda t: t['centroid'][0], reverse=True)
        q4.sort(key=lambda t: t['centroid'][0], reverse=True)

        numbering_q1 = [18, 17, 16, 15, 14, 13, 12, 11]
        numbering_q4 = [48, 47, 46, 45, 44, 43, 42, 41]

        # Process upper jaw (Quadrant 1)
        if len(q1) > len(numbering_q1):
            print(f"Warning: More upper teeth detected ({len(q1)}) than expected ({len(numbering_q1)})")
        elif len(q1) < len(numbering_q1):
            print(f"Warning: Fewer upper teeth detected ({len(q1)}) than expected ({len(numbering_q1)})")

        # Process lower jaw (Quadrant 4)
        if len(q4) > len(numbering_q4):
            print(f"Warning: More lower teeth detected ({len(q4)}) than expected ({len(numbering_q4)})")
        elif len(q4) < len(numbering_q4):
            print(f"Warning: Fewer lower teeth detected ({len(q4)}) than expected ({len(numbering_q4)})")

        # Assign numbers, using -1 for extra teeth
        for i, tooth in enumerate(q1):
            tooth['tooth_number'] = numbering_q1[i] if i < len(numbering_q1) else -1
            
        for i, tooth in enumerate(q4):
            tooth['tooth_number'] = numbering_q4[i] if i < len(numbering_q4) else -1

    def _assign_right(self, teeth, horizontal_boundary):
        """
        For a right-side view (patient's right side), assign FDI numbers for
        Quadrant 2 (upper left: 21-28)
        and Quadrant 3 (lower left: 31-38) sorted in ascending order.
        
        Note: If more teeth are detected than expected, they will be assigned -1 as tooth number.
        If fewer teeth are detected, only the available numbers will be used.
        """
        q2 = []  # Upper jaw (would show teeth from Q2)
        q3 = []  # Lower jaw (would show teeth from Q3)
        
        # First separate teeth into upper and lower jaw
        for tooth in teeth:
            x, y = tooth['centroid']
            boundary_y = horizontal_boundary(x) if callable(horizontal_boundary) else horizontal_boundary
            if y < boundary_y:
                q2.append(tooth)
            else:
                q3.append(tooth)

        # For right view, sort by x descending
        q2.sort(key=lambda t: t['centroid'][0], reverse=True)
        q3.sort(key=lambda t: t['centroid'][0], reverse=True)

        numbering_q2 = [21, 22, 23, 24, 25, 26, 27, 28]
        numbering_q3 = [31, 32, 33, 34, 35, 36, 37, 38]

        # Process upper jaw (Quadrant 2)
        if len(q2) > len(numbering_q2):
            print(f"Warning: More upper teeth detected ({len(q2)}) than expected ({len(numbering_q2)})")
        elif len(q2) < len(numbering_q2):
            print(f"Warning: Fewer upper teeth detected ({len(q2)}) than expected ({len(numbering_q2)})")

        # Process lower jaw (Quadrant 3)
        if len(q3) > len(numbering_q3):
            print(f"Warning: More lower teeth detected ({len(q3)}) than expected ({len(numbering_q3)})")
        elif len(q3) < len(numbering_q3):
            print(f"Warning: Fewer lower teeth detected ({len(q3)}) than expected ({len(numbering_q3)})")

        # Assign numbers, using -1 for extra teeth
        for i, tooth in enumerate(q2):
            tooth['tooth_number'] = numbering_q2[i] if i < len(numbering_q2) else -1
            
        for i, tooth in enumerate(q3):
            tooth['tooth_number'] = numbering_q3[i] if i < len(numbering_q3) else -1

    def _assign_main_views(self, teeth, view, vertical_boundary, horizontal_boundary):
        if view == "upper":
            self._assign_upper_view(teeth, vertical_boundary)
        elif view == "lower":
            self._assign_lower_view(teeth, vertical_boundary)
        elif view == "frontal":
            self._assign_frontal_view(teeth, vertical_boundary, horizontal_boundary)

    def _assign_upper_view(self, teeth, vertical_boundary):
        """
        For an upper view, split the upper jaw into Quadrant 1 (upper right: 11-18)
        and Quadrant 2 (upper left: 21-28) using the vertical boundary.
        """
        q1 = [t for t in teeth if t['centroid'][0] < vertical_boundary]
        q2 = [t for t in teeth if t['centroid'][0] >= vertical_boundary]

        # For Q1 (upper right), sort by x descending (central incisor first)
        q1.sort(key=lambda t: t['centroid'][0], reverse=True)
        # For Q2 (upper left), sort by x ascending (central incisor first)
        q2.sort(key=lambda t: t['centroid'][0])

        numbering_q1 = [11, 12, 13, 14, 15, 16, 17, 18]
        numbering_q2 = [21, 22, 23, 24, 25, 26, 27, 28]

        for i, tooth in enumerate(q1):
            if i < len(numbering_q1):
                tooth['tooth_number'] = numbering_q1[i]
        for i, tooth in enumerate(q2):
            if i < len(numbering_q2):
                tooth['tooth_number'] = numbering_q2[i]

    def _assign_lower_view(self, teeth, vertical_boundary):
        """
        For a lower view, split the lower jaw into Quadrant 3 (lower left: 31-38)
        and Quadrant 4 (lower right: 41-48) using the vertical boundary.
        """
        q3 = [t for t in teeth if t['centroid'][0] >= vertical_boundary]
        q4 = [t for t in teeth if t['centroid'][0] < vertical_boundary]

        # For Q3 (lower left), sort by x ascending (central incisor first)
        q3.sort(key=lambda t: t['centroid'][0])
        # For Q4 (lower right), sort by x descending (central incisor first)
        q4.sort(key=lambda t: t['centroid'][0], reverse=True)

        numbering_q3 = [31, 32, 33, 34, 35, 36, 37, 38]
        numbering_q4 = [41, 42, 43, 44, 45, 46, 47, 48]

        for i, tooth in enumerate(q3):
            if i < len(numbering_q3):
                tooth['tooth_number'] = numbering_q3[i]
        for i, tooth in enumerate(q4):
            if i < len(numbering_q4):
                tooth['tooth_number'] = numbering_q4[i]

    def _assign_frontal_view(self, teeth, vertical_boundary, horizontal_boundary):
        """
        For a frontal view, assign each tooth to a quadrant based on its position:
            Quadrant 1 (upper right): x < vertical_boundary, y < horizontal_boundary  -> 11-18
            Quadrant 2 (upper left):  x >= vertical_boundary, y < horizontal_boundary -> 21-28
            Quadrant 3 (lower left):  x >= vertical_boundary, y >= horizontal_boundary -> 31-38
            Quadrant 4 (lower right): x < vertical_boundary, y >= horizontal_boundary  -> 41-48
        Within each quadrant the teeth are sorted from the central incisor (first) toward the third molar.
        """
        for tooth in teeth:
            x, y = tooth['centroid']
            if x < vertical_boundary and y < horizontal_boundary:
                tooth['quadrant'] = 1
            elif x >= vertical_boundary and y < horizontal_boundary:
                tooth['quadrant'] = 2
            elif x >= vertical_boundary and y >= horizontal_boundary:
                tooth['quadrant'] = 3
            else:
                tooth['quadrant'] = 4

        q1 = [t for t in teeth if t.get('quadrant') == 1]
        q2 = [t for t in teeth if t.get('quadrant') == 2]
        q3 = [t for t in teeth if t.get('quadrant') == 3]
        q4 = [t for t in teeth if t.get('quadrant') == 4]

        # For Quadrant 1 (upper right), sort by x descending (central incisor first)
        q1.sort(key=lambda t: t['centroid'][0], reverse=True)
        # For Quadrant 2 (upper left), sort by x ascending (central incisor first)
        q2.sort(key=lambda t: t['centroid'][0])
        # For Quadrant 3 (lower left), sort by x ascending (central incisor first)
        q3.sort(key=lambda t: t['centroid'][0])
        # For Quadrant 4 (lower right), sort by x descending (central incisor first)
        q4.sort(key=lambda t: t['centroid'][0], reverse=True)

        numbering_q1 = [11, 12, 13, 14, 15, 16, 17, 18]
        numbering_q2 = [21, 22, 23, 24, 25, 26, 27, 28]
        numbering_q3 = [31, 32, 33, 34, 35, 36, 37, 38]
        numbering_q4 = [41, 42, 43, 44, 45, 46, 47, 48]

        for i, tooth in enumerate(q1):
            if i < len(numbering_q1):
                tooth['tooth_number'] = numbering_q1[i]
        for i, tooth in enumerate(q2):
            if i < len(numbering_q2):
                tooth['tooth_number'] = numbering_q2[i]
        for i, tooth in enumerate(q3):
            if i < len(numbering_q3):
                tooth['tooth_number'] = numbering_q3[i]
        for i, tooth in enumerate(q4):
            if i < len(numbering_q4):
                tooth['tooth_number'] = numbering_q4[i]
