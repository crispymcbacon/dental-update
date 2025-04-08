import matplotlib.pyplot as plt
import cv2
import numpy as np

class TeethVisualizer:
    def __init__(self, data, boundaries):
        self.data = data
        self.boundaries = boundaries

    def visualize(self, image_path=None, mask_path=None, view=None):
        """Visualize teeth numbering and orientation with curved boundaries for left/right views."""
        if view is None:
            view = self.data.get('view', 'frontal')

        if image_path:
            img = cv2.imread(image_path)
        elif mask_path:
            img = cv2.imread(mask_path)
        else:
            raise ValueError("Either image_path or mask_path must be provided")
        

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)

        # Draw boundaries
        if 'horizontal' in self.boundaries:
            boundary = self.boundaries['horizontal']
            if callable(boundary):
                # Plot curved boundary for left/right views
                x_vals = np.linspace(0, img.shape[1], 100)
                y_vals = [boundary(x) for x in x_vals]
                plt.plot(x_vals, y_vals, color='yellow', linestyle='--', alpha=0.7, label='Horizontal Boundary')
            else:
                # Linear boundary for other views
                plt.axhline(y=boundary, color='yellow', linestyle='--', alpha=0.7)

        if 'vertical' in self.boundaries and view in ["upper", "lower", "frontal"]:
            plt.axvline(x=self.boundaries['vertical'], color='yellow', linestyle='--', alpha=0.7)

        # Define jaw colors
        upper_jaw_numbers = list(range(1, 17))
        lower_jaw_numbers = list(range(17, 33))

        # Plot teeth numbers and orientation
        for tooth in self.data['teeth']:
            if 'tooth_number' not in tooth:
                continue
            x, y = tooth['centroid']
            number = tooth['tooth_number']
            color = 'orange' if number in upper_jaw_numbers else 'cyan'

            plt.text(x, y, str(number), color='white', fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor=color, alpha=0.7))

            if 'second_point' in tooth:
                x2, y2 = tooth['second_point']
                plt.plot([x, x2], [y, y2], color=color, linewidth=2, alpha=0.7)
                plt.plot(x2, y2, 'o', color=color, markersize=5, alpha=0.7)

        plt.axis('off')
        plt.show()