The `config_X.json` file should have the following structure, the see the example in `config_1.json`:

```json
{
  "visits": {
    "[visit_id]": { // String identifier for the visit (e.g., "0", "1")
      "distanceAB11": "Number", // (Optional) the height in cm of the 11th tooth, for scaling
      "data": {
        "frontal": "String", // Filename of the frontal image with no extension
        "upper": "String",   // Filename of the upper arch image with no extension
        "lower": "String",   // Filename of the lower arch image with no extension
        "right": "String",   // Filename of the right buccal image with no extension
        "left": "String",    // Filename of the left buccal image with no extension
        "dirs": "String",    // Path to the directory containing visit files
        "stl-upper": "String", // Filename of the upper arch STL model
        "stl-lower": "String", // Filename of the lower arch STL model
        "truth-upper": "String", // Filename of the upper ground truth points file
        "truth-lower": "String"  // Filename of the lower ground truth points file
      },
      "camera": {
        "intrinsics": {
          "focal_length_px": "Number", // Camera focal length in pixels
          "image_size": ["Number", "Number"] // Image dimensions [width, height] in pixels
        },
        "extrinsics": {
          // Extrinsic parameters (Rotation, Translation, Intrinsics Matrix K) for each view
          "frontal": {
            "R": [["Number"]*3]*3, // 3x3 Rotation Matrix
            "t": ["Number"]*3,    // 3x1 Translation Vector
            "K": [["Number"]*3]*3     // 3x3 Intrinsic Matrix (potentially view-specific adjustments)
          },
          "upper": { /* ... same structure as frontal ... */ },
          "lower": { /* ... same structure as frontal ... */ },
          "left": { /* ... same structure as frontal ... */ },
          "right": { /* ... same structure as frontal ... */ }
        }
      },
      "comparison": { // (Optional) Parameters for alignment/comparison, between reconstructed and STL model
        "upper": {
          "rotation_angle": "Number",          // Overall rotation angle
          "rotation_angle_points_x": "Number", // Rotation component/parameter for X-axis
          "rotation_angle_points_y": "Number", // Rotation component/parameter for Y-axis
          "rotation_angle_points_z": "Number", // Rotation component/parameter for Z-axis
          "scale_factor": "Number",            // Scaling factor applied
          "offset_x": "Number",                // Translation offset in X
          "offset_y": "Number",                // Translation offset in Y
          "offset_z": "Number",                // Translation offset in Z
          "truth_points_file": "String"        // Ground truth file used for this comparison
        },
        "lower": { /* ... same structure as upper ... */ }
      }
    }
    // ... more visits
  }
}
```
