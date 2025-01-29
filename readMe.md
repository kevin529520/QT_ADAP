# GUI_APAD:  Adaptive & Dynamic Arc Padding for Predicting Seam Profiles in Multi-Layer Multi-Pass Robotic Welding

## Overview

GUI_APAD is a project designed to read point cloud data, extract cross-sections, and perform multi-layer multi-pass welding position and bead profile predictions. The project includes a graphical user interface (GUI) for easy interaction and visualization.

## Features

- Load and visualize point cloud data.
- Extract cross-sections from point cloud data.
- Predict welding positions and bead profiles.
- Display and save prediction results.
- Calculate and display surface smoothness using the height standard deviation of the weld bead surface.
- Detect weld positions by moving the mouse over the image.

## Requirements

- Python 3.7+
- Open3D
- NumPy
- Matplotlib
- Pillow
- Tkinter
- PyTorch
- OpenCV

## Installation

1. Clone the repository:

   ```bash
   git clone  https://github.com/kevin529520/QT_ADAP.git
   cd GUI_apad
   ```
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python QT_APAD.py
   ```
2. Use the GUI to interact with the application:

   - **Load Point Cloud**: Load a point cloud file (PCD or PLY format).
   - **Load Image**: Load an image file (JPG or PNG format).
   - **Reset Image**: Reset the displayed image to the original.
   - **Predict Weld Position**: Predict the welding position on the loaded image.
   - **Confirm Prediction**: Confirm the prediction and update the displayed image.
   - **X Cross-section Slider**: Adjust the X cross-section value and update the point cloud projection.
   - **Mouse Movement**: Detect weld positions by moving the mouse over the image, and perform real-time prediction of weld bead profile.

## File Structure

- `QT_APAD.py`: Main application file containing the GUI and core logic.
- `point2image.py`: Contains the `PointCloudProcessor` class for processing point cloud data.
- `beadProfilePredict.py`: Contains the `beadProfilePredict` function for predicting bead profiles.
- `weldPosPredict.py`: Contains the `weldPosPredict` function for predicting welding positions.
- `modules/`: Directory containing the model definitions and training scripts.
- `images/`: Directory for storing images and processed results.
- `pointcloud/`: Directory for storing point cloud data and transformed point clouds.
- `weights/`: Directory for storing model weights.

## Example

1. Load a point cloud file:

   - Click the "load pointcloud" button and select a PCD or PLY file.
2. Load an image file:

   - Click the "load image" button and select a JPG or PNG file.
3. Predict the welding position:

   - Click the "predict weld position" button to predict the welding position on the loaded image.
4. Confirm the prediction:

   - Click the "confirm prediction" button to confirm the prediction and update the displayed image.
5. Adjust the X cross-section:

   - Use the slider to adjust the X cross-section value and update the point cloud projection.
6. Detect weld positions:

   - Move the mouse over the image to detect and predict the weld position.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Open3D](http://www.open3d.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)

## Contact

For any questions or issues, please contact [12233195@mail.sustech.edu.cn](12233195@mail.sustech.edu.cn).
