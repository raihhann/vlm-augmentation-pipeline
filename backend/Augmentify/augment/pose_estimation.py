"""Detect human poses with YOLO and draw keypoints and skeleton annotations."""

import os
import cv2
import numpy as np
from PIL import Image

# --- FIX: Resolve Absolute Path ---
# Get the absolute path of the current file's directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the shared 'models' folder
model_folder = os.path.abspath(os.path.join(current_file_dir, "..", "..", "..", "models"))

# Ensure the directory exists
os.makedirs(model_folder, exist_ok=True)

# Set environment variables to force Ultralytics to use the models folder
os.environ['YOLO_HOME'] = model_folder
os.environ['ULTRALYTICS_CONFIG_DIR'] = model_folder

from ultralytics import YOLO

# Define the absolute path to the pose weights
model_path = os.path.join(model_folder, 'yolov8n-pose.pt')

print(f" Loading YOLOv8-Pose from: {model_path}")

# Load the Pose model using the absolute path
pose_model = YOLO(model_path)

def run_pose_estimation(image, save_output=False, output_path=None):
    """
    Run YOLOv8 pose estimation and create a vertically stacked collage.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the annotated skeleton image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and pose-annotated image.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()

    # Run YOLOv8 inference
    # verbose=False keeps the console clean during batch processing
    results = pose_model(original_image, verbose=False)[0]
    
    # Generate the annotated image (skeleton plot)
    # boxes=False keeps the focus on keypoints/kinematics
    annotated_image = results.plot(boxes=False)

    # Save if requested
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    h, w = original_image.shape[:2]

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("Pose estimation augmentation completed.")
    return Image.fromarray(collage)