"""Run YOLO segmentation and render detected geometric regions on the image."""

import os
import cv2
import numpy as np
from PIL import Image

# --- FIX: Resolve Absolute Path ---
# Get the directory of the current script
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the shared 'models' folder
model_folder = os.path.abspath(os.path.join(current_file_dir, "..", "..", "..", "models"))

# Ensure the directory exists
os.makedirs(model_folder, exist_ok=True)

# Set environment variables for Ultralytics
os.environ['YOLO_HOME'] = model_folder
os.environ['ULTRALYTICS_CONFIG_DIR'] = model_folder

from ultralytics import YOLO

# Define the absolute path to the segmentation weights
model_path = os.path.join(model_folder, 'yolov8n-seg.pt')

print(f"🛠️ Loading YOLOv8-Seg from: {model_path}")

# Load the model using the absolute path
seg_model = YOLO(model_path)

def run_geometric_segmentation(image, save_output=False, output_path=None):
    """
    Extracts the exact geometric pixel masks of objects using YOLOv8.
    Returns a vertically stacked collage of original and masked images.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the annotated mask image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and segmented image.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()

    # Run YOLOv8 Segmentation inference
    # verbose=False keeps the console clean
    results = seg_model(original_image, verbose=False)[0]
    
    # Plot ONLY the geometric masks (boxes=False, labels=False to avoid text clutter)
    # This creates the "Annotated" version for the bottom half of the collage
    annotated_image = results.plot(boxes=False, labels=False)

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

    # Convert final collage back to PIL-friendly RGB
    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("YOLOv8 geometric segmentation completed.")
    return Image.fromarray(collage_rgb)