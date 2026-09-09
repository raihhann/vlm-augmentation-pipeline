"""Detect and annotate image objects with an Ultralytics RT-DETR model."""

import os
import cv2
import numpy as np
from PIL import Image

# --- FIX: Resolve Absolute Path ---
# This gets the directory of the current file
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Moves up to the 'models' folder (adjust the number of ".." to match your structure)
model_folder = os.path.abspath(os.path.join(current_file_dir, "..", "..", "..", "models"))

# Ensure the directory actually exists
os.makedirs(model_folder, exist_ok=True)

# Set environment variables for Ultralytics
os.environ['YOLO_HOME'] = model_folder
os.environ['ULTRALYTICS_CONFIG_DIR'] = model_folder

from ultralytics import RTDETR

# Define the absolute path to the weights file
model_path = os.path.join(model_folder, 'rtdetr-l.pt')

print(f" Loading RT-DETR from: {model_path}")

# Load RT-DETR model using the absolute path
# This forces the library to look in your models folder instead of the root
rtdetr_model = RTDETR(model_path)

def run_rtdetr(image, save_output=False, output_path=None):
    """
    Run RT-DETR transformer-based detection and create a vertical collage.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the annotated image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and transformer-annotated image.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h, w = original_image.shape[:2]

    # Run RT-DETR inference
    # conf=0.3 filter for confidence, verbose=False to keep logs clean
    results = rtdetr_model.predict(original_image, conf=0.3, verbose=False)[0]
    
    # Generate the annotated image (standard YOLO-style plotting)
    annotated_image = results.plot()

    # Save if requested
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    # Convert final collage back to PIL-friendly RGB
    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("RT-DETR augmentation completed.")
    return Image.fromarray(collage_rgb)