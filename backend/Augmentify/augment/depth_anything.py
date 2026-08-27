"""Estimate monocular depth with the Hugging Face Depth Anything pipeline."""

import cv2
import numpy as np
import os
from PIL import Image
from transformers import pipeline

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

# Load Depth pipeline once globally
depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", model_kwargs={"cache_dir": cache_dir})

def run_depth_anything(image, save_output=False, output_path=None):
    """
    Run Depth Anything V2 estimation and create a vertically stacked collage.
    
    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save annotated image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and depth map.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    
    # Convert BGR back to RGB for the transformer pipeline
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(image_rgb)

    # Run Inference
    depth_output = depth_pipe(pil_input)
    depth_map = np.array(depth_output["depth"])

    # Process depth map to colored visualization (Annotated version)
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    annotated_image = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

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
    print("Depth Anything V2 augmentation completed.")
    return Image.fromarray(collage)