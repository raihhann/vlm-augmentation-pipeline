"""Estimate surface normals or depth-derived surface structure with a transformer pipeline."""

import cv2
import numpy as np
import os
from PIL import Image
from transformers import pipeline

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

# Load Depth estimator once globally
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", model_kwargs={"cache_dir": cache_dir})

def run_surface_normals(image, save_output=False, output_path=None):
    """
    Generates a 3D Surface Normal Map and creates a vertically stacked collage.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save normal map image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Collage of original and surface normal map.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h, w = original_image.shape[:2]

    # Convert BGR back to PIL for the transformer pipeline
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(image_rgb)

    # 1. Inference: Get RAW depth tensor
    depth_output = depth_estimator(pil_input)
    raw_depth_tensor = depth_output["predicted_depth"]
    
    # Convert to float32 and resize to original resolution
    raw_depth_array = raw_depth_tensor.squeeze().cpu().numpy().astype(np.float32)
    depth_array = cv2.resize(raw_depth_array, (w, h), interpolation=cv2.INTER_CUBIC)
    depth_array = cv2.GaussianBlur(depth_array, (5, 5), 0)

    # 2. Calculate Surface Normals (Sobel derivatives)
    dzdx = cv2.Sobel(depth_array, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth_array, cv2.CV_32F, 0, 1, ksize=3)

    # 3. Construct and Normalize 3D Vectors
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth_array)))
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    normal_normalized = normal / (n + 1e-8)

    # 4. Map to RGB [0, 255] then to BGR for internal consistency
    normal_rgb = ((normal_normalized + 1.0) * 127.5).astype(np.uint8)
    annotated_image = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    # Save if requested
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("Surface Normalization completed. Returning collage image.")
    return Image.fromarray(collage)