"""Estimate monocular depth using the locally configured ZoeDepth model."""

import cv2
import torch
import numpy as np
import os
from PIL import Image

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
torch.hub.set_dir(cache_dir)

# Load ZoeDepth once globally
# Note: This requires 'timm' and 'torch' installed.
print("[System] Loading ZoeDepth (Metric Depth) Model...")
zoe_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
zoe_model.eval()

def run_zoe_depth(image, save_output=False, output_path=None):
    """
    Generates an Absolute Metric Depth Map using ZoeDepth and returns a vertical collage.
    The bottom image represents depth in physical meters, colorized for visualization.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the colorized depth map.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Vertical collage of original (top) and metric depth (bottom).
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h, w = original_image.shape[:2]

    # 1. Prepare for ZoeDepth (requires PIL input)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(image_rgb)

    # 2. Inference: Get raw metric depth (float32 in meters)
    with torch.no_grad():
        depth_numpy = zoe_model.infer_pil(pil_input)

    # 3. Visualization: Normalize metric data to [0, 255] for colorization
    depth_min = depth_numpy.min()
    depth_max = depth_numpy.max()
    depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # Apply Colormap and resize back to original resolution
    annotated_image = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    annotated_image = cv2.resize(annotated_image, (w, h), interpolation=cv2.INTER_CUBIC)

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
    print("ZoeDepth augmentation completed.")
    return Image.fromarray(collage_rgb)