# midas.py
"""Estimate relative monocular depth using the locally configured MiDaS model."""

import cv2
import torch
import numpy as np
import os
from PIL import Image

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
torch.hub.set_dir(cache_dir)

# Load MiDaS model globally for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "DPT_Hybrid"  # or "DPT_Large"
midas_model = torch.hub.load("intel-isl/MiDaS", model_type)
midas_model.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def run_midas_depth(image, output_path: str = None):
    """
    Run MiDaS depth estimation on an image and optionally save it.

    Args:
        image (str | PIL.Image.Image | np.ndarray): Input image (file path, PIL, or OpenCV BGR array)
        output_path (str, optional): If provided, saves the depth image to this path.

    Returns:
        PIL.Image.Image: Depth map normalized as RGB image.
    """
    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            print("❌ Image not found")
            return None
    # else assume it's already a NumPy BGR array

    h, w = image.shape[:2]

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Prepare input tensor
    input_tensor = transform(img_rgb).to(device)

    # Inference
    with torch.no_grad():
        depth = midas_model(input_tensor).squeeze().cpu().numpy()

    # Resize to original image size
    depth_resized = cv2.resize(depth, (w, h))

    # Normalize to 0-255
    depth_norm = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)

    # Convert to PIL Image (RGB)
    depth_pil = Image.fromarray(depth_img).convert("RGB")

    # Convert depth image to OpenCV BGR for stacking
    depth_bgr = cv2.cvtColor(np.array(depth_pil), cv2.COLOR_RGB2BGR)

    # -------- COLLAGE PART --------
    h, w = image.shape[:2]

    # Resize both images to half height
    orig_resized = cv2.resize(image, (w, h // 2))
    depth_resized = cv2.resize(depth_bgr, (w, h // 2))

    # Vertical collage with same final resolution
    collage = np.vstack((orig_resized, depth_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    collage_pil = Image.fromarray(collage)

    # Save if requested
    if output_path is not None:
        collage_pil.save(output_path)
        print(f"✅ Saved depth image to: {output_path}")

    print("MiDaS depth estimation completed.")

    return collage_pil