"""Remove image backgrounds with rembg and return a composited result image."""

from time import time

import cv2
import numpy as np
from PIL import Image
from rembg import remove
import time

def run_rembg(image, save_output=False, output_path=None):
    """Run rembg on the provided input.

    Args:
        image: Description of the parameter.
        save_output: Description of the parameter.
        output_path: Description of the parameter.

    Returns:
        numpy.ndarray: The processed image or output result.
    """    
    start_aug = time.time()
    """
    Runs saliency-based foreground extraction using rembg.
    Returns a vertical collage of the original and the saliency-masked overlay.

    Args:
        image (PIL.Image.Image or np.ndarray): Input image.
        save_output (bool): Whether to save the annotated image.
        output_path (str): Path to save image if save_output=True.

    Returns:
        PIL.Image.Image: Vertical collage of original and saliency overlay.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h, w = original_image.shape[:2]

    # 1. Prepare for Rembg (requires PIL RGB input)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_input = Image.fromarray(image_rgb)

    # 2. Run rembg to get RGBA output
    output_pil = remove(pil_input)
    rgba = np.array(output_pil)

    # 3. Extract Alpha Mask and create Saliency Visualization
    # Alpha channel is the 4th channel [3]
    mask = rgba[:, :, 3]
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Create the annotated version (Overlay)
    annotated_image = cv2.addWeighted(original_image, 0.6, mask_colored, 0.4, 0)

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
    end_aug = time.time()
    print(f"Rembg augmentation completed in {end_aug - start_aug:.2f} seconds.")
    return Image.fromarray(collage_rgb)