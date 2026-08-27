"""Apply the module's fixed stylizing filter and return an augmented image collage."""

import cv2
import numpy as np
from PIL import Image

def run_style_transfer_filter(image, save_output=False, output_path=None):
    """
    Applies OpenCV Stylization to mimic artistic/texture transfers.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    augmented = cv2.stylization(image, sigma_s=60, sigma_r=0.4)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Style Transfer Filter completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))