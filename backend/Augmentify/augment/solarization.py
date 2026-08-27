"""Invert pixels above a threshold to simulate photographic solarization."""

import cv2
import numpy as np
from PIL import Image

def run_solarization(image, threshold=128, save_output=False, output_path=None):
    """
    Inverts pixel values above a given threshold intensity.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Invert pixels brighter than threshold
    augmented = np.where(image > threshold, 255 - image, image).astype(np.uint8)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Solarization completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))