"""Randomly crop an image and resize the crop back to the original dimensions."""

import cv2
import numpy as np
from PIL import Image

def run_random_cropping_resizing(image, crop_ratio=0.8, save_output=False, output_path=None):
    """
    Crops a central region defined by crop_ratio and resizes it back to original size.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    augmented = cv2.resize(cropped, (w, h))

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Random Cropping and Resizing completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))