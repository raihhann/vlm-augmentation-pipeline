"""Apply configurable flipping and rotation transformations to an image."""

import cv2
import numpy as np
from PIL import Image

def run_geometric_transformations(image, flip_code=1, angle=15, save_output=False, output_path=None):
    """
    Applies Horizontal Flip + Rotation (Geometric Transformation).
    flip_code: 1 for horizontal flip, 0 for vertical, -1 for both.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    # Horizontal Flip
    augmented = cv2.flip(image, flip_code)

    # Rotation
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(augmented, matrix, (w, h))

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Geometric Transformation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))