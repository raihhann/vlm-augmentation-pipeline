"""Adjust image brightness and contrast and produce an original-result collage."""

import cv2
import numpy as np
from PIL import Image

def run_color_jittering(image, brightness=30, contrast=1.3, save_output=False, output_path=None):
    """Adjusts brightness and contrast of an input image and returns a collage.

    This function modifies the brightness and contrast of the input image using 
    OpenCV's convertScaleAbs. It also optionally saves the augmented image, and 
    returns a vertically stacked collage of the original and modified images.

    Args:
        image (Union[PIL.Image.Image, numpy.ndarray]): The input image to be 
            adjusted. Can be a PIL Image or a NumPy array.
        brightness (int, optional): The brightness adjustment value (typically 
            between -100 and 100). Defaults to 30.
        contrast (float, optional): The contrast adjustment scale factor (1.0 
            means no change, >1 increases contrast). Defaults to 1.3.
        save_output (bool, optional): If True, saves the augmented image to 
            `output_path`. Defaults to False.
        output_path (str, optional): File path where the augmented image should 
            be saved if `save_output` is True. Defaults to None.

    Returns:
        PIL.Image.Image: A PIL Image containing the vertically stacked collage 
            of the resized original and resized augmented images.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Apply brightness and contrast using cv2.convertScaleAbs
    augmented = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Color Jittering completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))