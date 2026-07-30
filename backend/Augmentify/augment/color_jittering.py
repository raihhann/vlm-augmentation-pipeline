import cv2
import numpy as np
from PIL import Image

def run_color_jittering(image, brightness=30, contrast=1.3, save_output=False, output_path=None):
    """
    Adjusts brightness and contrast of the input image.
    brightness: -100 to 100
    contrast: >0 scale factor (1.0 = normal)
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