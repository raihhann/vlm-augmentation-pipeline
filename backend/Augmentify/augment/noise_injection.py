import cv2
import numpy as np
from PIL import Image

def run_noise_injection(image, noise_level=25, save_output=False, output_path=None):
    """
    Adds Gaussian Noise over the pixel grid.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    gaussian_noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    augmented = np.clip(image.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Noise Injection completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))