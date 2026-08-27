"""Apply smooth random elastic warping to an image using Gaussian displacement fields."""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def run_elastic_deformation(image, alpha=34, sigma=4, save_output=False, output_path=None):
    """
    Applies local elastic displacement deformation (commonly used in medical imaging).
    Requires scipy: `pip install scipy`
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w, c = image.shape

    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    augmented = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Elastic Deformation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))