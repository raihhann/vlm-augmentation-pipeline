import cv2
import numpy as np
from PIL import Image

def run_shear_mapping(image, shear_factor=0.2, save_output=False, output_path=None):
    """
    Applies Affine Shear Mapping along the x-axis.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    # Shear matrix
    M = np.float32([[1, shear_factor, 0],
                    [0, 1, 0]])
    
    augmented = cv2.warpAffine(image, M, (int(w + h * abs(shear_factor)), h))
    augmented = cv2.resize(augmented, (w, h))

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Shear Mapping completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))