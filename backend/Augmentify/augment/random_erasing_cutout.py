import cv2
import numpy as np
from PIL import Image

def run_random_erasing_cutout(image, patch_size=(0.2, 0.2), fill_val=0, save_output=False, output_path=None):
    """
    Masks/erases a rectangular region with black or a solid pixel color.
    patch_size: ratio of (height, width) to mask relative to the image size.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    augmented = image.copy()
    h, w = image.shape[:2]

    patch_h = int(h * patch_size[0])
    patch_w = int(w * patch_size[1])

    y = np.random.randint(0, h - patch_h)
    x = np.random.randint(0, w - patch_w)

    augmented[y:y + patch_h, x:x + patch_w] = fill_val

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Random Erasing / Cutout completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))