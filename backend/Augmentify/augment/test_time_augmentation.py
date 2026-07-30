import cv2
import numpy as np
from PIL import Image

def run_test_time_augmentation(image, save_output=False, output_path=None):
    """
    Simulates Test-Time Augmentation (TTA) by averaging predictions/multi-views.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Generate multi-views and average them
    v1 = image.astype(np.float32)
    v2 = cv2.flip(image, 1).astype(np.float32)
    v2 = cv2.flip(v2, 1)  # un-flip for visual overlay

    augmented = np.clip((v1 * 0.5 + v2 * 0.5), 0, 255).astype(np.uint8)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Test-Time Augmentation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))