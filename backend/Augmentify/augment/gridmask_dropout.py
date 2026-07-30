import cv2
import numpy as np
from PIL import Image

def run_gridmask_dropout(image, d=60, ratio=0.5, save_output=False, output_path=None):
    """
    Applies GridMask dropout (regularly structured grid of dropped pixels).
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    grid_mask = np.ones((h, w), dtype=np.uint8)
    l = int(d * ratio)

    for y in range(0, h, d):
        for x in range(0, w, d):
            grid_mask[y:min(y + l, h), x:min(x + l, w)] = 0

    augmented = cv2.bitwise_and(image, image, mask=grid_mask)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("GridMask Dropout completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))