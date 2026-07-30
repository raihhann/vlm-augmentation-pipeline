import cv2
import numpy as np
from PIL import Image

def run_mosaic_augmentation(image, save_output=False, output_path=None):
    """
    Creates a 2x2 mosaic grid using 4 tiled variations of the input image.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    # Tile 4 transformed instances of the image
    img1 = cv2.resize(image, (w // 2, h // 2))
    img2 = cv2.resize(cv2.flip(image, 1), (w // 2, h // 2))
    img3 = cv2.resize(cv2.flip(image, 0), (w // 2, h // 2))
    img4 = cv2.resize(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), (w // 2, h // 2))

    top_row = np.hstack((img1, img2))
    bot_row = np.hstack((img3, img4))
    augmented = np.vstack((top_row, bot_row))

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Mosaic Augmentation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))