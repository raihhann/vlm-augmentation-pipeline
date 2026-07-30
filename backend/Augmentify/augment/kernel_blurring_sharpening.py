import cv2
import numpy as np
from PIL import Image

def run_kernel_blurring_sharpening(image, kernel_size=9, mode="blur", save_output=False, output_path=None):
    """
    Applies Gaussian Blur or Sharpening depending on `mode` ('blur' or 'sharpen').
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    if mode == "blur":
        augmented = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif mode == "sharpen":
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        augmented = cv2.filter2D(image, -1, sharpen_kernel)
    else:
        augmented = image.copy()

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print(f"Kernel {mode.capitalize()} completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))