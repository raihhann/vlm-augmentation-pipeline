import cv2
import numpy as np
from PIL import Image

def run_perspective_transform(image, distortion_scale=0.15, save_output=False, output_path=None):
    """
    Warps corner points to simulate perspective camera tilt/angle changes.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    dx = int(w * distortion_scale)
    dy = int(h * distortion_scale)

    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[dx, dy], [w - dx, dy], [0, h], [w, h]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    augmented = cv2.warpPerspective(image, M, (w, h))

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Perspective Transform completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))