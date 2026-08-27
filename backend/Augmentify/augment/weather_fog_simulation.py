"""Blend atmospheric haze into images to simulate configurable fog conditions."""

import cv2
import numpy as np
from PIL import Image

def run_weather_fog_simulation(image, fog_intensity=0.4, save_output=False, output_path=None):
    """
    Simulates environmental fog by blending a white overlay with low contrast.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()
    h, w = image.shape[:2]

    fog_mask = np.full((h, w, 3), 255, dtype=np.uint8)
    augmented = cv2.addWeighted(image, 1.0 - fog_intensity, fog_mask, fog_intensity, 0)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Weather Fog Simulation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))