"""Provide gamma, CLAHE, Retinex, and logarithmic illumination corrections."""

import cv2
import numpy as np
from PIL import Image

# 1. Gamma Correction
def run_gamma_correction(image, gamma=1.5, save_output=False, output_path=None):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Apply Gamma
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced_image = cv2.LUT(image, table)

    if save_output and output_path:
        cv2.imwrite(output_path, enhanced_image)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(enhanced_image, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Gamma correction completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))


# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
def run_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), save_output=False, output_path=None):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Process in LAB space to preserve color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    enhanced_image = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)

    if save_output and output_path:
        cv2.imwrite(output_path, enhanced_image)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(enhanced_image, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("CLAHE augmentation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))


# 3. Retinex SSR (Single Scale Retinex)
def run_retinex_ssr(image, sigma=15, save_output=False, output_path=None):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # Process in HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    v_float = np.float64(v_chan) + 1.0 
    k_size = int(sigma * 3) | 1 # Ensure odd
    illumination = cv2.GaussianBlur(v_float, (k_size, k_size), sigma)
    
    retinex_log = np.log10(v_float) - np.log10(illumination)
    min_val, max_val = np.percentile(retinex_log, (1, 99))
    v_enhanced = np.uint8(255 * (np.clip(retinex_log, min_val, max_val) - min_val) / (max_val - min_val + 1e-8))

    enhanced_image = cv2.cvtColor(cv2.merge([h_chan, s_chan, v_enhanced]), cv2.COLOR_HSV2BGR)

    if save_output and output_path:
        cv2.imwrite(output_path, enhanced_image)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(enhanced_image, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Retinex SSR augmentation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))


# 4. Log Transform
def run_log_transform(image, c=None, save_output=False, output_path=None):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    img_float = np.float32(image)
    if c is None:
        c = 255 / np.log(1 + np.max(img_float))
        
    log_data = c * (np.log(img_float + 1))
    enhanced_image = np.uint8(np.clip(log_data, 0, 255))

    if save_output and output_path:
        cv2.imwrite(output_path, enhanced_image)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(enhanced_image, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))
    print("Log Transform augmentation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))