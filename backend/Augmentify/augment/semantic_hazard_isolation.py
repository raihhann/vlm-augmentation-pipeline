"""Identify prompt-related hazard regions with CLIPSeg and isolate them visually."""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Initialize CLIPSeg model once at module level
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
MODEL = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)


def _locate_general_hazards(image_bgr: np.ndarray, prompt: str = "") -> np.ndarray:
    """Uses CLIPSeg to locate hazards and return a binary mask."""
    h, w = image_bgr.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Use prompt if provided; otherwise fallback to general semantic hazard categories
    hazard_phrase = prompt.strip() if prompt.strip() else "something dangerous, safety hazard, sharp object, spill"

    inputs = PROCESSOR(
        text=[hazard_phrase],
        images=[pil_image],
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    # Generate heat map
    heatmap = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply threshold and clean up noise
    _, binary_mask = cv2.threshold(heatmap, 95, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def _apply_hazard_overlay(image_bgr: np.ndarray, hazard_mask: np.ndarray) -> np.ndarray:
    """Applies semi-transparent red overlay and bounding boxes around detected hazards."""
    canvas = image_bgr.copy()

    if np.any(hazard_mask > 0):
        # Create red warning tint overlay
        alert_layer = np.zeros_like(image_bgr)
        alert_layer[:] = [0, 0, 240]  # BGR Warning Red

        mask_indices = hazard_mask > 0
        canvas[mask_indices] = cv2.addWeighted(canvas, 0.6, alert_layer, 0.4, 0)[mask_indices]

        # Draw bounding boxes around hazard areas
        contours, _ = cv2.findContours(hazard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                hx, hy, hw, hh = cv2.boundingRect(cnt)
                cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 4)

    return canvas


def run_semantic_hazard_isolation(image, prompt="", save_output=False, output_path=None):
    """
    Isolates semantic hazard regions and adds alert overlays.
    Returns stacked PIL Image (Original on top, Augmented on bottom).
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # 1. Detect hazard mask
    hazard_mask = _locate_general_hazards(original_image, prompt)

    # 2. Render warning overlay
    augmented = _apply_hazard_overlay(original_image, hazard_mask)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))

    print("Semantic Hazard Isolation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))