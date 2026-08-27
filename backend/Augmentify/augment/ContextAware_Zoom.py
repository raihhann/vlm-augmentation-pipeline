"""Locate a prompt-named visual region with CLIPSeg and create a padded zoom."""

import cv2
import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import ollama

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

# Initialize CLIPSeg globally
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache_dir)
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache_dir).to(device)

def _get_visual_query(user_prompt):
    """Internal helper to convert prompt to visual object via LLM."""
    system_prompt = "Output ONLY the main visual object from the prompt (e.g., screen, sign, text). No abstract ideas."
    try:
        response = ollama.chat(
            model="llava",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            options={"temperature": 0.0, "num_predict": 15}
        )
        return response["message"]["content"].strip().lower()
    except:
        return "object"

def run_automated_zoom(image, user_prompt, save_output=False, output_path=None, padding=60):
    """
    Finds a specific object based on a prompt, zooms in (ROI), 
    and returns a vertical collage.
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h, w = original_image.shape[:2]

    # 1. Planner: Interpret the user prompt
    visual_query = _get_visual_query(user_prompt)

    # 2. CLIPSeg: Get heatmap and ROI
    pil_input = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[visual_query], images=[pil_input], padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    heatmap = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Calculate bounding box for the ROI
    ys, xs = np.where(heatmap > 200)
    
    if len(xs) > 0 and len(ys) > 0:
        x1 = max(0, xs.min() - padding)
        y1 = max(0, ys.min() - padding)
        x2 = min(w, xs.max() + padding)
        y2 = min(h, ys.max() + padding)
        annotated_image = original_image[y1:y2, x1:x2]
    else:
        annotated_image = original_image.copy()

    # Save if requested (saving just the ROI crop)
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    # Resize original and the ROI crop to half height while keeping original width
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("Context-Aware Zoom completed.")
    return Image.fromarray(collage_rgb)