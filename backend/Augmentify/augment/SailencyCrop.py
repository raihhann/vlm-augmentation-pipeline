"""Use CLIPSeg prompt saliency to crop an image around the relevant region."""

import cv2
import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Define cache directory for models
cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")

# Load CLIPSeg once globally
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache_dir)
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=cache_dir).to(device)

def run_saliency_crop(image, user_prompt, save_output=False, output_path=None, padding=50):
    """
    Generates a Saliency Heatmap and returns a collage of the 
    Heatmap Overlay (top) and the Original image (bottom).
    """

    # Convert PIL to OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Keep original copy for collage
    original_image = image.copy()
    h_orig, w_orig = original_image.shape[:2]

    # 1. Prepare for CLIPSeg
    pil_input = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    inputs = processor(
        text=[user_prompt], 
        images=[pil_input], 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    # 2. Inference: Extract Heatmap
    with torch.no_grad():
        outputs = model(**inputs)
    
    heatmap = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (w_orig, h_orig))
    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Locate Saliency Center (Contour Logic)
    _, thresh = cv2.threshold(heatmap_normalized, int(255 * 0.7), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Calculate crop coordinates (for potential use or saving)
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w_orig, x + w_box + padding), min(h_orig, y + h_box + padding)
        # To strictly follow the 'Annotated' vs 'Original' structure, 
        # we generate the heatmap overlay as the annotated_image.
        colormap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        annotated_image = cv2.addWeighted(original_image, 0.5, colormap, 0.5, 0)
    else:
        annotated_image = original_image.copy()

    # Save if requested (Saving the heatmap overlay)
    if save_output and output_path:
        cv2.imwrite(output_path, annotated_image)

# -------- COLLAGE PART --------

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(original_image, (w_orig, h_orig // 2))
    annot_resized = cv2.resize(annotated_image, (w_orig, h_orig // 2))

    # Vertical collage: Heatmap on top, Original on bottom
    collage = np.vstack((annot_resized, orig_resized))

    # Convert final collage back to PIL-friendly RGB
    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("Saliency Crop completed. Returning collage image.")
    return Image.fromarray(collage_rgb)