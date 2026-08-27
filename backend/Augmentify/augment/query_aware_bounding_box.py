"""Extract a visual query and draw a CLIPSeg-based region-of-interest box."""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import ollama

# Initialize CLIPSeg model once at module level
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
MODEL = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)


def _get_visual_query(user_prompt: str) -> str:
    """Uses LLM to convert a question/prompt into a visual object name."""
    system_prompt = (
        "Convert the user question into a VISUAL object that can be seen in an image.\n"
        "RULES:\n"
        "- Output ONLY what should be visible in the image\n"
        "- No explanations or abstract ideas\n"
        "- Focus on objects like: screen, sign, board, text, display, monitor, document, plant"
    )
    try:
        response = ollama.chat(
            model="llava-phi3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0, "num_predict": 15}
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        print(f"[Planner Error] {e}")
        return "object"


def _get_roi_bounding_box(image_bgr: np.ndarray, visual_query: str, threshold: int = 110, thickness: int = 3) -> np.ndarray:
    """Detects target region using CLIPSeg and draws a bounding box."""
    h, w = image_bgr.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    inputs = PROCESSOR(
        text=[visual_query],
        images=[pil_image],
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)

    heatmap = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    ys, xs = np.where(heatmap > threshold)

    if len(xs) == 0 or len(ys) == 0:
        print("-> [ROI] No region detected above threshold.")
        return image_bgr.copy()

    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), thickness)
    return canvas


def run_query_aware_bounding_box(image, prompt="object", save_output=False, output_path=None):
    """
    Applies Query-Aware Bounding Box detection based on text prompt.
    Returns stacked PIL Image (Original on top, Augmented on bottom).
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # 1. Plan target visual query
    visual_query = _get_visual_query(prompt)
    print(f"-> [Visual Target]: {visual_query}")

    # 2. Draw ROI bounding box
    augmented = _get_roi_bounding_box(original_image, visual_query)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))

    print("Query Aware Bounding Box completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))