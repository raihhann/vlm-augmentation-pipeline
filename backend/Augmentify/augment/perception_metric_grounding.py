"""Overlay prompt-grounded object detections and depth-derived perception metrics."""

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Global model pointers for lazy initialization
YOLO_MODEL = None
ZOE_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _init_models():
    """Initializes YOLOv8 and ZoeDepth once upon first call."""
    global YOLO_MODEL, ZOE_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO("yolov8s.pt")

    if ZOE_MODEL is None:
        # Load ZoeDepth model (ZoeD_NK)
        ZOE_MODEL = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        ZOE_MODEL.eval()
        ZOE_MODEL.to(DEVICE)


def _draw_metric_hud(image_bgr: np.ndarray, boxes, depth_numpy: np.ndarray) -> np.ndarray:
    """Renders green HUD-style metric distance bounding boxes over detected objects."""
    canvas = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    color = (0, 255, 0)  # Glowing Robot Green
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Boundary clipping
        y1_c, y2_c = max(0, y1), min(h, y2)
        x1_c, x2_c = max(0, x1), min(w, x2)

        depth_patch = depth_numpy[y1_c:y2_c, x1_c:x2_c]
        if depth_patch.size == 0:
            continue

        # Extract median metric distance
        obj_dist = float(np.median(depth_patch))
        label = f"{obj_dist:.1f}m"

        # Draw box & HUD background
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 5, y1 - 7), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def run_perception_metric_grounding(image, prompt="", save_output=False, output_path=None):
    """
    Detects COCO objects and overlays real-world metric depth labels.
    Returns stacked PIL Image (Original on top, Augmented on bottom).
    """
    _init_models()

    if isinstance(image, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()

    original_image = image_bgr.copy()
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # 1. Detect objects with YOLOv8
    results = YOLO_MODEL(image_bgr, verbose=False)[0]
    boxes = results.boxes

    # 2. Extract Metric Depth map
    with torch.no_grad():
        depth_numpy = ZOE_MODEL.infer_pil(pil_image)

    # 3. Apply HUD Overlay
    augmented = _draw_metric_hud(original_image, boxes, depth_numpy)

    if save_output and output_path:
        cv2.imwrite(output_path, augmented)

    # -------- COLLAGE PART --------
    h, w = original_image.shape[:2]
    orig_resized = cv2.resize(original_image, (w, h // 2))
    annot_resized = cv2.resize(augmented, (w, h // 2))
    collage = np.vstack((orig_resized, annot_resized))

    print("Perception Metric Grounding completed.")
    return Image.fromarray(cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB))