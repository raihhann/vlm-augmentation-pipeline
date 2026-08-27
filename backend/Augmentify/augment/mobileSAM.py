"""Segment image objects with MobileSAM and render the detected masks."""

import os
import cv2
import numpy as np
import supervision as sv
from PIL import Image

# --- FIX: Resolve Absolute Path ---
# Get the directory of the current script
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the shared 'models' folder
model_folder = os.path.abspath(os.path.join(current_file_dir, "..", "..", "..", "models"))

# Ensure the directory exists
os.makedirs(model_folder, exist_ok=True)

# Set environment variables for Ultralytics
os.environ['YOLO_HOME'] = model_folder
os.environ['ULTRALYTICS_CONFIG_DIR'] = model_folder

from ultralytics import SAM

# Define the absolute path to the MobileSAM weights
model_path = os.path.join(model_folder, 'mobile_sam.pt')

print(f"🛠️ Loading MobileSAM from: {model_path}")

# Load the model using the absolute path to prevent stray downloads
mobile_sam_model = SAM(model_path)
def run_mobilesam(image, output_path: str = None):
    """
    Run MobileSAM on an image and return annotated PIL image.

    Args:
        image (str | PIL.Image.Image | np.ndarray): Input image (path, PIL, or OpenCV BGR array)
        output_path (str, optional): If provided, saves annotated image to this path.

    Returns:
        PIL.Image.Image: Annotated image.
    """
    # Convert PIL -> OpenCV BGR if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            print("❌ Image not found")
            return None
    # else assume it's already a NumPy BGR array

    h, w, _ = image.shape

    # Grid points for segmenting everything
    grid_size = 8
    points = [[x, y] for x in range(0, w, max(w // grid_size, 1))
                       for y in range(0, h, max(h // grid_size, 1))]
    labels = [1] * len(points)

    print("Segmenting everything...")
    results_list = mobile_sam_model.predict(image, points=points, labels=labels)
    results = results_list[0]  # single image

    # Convert to Supervision detections
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.area > 400]

    print(f"Found {len(detections)} distinct objects/regions.")

    # Annotators
    mask_annotator = sv.MaskAnnotator(opacity=0.3)
    box_annotator = sv.BoxAnnotator(thickness=1, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_position=sv.Position.CENTER,
        color_lookup=sv.ColorLookup.INDEX
    )

    labels_text = [f"{i}" for i in range(len(detections))]

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_text)

    # Optionally save
    if output_path is not None:
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Saved output to: {output_path}")

# -------- COLLAGE PART --------

    h, w = image.shape[:2]

    # Resize both images to half height while keeping width
    orig_resized = cv2.resize(image, (w, h // 2))
    annot_resized = cv2.resize(annotated_image, (w, h // 2))

    # Vertical collage with same final resolution as original
    collage = np.vstack((orig_resized, annot_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("MobileSAM augmentation completed.")
    return Image.fromarray(collage)