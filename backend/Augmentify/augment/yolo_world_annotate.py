"""Apply contrast and rotation, then annotate open-vocabulary objects with YOLO-World."""

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

from ultralytics import YOLOWorld

# Define the absolute path to the YOLO-World weights
model_path = os.path.join(model_folder, 'yolov8s-world.pt')

print(f"🛠️ Loading YOLO-World from: {model_path}")

# Load the model using the absolute path
yolo_world_model = YOLOWorld(model_path)

# Define the classes to detect
CLASSES_TO_FIND = [
    "tree", "tree branch", "leaves", "foliage", "bush",
    "view outside window", "nature", "plant",
    "cardboard box", "plastic bag", "electronic device", "furniture", 
    "clothing", "trash", "musical instrument", "container", "bottle",
    "person","pillow","trees","tree","plant","leaves",
]
yolo_world_model.set_classes(CLASSES_TO_FIND)

def annotate(image, output_path: str = None):
    """
    Annotate an image using YOLO-World.

    Args:
        image (str | PIL.Image.Image | np.ndarray): Input image.
        output_path (str, optional): Path to save annotated image.

    Returns:
        PIL.Image.Image: Annotated image.
    """
    # --- Convert to OpenCV BGR if needed ---
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            print(f"❌ Error: Image '{image}' not found.")
            return None
    # else assume it's already a NumPy BGR array

    # --- STEP 1: AUGMENTATION ---
    # Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    aug_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Slight rotation (+2 degrees)
    rows, cols = aug_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 2, 1)
    aug_image = cv2.warpAffine(aug_image, rotation_matrix, (cols, rows))
    print("✅ Image Augmented (Contrast Boost + Rotation applied)")

    # --- STEP 2: SEMANTIC DETECTION ---
    results = yolo_world_model.predict(aug_image, conf=0.1, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Non-Maximum Suppression
    detections = detections.with_nms(threshold=0.5)
    print(f"Found {len(detections)} semantic objects.")

    # --- STEP 3: ANNOTATION ---
    # Box Annotator
    box_annotator = sv.BoxAnnotator(thickness=2)

    # Label Annotator
    labels = [
        f"#{index} {CLASSES_TO_FIND[class_id]} {confidence:0.2f}"
        for index, (class_id, confidence)
        in enumerate(zip(detections.class_id, detections.confidence))
    ]
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
        text_color=sv.Color.BLACK,
        color_lookup=sv.ColorLookup.INDEX
    )

    annotated_frame = box_annotator.annotate(scene=aug_image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # --- STEP 4: OPTIONAL SAVE ---
    if output_path is not None:
        cv2.imwrite(output_path, annotated_frame)
        print(f"🚀 Success! Saved annotated image to: {output_path}")

    # -------- COLLAGE PART --------
    h, w = image.shape[:2]

    # Resize both images to half height
    orig_resized = cv2.resize(image, (w, h // 2))
    annot_resized = cv2.resize(annotated_frame, (w, h // 2))

    # Vertical collage with same final resolution
    collage = np.vstack((orig_resized, annot_resized))

    collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    print("YOLO-World annotation completed.")
    return Image.fromarray(collage)