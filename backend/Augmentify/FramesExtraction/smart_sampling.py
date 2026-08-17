"""
Smart Sampling (Open-Vocabulary Grounded Keyframe Extractor):
1. Parses noun targets from the evaluation prompt using spaCy NLP.
2. Evaluates candidate video frames using open-vocabulary YOLO-World.
3. Overlays bounding boxes and timestamps on detected target objects.
4. Ranks frames by detection confidence and outputs chronologically up to max_frames.
"""

import os
from typing import List

import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLOWorld

_NLP_CACHE = None
_YOLO_CACHE = None


def _get_nlp_model():
    """Lazy loader for spaCy English pipeline."""
    global _NLP_CACHE
    if _NLP_CACHE is None:
        import spacy
        try:
            _NLP_CACHE = spacy.load("en_core_web_sm")
        except Exception:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _NLP_CACHE = spacy.load("en_core_web_sm")
    return _NLP_CACHE


def _get_yolo_model(model_name: str = "yolov8s-world.pt"):
    """Singleton cache loader for YOLO-World."""
    global _YOLO_CACHE
    if _YOLO_CACHE is None:
        _YOLO_CACHE = YOLOWorld(model_name)
    return _YOLO_CACHE


def _extract_target_nouns(prompt: str) -> List[str]:
    """Extracts entity nouns from prompt, excluding generic trigger words."""
    nlp = _get_nlp_model()
    doc = nlp(prompt)
    nouns = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
    
    ignore_words = {
        "what", "where", "who", "how", "describe", "identify", "locate", 
        "scene", "image", "video", "person", "object", "detail", "find"
    }
    filtered = [n for n in nouns if n not in ignore_words and len(n) > 2]
    return filtered if filtered else [prompt]


def extract_keyframes(
    video_path: str,
    prompt: str = "main subject objects",
    max_frames: int = 8,
    output_folder: str = "output_smart_sampling",
    confidence_thresh: float = 0.25,
    **kwargs,
) -> List[str]:
    """
    Extracts and annotates keyframes containing prompt-specified objects using YOLO-World Smart Sampling.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    # 1. Parse target nouns
    target_classes = _extract_target_nouns(prompt)

    # 2. Configure YOLO-World with prompt vocabulary
    model = _get_yolo_model("yolov8s-world.pt")
    model.set_classes(target_classes)

    # 3. Read video and compute sampling timeline
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample at ~1 fps + boundary endpoints
    sampled_indices = {0, max(0, total_frames - 1)}
    step = max(1, int(fps))
    for idx in range(0, total_frames, step):
        sampled_indices.add(idx)

    sorted_indices = sorted(list(sampled_indices))
    candidate_results = []

    # 4. Detect and annotate candidate frames
    for frame_idx in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp_sec = round(frame_idx / fps, 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        results = model.predict(pil_img, conf=confidence_thresh, verbose=False)[0]

        if len(results.boxes) > 0:
            top_conf = float(results.boxes.conf.max().cpu().item())
            annotated_img = pil_img.copy()
            draw = ImageDraw.Draw(annotated_img)

            # Draw timestamp badge
            timestamp_text = f" Time: {timestamp_sec:.1f}s "
            draw.rectangle([10, 10, 150, 38], fill="black")
            draw.text((15, 16), timestamp_text, fill="yellow")

            # Draw bounding boxes
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().item())
                label_name = target_classes[cls_id] if cls_id < len(target_classes) else "object"

                draw.rectangle(xyxy.tolist(), outline="red", width=3)
                draw.text((xyxy[0] + 5, max(0, xyxy[1] - 15)), f"{label_name}", fill="red")

            candidate_results.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp_sec,
                "confidence": top_conf,
                "image": annotated_img,
            })

    cap.release()

    # 5. Fallback if no target entities were detected
    if not candidate_results:
        cap = cv2.VideoCapture(video_path)
        mid_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fallback_img = Image.fromarray(rgb_frame)
            fallback_path = os.path.join(output_folder, "smart_sampling_000.jpg")
            fallback_img.save(fallback_path)
            return [os.path.abspath(fallback_path)]
        return []

    # 6. Rank by confidence, take top max_frames, and sort chronologically
    candidate_results = sorted(candidate_results, key=lambda x: x["confidence"], reverse=True)[:max_frames]
    candidate_results = sorted(candidate_results, key=lambda x: x["timestamp"])

    saved_paths = []
    for idx, item in enumerate(candidate_results):
        file_path = os.path.join(output_folder, f"smart_sampling_{idx:03d}.jpg")
        item["image"].save(file_path)
        saved_paths.append(os.path.abspath(file_path))

    return saved_paths