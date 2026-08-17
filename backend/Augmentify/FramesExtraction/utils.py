"""
Utility functions for video frame decoding, file persistence, 
and cached deep multimodal feature extraction.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import PIL.Image
import torch
from transformers import CLIPModel, CLIPProcessor

# Prevent multithreading memory collisions
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_CLIP_CACHE = {}


def load_candidate_frames(
    video_path: str, target_fps: float = 2.0
) -> Tuple[List[np.ndarray], List[float]]:
    """Loads video frames sampled at a uniform target FPS along with timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hop = max(1, int(fps / target_fps))

    frames, timestamps = [], []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % hop == 0:
            frames.append(frame)
            timestamps.append(count / fps)
        count += 1
    cap.release()
    return frames, timestamps


def save_extracted_frames(
    frames: List[np.ndarray], output_folder: str, prefix: str = "frame"
) -> List[str]:
    """Saves selected frames into output folder and returns their absolute file paths."""
    os.makedirs(output_folder, exist_ok=True)
    saved_paths = []
    for idx, frame in enumerate(frames):
        file_path = os.path.join(output_folder, f"{prefix}_{idx:03d}.jpg")
        cv2.imwrite(file_path, frame)
        saved_paths.append(os.path.abspath(file_path))
    return saved_paths


def get_clip_embeddings(
    candidate_frames: List[np.ndarray],
    prompt: Optional[str] = None,
    model_name: str = "openai/clip-vit-base-patch32",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generates L2-normalized image and text embeddings using a cached CLIP model."""
    global _CLIP_CACHE
    if model_name not in _CLIP_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        _CLIP_CACHE[model_name] = (model, processor, device)

    model, processor, device = _CLIP_CACHE[model_name]

    pil_imgs = [
        PIL.Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        for f in candidate_frames
    ]
    inputs = processor(images=pil_imgs, return_tensors="pt").to(device)

    with torch.no_grad():
        img_out = model.get_image_features(**inputs)
        img_tensor = (
            img_out.pooler_output
            if hasattr(img_out, "pooler_output")
            else (img_out.image_embeds if hasattr(img_out, "image_embeds") else img_out)
        )
        img_embeds = (
            (img_tensor / img_tensor.norm(dim=-1, keepdim=True)).cpu().numpy()
        )

    text_embeds = None
    if prompt:
        text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            text_out = model.get_text_features(**text_inputs)
            text_tensor = (
                text_out.pooler_output
                if hasattr(text_out, "pooler_output")
                else (
                    text_out.text_embeds
                    if hasattr(text_out, "text_embeds")
                    else text_out
                )
            )
            text_embeds = (
                (text_tensor / text_tensor.norm(dim=-1, keepdim=True))
                .cpu()
                .numpy()
            )

    return img_embeds, text_embeds