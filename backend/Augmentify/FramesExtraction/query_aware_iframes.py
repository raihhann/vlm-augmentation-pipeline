"""
Query-Aware I-Frame Extraction (QA-IF):
1. Demuxes intra-coded I-frames directly using PyAV.
2. Decomposes the user query prompt into visual entity nouns via Ollama (e.g., LLaVA-Phi3 / phi3:mini).
3. Evaluates open-vocabulary spatial grounding per I-frame using CLIPSeg.
4. Ranks and returns top surviving keyframes up to max_frames.
"""

import os
from typing import List, Tuple

import av
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

_CLIPSEG_CACHE = {}


def _get_clipseg_model():
    """Singleton cache loader for CLIPSeg visual grounding backbone."""
    global _CLIPSEG_CACHE
    if "model" not in _CLIPSEG_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)
        model.eval()
        _CLIPSEG_CACHE["model"] = (model, processor, device)
    return _CLIPSEG_CACHE["model"]


def _extract_query_entity(query_prompt: str, ollama_model: str = "phi3:mini") -> str:
    """Extracts the primary target visual entity noun phrase using Ollama."""
    try:
        import ollama

        system_instruction = (
            "You are an expert visual entity parser. Extract "
            "the primary visual target object or noun phrase from the prompt. "
            "Output ONLY the raw target entity string without punctuation or explanations."
        )
        full_prompt = f"{system_instruction}\n\nUser Query: '{query_prompt}'\nTarget Visual Entity:"

        response = ollama.generate(
            model=ollama_model,
            prompt=full_prompt,
            options={"temperature": 0.0, "num_predict": 16},
        )
        entity = response["response"].strip().strip("\"'.")
        if entity:
            return entity
    except Exception:
        pass
    return query_prompt


def extract_keyframes(
    video_path: str,
    prompt: str = "main subject actions",
    max_frames: int = 8,
    output_folder: str = "output_qa_iframes",
    presence_threshold: float = 115.0,
    ollama_model: str = "phi3:mini",
    **kwargs,
) -> List[str]:
    """
    Extracts query-relevant I-frames using Ollama entity parsing and CLIPSeg spatial activation.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    # 1. Codec-level Demuxing for I-Frames
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    raw_iframes: List[Tuple[int, Image.Image]] = []
    frame_idx = 0

    try:
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame.key_frame:
                    pil_img = frame.to_image().convert("RGB")
                    raw_iframes.append((frame_idx, pil_img))
                frame_idx += 1
    finally:
        container.close()

    if not raw_iframes:
        return []

    # 2. Decompose user prompt into visual entity target
    target_entity = _extract_query_entity(prompt, ollama_model=ollama_model)

    # 3. Open-Vocabulary Spatial Presence Verification via CLIPSeg
    model, processor, device = _get_clipseg_model()
    scored_keyframes = []

    for f_idx, pil_img in raw_iframes:
        inputs = processor(
            text=[target_entity],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.sigmoid(outputs.logits).cpu().numpy()
        heatmap = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8) * 255.0
        max_activation = float(np.max(heatmap))
        scored_keyframes.append((f_idx, pil_img, max_activation))

    # Filter by threshold, sort by activation descending
    filtered = [k for k in scored_keyframes if k[2] >= presence_threshold]
    if not filtered:
        filtered = scored_keyframes

    filtered.sort(key=lambda x: x[2], reverse=True)
    selected = filtered[:max_frames]

    # Re-order chronologically by frame index
    selected.sort(key=lambda x: x[0])

    # 4. Save keyframes
    saved_paths = []
    for rank_idx, (f_idx, img, score) in enumerate(selected):
        file_path = os.path.join(output_folder, f"qa_iframe_{rank_idx:03d}.jpg")
        img.save(file_path)
        saved_paths.append(os.path.abspath(file_path))

    return saved_paths