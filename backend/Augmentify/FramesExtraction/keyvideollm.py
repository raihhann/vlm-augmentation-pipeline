"""
KeyVideoLLM (CLIP Query Top-K Ranking):
Calculates cosine similarity between a natural language prompt and candidate frames, greedily selecting the top highest-scoring frames.
"""

from typing import List

import numpy as np

from .utils import get_clip_embeddings, load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    prompt: str = "main subject actions",
    max_frames: int = 8,
    output_folder: str = "output_keyvideollm",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="keyvideollm")

    img_embeds, text_embeds = get_clip_embeddings(frames, prompt=prompt)
    sims = (img_embeds @ text_embeds.T).squeeze()
    top_indices = sorted(np.argsort(sims)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="keyvideollm"
    )