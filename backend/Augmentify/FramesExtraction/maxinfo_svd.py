"""
MaxInfo (SVD + MaxVol Principle):
Identifies the subset of frames that spans the maximal geometric volume in the latent space to eliminate redundancy.
"""

from typing import List

import numpy as np

from .utils import get_clip_embeddings, load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_maxinfo",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="maxinfo")

    img_embeds, _ = get_clip_embeddings(frames)
    U, S, Vh = np.linalg.svd(img_embeds.T, full_matrices=False)
    sub = Vh[:max_frames, :]
    selected, remaining = [], list(range(len(frames)))

    for _ in range(min(max_frames, len(frames))):
        norms = [np.linalg.norm(sub[:, j]) for j in remaining]
        selected.append(remaining.pop(np.argmax(norms)))

    return save_extracted_frames(
        [frames[i] for i in sorted(selected)], output_folder, prefix="maxinfo"
    )