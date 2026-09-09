"""
BOLT (Inference-Time Inverse Transform Prioritization):
Treats prompt alignment scores as a probability density curve over time, sampling frames evenly across the cumulative distribution (CDF).
"""

from typing import List

import numpy as np

from .utils import get_clip_embeddings, load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    prompt: str = "main subject actions",
    max_frames: int = 8,
    output_folder: str = "output_bolt",
    **kwargs,
) -> List[str]:
    """Extract keyframes from the input data.

    Args:
        video_path (str): Description of the parameter.
        prompt (str): Description of the parameter.
        max_frames (int): Description of the parameter.
        output_folder (str): Description of the parameter.
        kwargs: Additional keyword arguments for the operation.

    Returns:
        list[str]: The extracted frame paths or keyframe results.
    """    
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="bolt")

    img_embeds, text_embeds = get_clip_embeddings(frames, prompt=prompt)
    sims = (img_embeds @ text_embeds.T).squeeze()
    sims = sims - np.min(sims) + 1e-5
    pdf = sims / np.sum(sims)
    cdf = np.cumsum(pdf)

    targets = np.linspace(1.0 / max_frames, 1.0, max_frames)
    selected = [
        min(np.searchsorted(cdf, t), len(frames) - 1) for t in targets
    ]
    return save_extracted_frames(
        [frames[i] for i in sorted(list(set(selected)))],
        output_folder,
        prefix="bolt",
    )