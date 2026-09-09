"""
TSDPC (Temporal Segment Density Peaks Clustering - Tang et al., 2022):
Partitions the video timeline into segments and finds localized feature density peaks based on pairwise embedding distances.
"""

from typing import List

import numpy as np

from .utils import get_clip_embeddings, load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_tsdpc",
    **kwargs,
) -> List[str]:
    """Extract keyframes from the input data.

    Args:
        video_path (str): Description of the parameter.
        max_frames (int): Description of the parameter.
        output_folder (str): Description of the parameter.
        kwargs: Additional keyword arguments for the operation.

    Returns:
        list[str]: The extracted frame paths or keyframe results.
    """    
    frames, _ = load_candidate_frames(video_path)
    n_frames = len(frames)
    if n_frames <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="tsdpc")

    img_embeds, _ = get_clip_embeddings(frames)
    num_seg = max(1, max_frames // 2)
    seg_size = n_frames // num_seg
    selected = []

    for s in range(num_seg):
        st = s * seg_size
        en = n_frames if s == num_seg - 1 else (s + 1) * seg_size
        embeds = img_embeds[st:en]
        if len(embeds) == 0:
            continue
        dists = np.linalg.norm(embeds[:, None, :] - embeds[None, :, :], axis=-1)
        dc = np.percentile(dists, 40) or 1e-5
        rho = np.sum(np.exp(-((dists / dc) ** 2)), axis=1)
        delta = np.array(
            [
                np.min(dists[i, rho > rho[i]])
                if np.any(rho > rho[i])
                else np.max(dists[i])
                for i in range(len(embeds))
            ]
        )
        gamma = rho * delta
        selected.append(st + np.argmax(gamma))

    return save_extracted_frames(
        [frames[i] for i in sorted(list(set(selected)))][:max_frames],
        output_folder,
        prefix="tsdpc",
    )