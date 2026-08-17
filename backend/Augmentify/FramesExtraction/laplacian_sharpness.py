"""
Laplacian Variance Sharpness:
Measures high-frequency spatial edge gradients to extract the sharpest and least motion-blurred frames.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_sharpness",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="sharp")

    scores = [
        cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        for f in frames
    ]
    top_indices = sorted(np.argsort(scores)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="sharp"
    )