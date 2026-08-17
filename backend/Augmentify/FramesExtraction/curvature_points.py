"""
Curvature Points CP (Ciocca & Schettini, 2006):
Tracks the cumulative difference trajectory over time and extracts keyframes at corner inflection points.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_curvature",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    n_frames = len(frames)
    if n_frames <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="curvature")

    diffs = [0.0]
    for i in range(1, n_frames):
        g1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diffs.append(np.mean(np.abs(g1.astype(float) - g2.astype(float))))

    cum_curve = np.cumsum(diffs)
    corners = (
        [0]
        + list(
            np.where(
                np.diff(np.diff(cum_curve)) > np.std(cum_curve) * 0.1
            )[0]
        )
        + [n_frames - 1]
    )
    midpoints = [
        (corners[i] + corners[i + 1]) // 2 for i in range(len(corners) - 1)
    ]
    selected_indices = sorted(list(set(midpoints)))[:max_frames]
    return save_extracted_frames(
        [frames[i] for i in selected_indices], output_folder, prefix="curvature"
    )