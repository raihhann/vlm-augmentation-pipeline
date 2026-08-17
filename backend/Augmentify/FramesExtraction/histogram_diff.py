"""
HSV Color Histogram Dissimilarity:
Measures color distribution shifts between adjacent frames using Bhattacharyya distance, ranking the highest deviations.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_hist",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="hist")

    hists = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hists.append(hist)

    diffs = [0.0] + [
        cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        for i in range(1, len(hists))
    ]
    top_indices = sorted(np.argsort(diffs)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="hist"
    )