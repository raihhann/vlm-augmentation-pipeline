"""
Covariance Minimum Eigenvalue (Benni et al., 2015):
Computes covariance matrices on low-resolution image vectors, identifying shot boundaries via eigenvalue variations.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_covariance",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="coveig")

    scores = [0.0]
    for i in range(1, len(frames)):
        f1 = cv2.resize(
            cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY), (32, 32)
        ).flatten()
        f2 = cv2.resize(
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (32, 32)
        ).flatten()
        cov = np.cov(f1, f2)
        eigvals, _ = np.linalg.eig(cov)
        scores.append(-np.min(np.real(eigvals)))

    top_indices = sorted(np.argsort(scores)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="coveig"
    )