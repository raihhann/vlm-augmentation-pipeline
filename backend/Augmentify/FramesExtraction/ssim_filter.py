"""Select frames whose structural similarity to the previous selected frame falls below a threshold.

The extractor samples candidate frames, keeps the first frame, and persists
subsequent frames that represent a sufficiently large visual change.
"""

from typing import List

import cv2
from skimage.metrics import structural_similarity as ssim

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_ssim",
    threshold: float = 0.70,
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    selected, prev_gray = [], None
    for frame in frames:
        if len(selected) >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            selected.append(frame)
            prev_gray = gray
        else:
            score, _ = ssim(prev_gray, gray, full=True)
            if score < threshold:
                selected.append(frame)
                prev_gray = gray
    return save_extracted_frames(selected, output_folder, prefix="ssim")