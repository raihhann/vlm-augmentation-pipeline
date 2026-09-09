"""
Dense Optical Flow Energy (Farneback):
Calculates the magnitude of full-field pixel velocity vectors, capturing frames at the highest dynamic energy states.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_dense_flow",
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
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="dense_flow")

    energies = [0.0]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energies.append(np.mean(mag**2))
        prev_gray = curr_gray

    top_indices = sorted(np.argsort(energies)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="dense_flow"
    )