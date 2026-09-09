"""
Sparse Optical Flow (Shi-Tomasi + Lucas-Kanade):
Tracks prominent visual corner points across time and selects frames corresponding to peak motion vectors.
"""

from typing import List

import cv2
import numpy as np

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_sparse_flow",
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
        return save_extracted_frames(frames, output_folder, prefix="sparse_flow")

    f_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    motion_scores = [0.0]

    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g1, mask=None, **f_params)
        if p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, **lk_params)
            motion = (
                np.mean(np.linalg.norm(p1[st == 1] - p0[st == 1], axis=1))
                if p1 is not None and st is not None
                else 0.0
            )
        else:
            motion = 0.0
        motion_scores.append(motion)

    top_indices = sorted(np.argsort(motion_scores)[-max_frames:])
    return save_extracted_frames(
        [frames[i] for i in top_indices], output_folder, prefix="sparse_flow"
    )