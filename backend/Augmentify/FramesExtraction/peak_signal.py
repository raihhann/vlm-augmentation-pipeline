"""Extract keyframes at prominent peaks in an inter-frame pixel-difference signal.

OpenCV grayscale differences are analyzed with SciPy peak detection, with a
highest-difference fallback when no local peaks meet the prominence threshold.
"""

from typing import List

import cv2
import numpy as np
from scipy.signal import find_peaks

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_peaksignal",
    target_fps: float = 2.0,
    **kwargs,
) -> List[str]:
    """
    Extracts keyframes at local difference peaks in the temporal signal.
    """
    frames, _ = load_candidate_frames(video_path, target_fps=target_fps)
    if not frames:
        return []

    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="peak")

    # Step 1: Compute 1D inter-frame difference signal
    diff_signal = [0.0]
    for i in range(1, len(frames)):
        g1 = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = np.mean(cv2.absdiff(g1, g2))
        diff_signal.append(diff)

    diff_signal = np.array(diff_signal)

    # Step 2: Detect local peaks in the difference signal
    peaks, properties = find_peaks(
        diff_signal,
        distance=max(1, len(frames) // (max_frames * 2)),
        prominence=np.std(diff_signal) * 0.2,
    )

    # Step 3: Select top-K peaks by prominence or fallback to highest signal values
    if len(peaks) > 0:
        prominences = properties.get("prominences", diff_signal[peaks])
        top_peak_indices = peaks[np.argsort(prominences)[-max_frames:]]
        selected_indices = sorted(list(top_peak_indices))
    else:
        # Fallback if no prominent local peaks are identified
        selected_indices = sorted(np.argsort(diff_signal)[-max_frames:])

    selected_frames = [frames[i] for i in selected_indices]
    return save_extracted_frames(selected_frames, output_folder, prefix="peak")