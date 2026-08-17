"""
Codec Native I-Frames Extraction (PyAV Engine):
Extracts genuine intra-coded keyframes (I-Frames) directly from the video 
bitstream by skipping non-key packet decoding at the container level.
"""

import os
from typing import List

import av
import numpy as np

from .utils import save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_iframes",
    **kwargs,
) -> List[str]:
    """
    Extracts native codec I-Frames using PyAV and uniformly samples up to max_frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    container = av.open(video_path)
    stream = container.streams.video[0]
    
    # Instruct codec to skip decoding non-keyframes (P-frames and B-frames)
    stream.codec_context.skip_frame = "NONKEY"

    frames = []
    try:
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="bgr24")
            frames.append(img)
    finally:
        container.close()

    if not frames:
        return []

    # If keyframes found are within budget, return all of them
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="iframe")

    # Uniformly downsample to budget K if more I-frames exist than requested
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    selected = [frames[i] for i in indices]

    return save_extracted_frames(selected, output_folder, prefix="iframe")