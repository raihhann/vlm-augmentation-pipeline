"""
Katna Multi-Stage Extraction:
Downsamples video to prevent memory spikes, applies LUV color-space clustering (K-Means),
and runs brightness/blur filtering to extract optimal keyframes.
"""

import glob
import os
import tempfile
from typing import List

import cv2
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


def _preprocess_downsample_video(
    input_path: str, temp_output_path: str, target_height: int = 540
) -> str:
    """Downscales high-resolution video to low-RAM resolution before Katna processing."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_path}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if orig_height <= target_height:
        cap.release()
        return input_path

    scale = target_height / float(orig_height)
    new_width = int(orig_width * scale)
    new_height = target_height

    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        out.write(resized)

    cap.release()
    out.release()
    return temp_output_path


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_katna",
    target_height: int = 540,
    **kwargs,
) -> List[str]:
    """
    Extracts keyframes using Katna with automatic RAM-safe resolution scaling.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix="_downsampled.mp4")
    os.close(temp_video_fd)

    processed_video_path = video_path

    try:
        processed_video_path = _preprocess_downsample_video(
            input_path=video_path,
            temp_output_path=temp_video_path,
            target_height=target_height,
        )

        vd = Video()
        disk_writer = KeyFrameDiskWriter(location=output_folder)

        vd.extract_video_keyframes(
            no_of_frames=max_frames,
            file_path=processed_video_path,
            writer=disk_writer,
        )

        extracted_files = sorted(
            glob.glob(os.path.join(output_folder, "*.jpeg"))
            + glob.glob(os.path.join(output_folder, "*.jpg"))
            + glob.glob(os.path.join(output_folder, "*.png"))
        )

        return [os.path.abspath(f) for f in extracted_files[:max_frames]]

    finally:
        if (
            processed_video_path != video_path
            and os.path.exists(temp_video_path)
        ):
            os.remove(temp_video_path)