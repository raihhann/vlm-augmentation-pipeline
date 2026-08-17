"""
PySceneDetect (Adaptive Shot Boundary Cut):
Identifies abrupt visual shot changes and camera cuts, extracting the middle frame of each detected scene.
"""

from typing import List

import cv2
from scenedetect import AdaptiveDetector, detect

from .utils import save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_scenedetect",
    **kwargs,
) -> List[str]:
    scene_list = detect(video_path, AdaptiveDetector())
    if not scene_list:
        return []

    cap = cv2.VideoCapture(video_path)
    selected = []
    for scene in scene_list:
        if len(selected) >= max_frames:
            break
        mid_idx = (scene[0].get_frames() + scene[1].get_frames()) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        if ret:
            selected.append(frame)
    cap.release()
    return save_extracted_frames(selected, output_folder, prefix="scenedetect")