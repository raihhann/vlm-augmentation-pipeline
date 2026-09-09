"""Random frame sampling strategy serving as a latency-optimized placeholder."""

import os
import random
from typing import List
import cv2

def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_random_sampling",
    **kwargs,
) -> List[str]:
    """Extract random keyframes from the input video data natively using OpenCV.

    [FUTURE WORK]: Update this sampling method to super smart sampling 
    utilizing intelligent query-grounded scoring to optimize frame selection 
    and further reduce downstream inference latency.

    Args:
        video_path (str): Path to the input video file.
        max_frames (int, optional): Maximum number of keyframes to extract. Defaults to 8.
        output_folder (str, optional): Directory path where extracted frames will be saved. Defaults to "output_random_sampling".
        kwargs: Additional keyword arguments for the operation.

    Returns:
        list[str]: The file paths of the randomly selected and saved keyframes.
    """
    # Ensure the destination directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read all frames from the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    n_frames = len(all_frames)
    if n_frames == 0:
        return []

    # Select indices (either all if under budget, or randomly sampled)
    if n_frames <= max_frames:
        selected_indices = list(range(n_frames))
    else:
        selected_indices = sorted(random.sample(range(n_frames), max_frames))

    # Save the selected frames to disk and collect their file paths
    saved_paths = []
    for idx, frame_idx in enumerate(selected_indices):
        file_name = f"random_sample_{idx}_{frame_idx}.jpg"
        out_path = os.path.join(output_folder, file_name)
        cv2.imwrite(out_path, all_frames[frame_idx])
        saved_paths.append(out_path)

    print(f"Extracted {len(saved_paths)} frames to '{output_folder}'.")
    return saved_paths