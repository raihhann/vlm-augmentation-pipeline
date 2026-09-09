"""
Latent Space K-Means Clustering:
Clusters deep CLIP visual embeddings into K distinct semantic groups and retrieves the closest medoid frame for each centroid.
"""

from typing import List

import numpy as np
from sklearn.cluster import KMeans

from .utils import get_clip_embeddings, load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_latent_kmeans",
    **kwargs,
) -> List[str]:
    """Extracts keyframes from a video using K-Means clustering on visual embeddings.

    This function extracts candidate frames, computes their CLIP embeddings,
    clusters the embeddings into `max_frames` clusters using K-Means, and selects
    the closest medoid frame for each cluster. If the total number of candidate
    frames is less than or equal to `max_frames`, all candidate frames are returned.

    Args:
        video_path: Path to the input video file.
        max_frames: Maximum number of keyframes to extract. Defaults to 8.
        output_folder: Path to the directory where extracted keyframes will be saved.
            Defaults to "output_latent_kmeans".
        **kwargs: Additional keyword arguments.

    Returns:
        List[str]: A list of file paths to the saved keyframes.
    """
    frames, _ = load_candidate_frames(video_path)
    if len(frames) <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="kmeans")

    img_embeds, _ = get_clip_embeddings(frames)
    k = min(max_frames, len(frames))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(img_embeds)
    selected = [
        np.argmin(np.linalg.norm(img_embeds - c, axis=1))
        for c in kmeans.cluster_centers_
    ]
    return save_extracted_frames(
        [frames[i] for i in sorted(list(set(selected)))],
        output_folder,
        prefix="kmeans",
    )