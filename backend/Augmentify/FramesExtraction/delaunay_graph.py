"""
Dynamic Delaunay Graph Clustering (Kuanar et al., 2013):
Projects visual features onto a PCA subspace, constructs a Delaunay graph triangulation, and extracts medoids of connected components.
"""

from typing import List

import cv2
import numpy as np
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

from .utils import load_candidate_frames, save_extracted_frames


def extract_keyframes(
    video_path: str,
    max_frames: int = 8,
    output_folder: str = "output_delaunay",
    **kwargs,
) -> List[str]:
    frames, _ = load_candidate_frames(video_path)
    n_frames = len(frames)
    if n_frames <= max_frames:
        return save_extracted_frames(frames, output_folder, prefix="delaunay")

    feats = np.array(
        [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 32)).flatten()
            for f in frames
        ]
    )
    pca = PCA(n_components=min(3, n_frames - 1))
    pts = pca.fit_transform(feats)
    tri = Delaunay(pts)

    edges = set()
    for sim in tri.simplices:
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                edges.add((min(sim[i], sim[j]), max(sim[i], sim[j])))

    adj = {i: [] for i in range(n_frames)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited, medoids = set(), []
    for i in range(n_frames):
        if i not in visited:
            comp, q = [], [i]
            visited.add(i)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)
            medoids.append(comp[len(comp) // 2])

    return save_extracted_frames(
        [frames[i] for i in sorted(medoids)[:max_frames]],
        output_folder,
        prefix="delaunay",
    )