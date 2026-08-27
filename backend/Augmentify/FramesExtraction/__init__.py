"""Expose the unified keyframe-extraction interface and registered strategies.

The package maps twenty pixel-, motion-, feature-, and query-aware extractors
to method names and saves their selected video frames through one entry point.
"""

from typing import List

from .bitwise_xor import extract_keyframes as extract_bitwise_xor
from .bolt import extract_keyframes as extract_bolt
from .covariance_eigen import extract_keyframes as extract_covariance
from .curvature_points import extract_keyframes as extract_curvature
from .delaunay_graph import extract_keyframes as extract_delaunay
from .dense_optical_flow import extract_keyframes as extract_dense_flow
from .histogram_diff import extract_keyframes as extract_histogram
from .iframes import extract_keyframes as extract_iframes
from .katna_extractor import extract_keyframes as extract_katna
from .keyvideollm import extract_keyframes as extract_keyvideollm
from .laplacian_sharpness import extract_keyframes as extract_laplacian
from .latent_kmeans import extract_keyframes as extract_kmeans
from .maxinfo_svd import extract_keyframes as extract_maxinfo
from .peak_signal import extract_keyframes as extract_peak_signal
from .scenedetect import extract_keyframes as extract_scenedetect
from .sparse_optical_flow import extract_keyframes as extract_sparse_flow
from .ssim_filter import extract_keyframes as extract_ssim
from .tsdpc import extract_keyframes as extract_tsdpc
from .query_aware_iframes import extract_keyframes as extract_qa_iframes
from .smart_sampling import extract_keyframes as extract_smart_sampling

FRAME_EXTRACTORS = {
    "iframes": extract_iframes,
    "scenedetect": extract_scenedetect,
    "peak_signal": extract_peak_signal,
    "ssim": extract_ssim,
    "histogram": extract_histogram,
    "laplacian": extract_laplacian,
    "curvature": extract_curvature,
    "bitwise_xor": extract_bitwise_xor,
    "covariance": extract_covariance,
    "sparse_flow": extract_sparse_flow,
    "dense_flow": extract_dense_flow,
    "katna": extract_katna,
    "kmeans": extract_kmeans,
    "maxinfo": extract_maxinfo,
    "tsdpc": extract_tsdpc,
    "delaunay": extract_delaunay,
    "keyvideollm": extract_keyvideollm,
    "bolt": extract_bolt,
    "qa_iframes": extract_qa_iframes,
    "smart_sampling": extract_smart_sampling,
}


def extract_frames(
    video_path: str,
    method: str = "bolt",
    max_frames: int = 8,
    prompt: str = None,
    output_folder: str = "extracted_frames",
    **kwargs,
) -> List[str]:
    """Unified entry point for keyframe extraction."""
    if method not in FRAME_EXTRACTORS:
        raise ValueError(
            f"Method '{method}' is invalid. Available: {list(FRAME_EXTRACTORS.keys())}"
        )

    return FRAME_EXTRACTORS[method](
        video_path=video_path,
        max_frames=max_frames,
        prompt=prompt,
        output_folder=output_folder,
        **kwargs,
    )