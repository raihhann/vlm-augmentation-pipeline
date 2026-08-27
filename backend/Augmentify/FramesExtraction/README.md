# Video Frame Extraction

`FramesExtraction` provides a common interface for selecting representative keyframes from videos. Each strategy exposes an `extract_keyframes` function and returns paths to saved image files.

## Public API

The package registry and dispatcher are defined in [`__init__.py`](__init__.py):

```python
from Augmentify.FramesExtraction import extract_frames

frame_paths = extract_frames(
    video_path="input.mp4",
    method="bolt",
    max_frames=8,
    prompt="find the traffic signs",
    output_folder="extracted_frames",
)
```

An invalid method raises `ValueError`. The selected extractor receives `video_path`, `max_frames`, `prompt`, `output_folder`, and any additional keyword arguments.

## Available strategies

| Method | Module | Selection principle |
| --- | --- | --- |
| `iframes` | `iframes.py` | Native codec I-frames from the video bitstream. |
| `scenedetect` | `scenedetect.py` | Adaptive shot-boundary detection. |
| `peak_signal` | `peak_signal.py` | Prominent peaks in inter-frame pixel differences. |
| `ssim` | `ssim_filter.py` | Frames whose structural similarity drops below a threshold. |
| `histogram` | `histogram_diff.py` | HSV color-distribution changes. |
| `laplacian` | `laplacian_sharpness.py` | High-frequency sharpness and blur resistance. |
| `curvature` | `curvature_points.py` | Inflection points in the cumulative difference curve. |
| `bitwise_xor` | `bitwise_xor.py` | Bitwise-XOR dissimilarity between consecutive frames. |
| `covariance` | `covariance_eigen.py` | Covariance minimum-eigenvalue variation. |
| `sparse_flow` | `sparse_optical_flow.py` | Shi-Tomasi corners tracked with Lucas-Kanade flow. |
| `dense_flow` | `dense_optical_flow.py` | Farneback full-field motion energy. |
| `katna` | `katna_extractor.py` | Katna multi-stage color, brightness, blur, and clustering selection. |
| `kmeans` | `latent_kmeans.py` | CLIP latent-space K-Means medoids. |
| `maxinfo` | `maxinfo_svd.py` | SVD/MaxVol geometric diversity selection. |
| `tsdpc` | `tsdpc.py` | Temporal segment density-peaks clustering. |
| `delaunay` | `delaunay_graph.py` | PCA and Delaunay-graph connected-component medoids. |
| `keyvideollm` | `keyvideollm.py` | CLIP prompt-to-frame similarity ranking. |
| `bolt` | `bolt.py` | Prompt alignment scores sampled through a cumulative distribution. |
| `qa_iframes` | `query_aware_iframes.py` | Prompt-grounded ranking of codec I-frames with CLIPSeg. |
| `smart_sampling` | `smart_sampling.py` | Prompt noun extraction and YOLO-World object grounding. |

## Shared utilities

[`utils.py`](utils.py) contains the shared operations used by most strategies:

- `load_candidate_frames` samples video frames at a target FPS and returns frames with timestamps.
- `save_extracted_frames` writes selected frames as numbered JPEGs and returns absolute paths.
- `get_clip_embeddings` caches CLIP models and returns normalized image and optional text embeddings.

PyAV is used by the codec-aware extractors. Other methods use OpenCV, SciPy, scikit-image, scikit-learn, Katna, Transformers, or Ultralytics according to their implementation.

## Adding a strategy

1. Create a module with a module-level docstring describing the algorithm.
2. Implement `extract_keyframes(video_path, max_frames=8, prompt=None, output_folder=..., **kwargs)`.
3. Use the shared utilities when the algorithm samples and saves ordinary image frames.
4. Register the function in `FramesExtraction/__init__.py`.
5. Add a readable label in `backend/config.py`.
6. Test empty videos, videos with fewer than `max_frames` frames, and normal output-path creation.

The keyframe budget is a limit, not a guarantee that every method returns exactly `max_frames`; strategies that deduplicate or filter frames may return fewer.
