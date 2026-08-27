# Image Augmentations

This folder contains image transformations and model-based visual processing functions used to test VLM robustness. Implementations generally accept a PIL image or an OpenCV/NumPy image and return a PIL image, often arranged as an original/processed collage for inspection.

## Implementations

| Module | Main operation |
| --- | --- |
| `color_jittering.py` | Brightness and contrast adjustment. |
| `ContextAware_Zoom.py` | CLIPSeg-based prompt region detection followed by a padded zoom. |
| `depth_anything.py` | Monocular depth estimation with Depth Anything. |
| `detr.py` | RT-DETR object detection and annotation. |
| `elastic_deformation.py` | Smooth random elastic image warping. |
| `fastSAM.py` | FastSAM segmentation with masks, boxes, and labels. |
| `geometric_segmentation.py` | YOLO segmentation and region annotation. |
| `geometric_transformation.py` | Image flipping and rotation. |
| `gridmask_dropout.py` | Regular GridMask region dropout. |
| `Illumination.py` | Gamma correction, CLAHE, Retinex, and log transforms. |
| `kernel_blurring_sharpening.py` | Kernel-based blur or sharpening. |
| `midas.py` | MiDaS monocular depth estimation. |
| `mobileSAM.py` | MobileSAM segmentation and annotation. |
| `mosaic_augmentation.py` | Mosaic-style tile transformation. |
| `noise_injection.py` | Random noise addition. |
| `perception_metric_grounding.py` | Prompt-grounded detection and depth metric overlay. |
| `perspective_transform.py` | Corner-based perspective distortion. |
| `pose_estimation.py` | YOLO human pose and skeleton annotation. |
| `query_aware_bounding_box.py` | Prompt extraction and CLIPSeg ROI bounding box. |
| `random_cropping_resizing.py` | Random crop resized to the source dimensions. |
| `random_erasing_cutout.py` | Random rectangular patch erasure. |
| `rembg.py` | Background removal with `rembg`. |
| `SailencyCrop.py` | Prompt-guided saliency crop. |
| `scene_graph_genration.py` | YOLO-World detection, REACT++ relations, and graph rendering. |
| `semantic_hazard_isolation.py` | Prompt-related hazard isolation and overlay. |
| `shear_mapping.py` | Horizontal shear transformation. |
| `solarization.py` | Threshold-based photographic solarization. |
| `style_transfer_filter.py` | Fixed image stylization filter. |
| `surface_normalization.py` | Transformer-based surface/depth normalization. |
| `test_time_augmentation.py` | Generation of transformed test-time variants. |
| `weather_fog_simulation.py` | Atmospheric fog simulation. |
| `yolo_world_annotate.py` | YOLO-World open-vocabulary detection and annotation. |
| `zoe_depth.py` | ZoeDepth monocular depth estimation. |

## What is currently enabled

The web-facing dispatcher in [`backend/augmentation.py`](../../augmentation.py) and the option list in [`backend/config.py`](../../config.py) determine what can be selected from the UI. The source folder contains more implementations than the current UI exposes. A module is not automatically active merely because it exists here.

## Common interface

A typical augmenter has a function shaped like this:

```python
def run_method(image, save_output=False, output_path=None):
    ...
    return pil_image
```

Some methods also accept a prompt or algorithm-specific parameters. Preserve the return type expected by the backend, and keep optional saving behavior consistent with neighboring modules.

## Model-loading notes

Model-based modules may load weights during import or on first call. This can make startup slow and can require model files, Hugging Face cache access, Ollama, or CPU/GPU memory. Use explicit local paths where the surrounding module already does so, and document new model requirements in the module docstring.

## Adding an augmenter

1. Implement the transformation in a dedicated module with a top-level multiline docstring.
2. Test it directly with a representative PIL image.
3. Add a branch to `backend/augmentation.py`.
4. Add its display name to `backend/config.py` only after the branch works.
5. Confirm that saved output paths and image dimensions behave correctly in the dashboard workflow.
