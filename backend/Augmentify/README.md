# Augmentify

`Augmentify` is the research package containing image augmentation algorithms and video keyframe-extraction strategies used by the backend evaluation service.

## Package responsibilities

The package is split into two implementation areas:

- [`augment/`](augment/README.md) contains image transformations, model-based visual analysis, annotation utilities, and proposed robustness augmentations.
- [`FramesExtraction/`](FramesExtraction/README.md) contains algorithms that select representative frames from videos before VLM inference.
- [`FutureWork/`](FutureWork/README.md) contains planned extensions, empty module boilerplates, and design notes for work that has not been implemented yet.

The package-level API for video processing is exported from `FramesExtraction`. Image processing is currently routed by `backend/augmentation.py`.

## How it fits into the backend

The request pipeline in `backend/main.py` follows this general sequence:

1. Receive an uploaded image, video, or dataset row.
2. Run baseline VLM inference through `backend/inference.py`.
3. Call an image augmenter from `augment/`, or extract video frames through `FramesExtraction`.
4. Optionally augment extracted video frames.
5. Run VLM inference on the processed media.
6. Compare both responses with `backend/evaluation.py`.

## Image augmentation contract

Most image functions accept a PIL image or an OpenCV/NumPy image, perform one transformation or model-based annotation, and return a PIL image. Several functions also support optional output persistence through `save_output` and `output_path`.

The dispatcher does not currently expose every implementation in the folder. Check `backend/config.py` and `backend/augmentation.py` before treating an algorithm as selectable from the UI.

## Video extraction contract

The extraction package exposes:

```python
from Augmentify.FramesExtraction import extract_frames

paths = extract_frames(
	video_path="input.mp4",
	method="bolt",
	max_frames=8,
	prompt="describe the road signs",
	output_folder="output_frames",
)
```

Each registered strategy returns a list of saved frame paths. Shared decoding, persistence, and CLIP embedding behavior lives in [`FramesExtraction/utils.py`](FramesExtraction/utils.py).

## Model and resource considerations

Some modules load neural network weights at import time, while others initialize models lazily. This can make importing a module expensive and can require local files under the repository `models/` directory. Keep model paths and cache behavior in mind when adding a new method.

## Adding a new method

For a new image augmenter:

1. Add the implementation under `augment/`.
2. Add a module-level docstring describing its purpose and model requirements.
3. Add a dispatcher branch in `backend/augmentation.py`.
4. Add the user-facing name to `backend/config.py` only when the branch is usable.
5. Run a small image experiment and verify that the return type is compatible with the backend.

For a new frame extractor, follow the package guidance in [`FramesExtraction/README.md`](FramesExtraction/README.md), then register the method in `FramesExtraction/__init__.py` and `backend/config.py`.
