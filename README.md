# Vision-Language Model Augmentation Evaluation Pipeline

This repository contains an ongoing University of Stuttgart research prototype for studying how image augmentations and video keyframe-selection strategies affect vision-language model (VLM) outputs. It compares inference on original media with inference on processed media and records linguistic, visual, timing, and token-level metrics.

The project is experimental software rather than a production service. Some modules are complete but not currently exposed in the web UI, some features require a large local model weights, and the `FutureWork` directory documents planned extensions.

## What the application does

The web application supports four related evaluation workflows:

- **Manual image evaluation:** upload multiple images, provide prompts and optional ground-truth descriptions, apply one or more image augmentations, and compare baseline and processed outputs.
- **Manual video evaluation:** upload videos, select keyframe extraction methods, optionally augment the extracted frames, and evaluate the resulting VLM output.
- **Excel image-dataset evaluation:** upload a spreadsheet whose rows point to image files, map filename/prompt/ground-truth columns, and run a repeatable augmentation matrix.
- **Excel video-dataset evaluation:** use spreadsheet rows to run batch video extraction and augmentation experiments.

Progress is streamed to the browser with Server-Sent Events. Results are shown in an evaluation dashboard and can be downloaded as CSV or Excel reports.

## Evaluation metrics

`backend/evaluation.py` aggregates the following values for original and processed outputs:

- **BERTScore F1:** semantic similarity against the ground-truth description.
- **BLEU:** smoothed n-gram overlap against the ground truth.
- **CLIPScore:** reference-free image-text alignment using CLIP, scaled by the configured weight.
- **Latency:** measured inference/processing time and the absolute difference between baseline and processed runs.
- **Token count:** basic whitespace token count for each generated response.

BERTScore and CLIP resources are initialized lazily and cached after their first use. CUDA is selected when available; otherwise the application falls back to CPU.

## Current selectable options

The web UI receives its options from `backend/config.py`.

Currently configured VLM choices are `qwen3-vl:4b`, `llava-phi3`, and `mock`. The live models are accessed through Ollama; `mock` is intended for local workflow checks without a live model response.

The currently configured image augmentation choices are `FastSAM`, `MobileSAM`, `PerceptionMetricGrounding`, and `SceneGraphGeneration`. The source tree contains additional augmentation implementations, but they are not all enabled in the dispatcher and configuration yet.

The video interface registers twenty keyframe strategies, including codec I-frame extraction, scene detection, SSIM, histogram difference, optical flow, CLIP-based ranking, clustering, BOLT, query-aware I-frames, and Smart Sampling. See [backend/Augmentify/FramesExtraction/README.md](backend/Augmentify/FramesExtraction/README.md) for the complete list.

## Requirements and setup

### Prerequisites

- Python 3.10 or newer is recommended.
- An Ollama installation and the desired local VLMs are required for live inference. For example, install the model names configured in `backend/config.py`.
- Sufficient disk space is required for the checked-in and downloaded model weights under `models/`.
- A GPU is optional. PyTorch and the evaluation modules can use CPU, although model-heavy operations will be slower.

### Install dependencies

From the repository root, create and activate a virtual environment, then install the complete dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`backend/Augmentify/requirements.txt` is only a small package-specific list and does not replace the root requirements file.

### Start the application

The backend uses imports that are resolved from the `backend` directory. Start Uvicorn from that directory:

```powershell
cd backend
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in a browser. The application serves the HTML files from `../frontend` and static output from `../static` relative to the backend working directory.

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Serves the manual image/video evaluation page. |
| `GET /excel_studio` | Serves the spreadsheet batch-evaluation page. |
| `POST /process` | Runs the selected workflow and streams progress/results. |
| `GET /download_csv/{session_token}` | Downloads cached or in-memory results as CSV. |
| `GET /download_metrics?session_token=...` | Downloads the current session as an Excel workbook. |

## Repository structure

```text
Software/
├── README.md                         # Project-wide documentation
├── requirements.txt                  # Complete Python dependency list
├── backend/
│   ├── README.md                     # Backend architecture and runtime guide
│   ├── main.py                       # FastAPI application and evaluation routes
│   ├── config.py                     # Models and selectable processing methods
│   ├── inference.py                  # Ollama/VLM inference adapter
│   ├── evaluation.py                 # BERTScore, BLEU, CLIPScore, timing, tokens
│   ├── augmentation.py               # Augmentation dispatcher
│   ├── model_manager.py              # Ollama model preloading helper
│   └── Augmentify/
│       ├── README.md                 # Package overview
│       ├── augment/                  # Image augmentation implementations
│       │   └── README.md
│       ├── FramesExtraction/         # Video keyframe strategies
│       │   └── README.md
│       └── FutureWork/               # Planned modules and research extensions
│           └── README.md
├── frontend/
│   ├── README.md                     # Browser UI documentation
│   ├── index.html                    # Manual image/video workflow
│   ├── excel_evaluator.html          # Spreadsheet workflow
│   └── dashboard.html                # Results dashboard
├── models/                           # Model weights and downloaded repositories
└── static/
    ├── styles.css                    # Shared static styles
    ├── uploads/                      # Runtime uploaded/processed media
    └── cache/                        # Runtime CSV evaluation caches
```

The runtime `static/uploads/` and `static/cache/` directories can grow during experiments. Treat generated files as experiment artifacts and remove them when they are no longer needed. Model files can be very large and should be managed separately from source changes when possible.

## Typical experiment flow

1. Start Ollama and verify that the selected VLM is available.
2. Start the FastAPI application from `backend/`.
3. Open the manual or Excel workflow in the browser.
4. Choose a model, media, prompt, ground truth, and processing method.
5. Wait for the streamed processing run to finish.
6. Inspect original/processed outputs and metrics in the dashboard.
7. Export CSV or Excel results for statistical analysis.

## Development notes

- Image augmentations generally accept a PIL image and return a PIL image, often as an original/processed collage for visual inspection.
- Video extraction functions share the `extract_keyframes(video_path, max_frames, prompt, output_folder, **kwargs)` convention and return saved frame paths.
- The backend currently stores session results in an in-memory dictionary. Restarting the server removes sessions that are not represented by a runtime cache file.
- `backend/model_manager.py` exists as a helper for Ollama model preloading but is not currently part of the main request path.
- There is no comprehensive automated test suite in the repository yet. Validate changes with the local environment and a small representative media run.

## Future work

Planned work is documented in [backend/Augmentify/FutureWork/README.md](backend/Augmentify/FutureWork/README.md). The folder is intentionally a home for future module boilerplates and module-level docstrings describing the intended implementation before code is added.

## Research status

This codebase is maintained as research software for experimentation and reproducibility. Results should be interpreted alongside the selected model, hardware, prompt, dataset, augmentation parameters, and keyframe budget. No license or publication-specific usage terms have been declared in this repository; consult the project owner before redistributing model weights or derived artifacts.
